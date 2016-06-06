
-- teach.lua

require 'torch'
require 'image'
require 'lfs'
require 'nn'

-- Stores configuration variables to make it a bit cleaner
local conf = {
  folder = '../img/learning',
  im_x = 32,
  im_y = 32,
  inputs = 32 * 32, -- The NN will have that many inputs (total number of pixels)
  outputs = 10, -- Numbers can be from 0-9
  hidden = 10, -- Number of hidden neurons
  iterations = 500, -- Number of epochs the trainer is going to train
  learning_rate = 0.01 -- The step of change per iteration
}

-- This function takes the image and returns its
-- pixel values joined together in a long table
function give_classifier(image)
  local image_size = image:size()
  local out = {}
  for row = 1, image_size[1] do
    for col = 1, image_size[2] do
      if image[row][col] == 255 then
        table.insert(out, 0) -- Background is white
      else
        table.insert(out, 1) -- The numbers are black
      end
    end
  end
  return out
end

function main()
  local dataset = {}
  -- For each valid file in the learning folder
  for file in lfs.dir(conf.folder) do
    if file ~= '.' and file ~= '..' then
      local full_name = conf.folder .. '/' .. file
      local number = tonumber(string.sub(file,1,1)) -- Take the first character and convert it to a number
      local img = image.load(full_name, 1, 'byte') -- Load as a grayscale image
      img = img[1]
      local classifier = give_classifier(img)
      local n_inputs = torch.Tensor(classifier) -- Construct input tensor
      local n_outputs = torch.Tensor(conf.outputs):zero() -- Construct output tensor
      n_outputs[number+1] = 1
      table.insert(dataset, {n_inputs, n_outputs}) -- Insert the input - output pair into the dataset table
    end
  end
  function dataset:size() return #dataset end -- The dataset constructed from a table needs a [size] method
  -- Now the dataset is ready to be used
  -- Lets prepare the neural network
  local model = nn.Sequential()
  model:add(nn.Linear(conf.inputs, conf.hidden)) -- This is the input linear layer
  model:add(nn.Tanh()) -- This is the hidden layer
  model:add(nn.Linear(conf.hidden, conf.outputs)) -- This is the output linear layer
  local criterion = nn.MSECriterion()
  local trainer = nn.StochasticGradient(model, criterion) -- This is the trainer which will teach the network
  trainer.maxIteration = conf.iterations
  trainer.learningRate = conf.learning_rate
  trainer:train(dataset) -- Train
  torch.save("model.dat", model) -- The model is now ready to be exported
end

main() -- Run the main function
