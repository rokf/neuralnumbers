
-- recognize.lua

require 'torch'
require 'image'
require 'lfs'
require 'nn'

local conf = {
  folder = '../img/remaining',
  im_x = 32,
  im_y = 32
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
  local model = torch.load("model.dat") -- Load the network back in
  for file in lfs.dir(conf.folder) do
    if file ~= '.' and file ~= '..' then
      local full_name = conf.folder .. '/' .. file
      local number = tonumber(string.sub(file,1,1)) -- Take the first character and convert it to a number
      local img = image.load(full_name, 1, 'byte') -- Load as a grayscale image
      img = img[1]
      local classifier = give_classifier(img)
      local output = model:updateOutput(torch.Tensor(classifier))
      local maximum = torch.max(output)
      for i=1, 10 do -- [output] has 10 numbers, the index with the maximum value should be the number
        if output[i] == maximum then
          print(string.format("Should be %d, is %d, [%s]", number, i-1, tostring(number == (i-1))))
        end
      end
    end
  end
end

main()
