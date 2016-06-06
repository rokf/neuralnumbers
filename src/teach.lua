
-- teach.lua

require 'torch'
require 'image'
require 'lfs'
require 'nn'

local conf = {
  folder = '../img/learning',
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
      table.insert(out, image[row][col])
    end
  end
  return out
end

function main()
  -- For each valid file in the learning folder
  for file in lfs.dir(conf.folder) do
    if file ~= '.' and file ~= '..' then
      local full_name = conf.folder .. '/' .. file
      local number = tonumber(string.sub(file,1,1)) -- Take the first character and convert it to a number
      local img = image.load(f, 1, 'byte') -- Load as a grayscale image

    end
  end
end
