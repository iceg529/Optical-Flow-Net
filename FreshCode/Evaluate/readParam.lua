require 'torch'
require 'hdf5'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'

local model = torch.load('../logFiles/flownet_0.05_Model.t7')
local param,gradParam
param,gradParam = model:getParameters()
print(gradParam:max())
print(gradParam:min())
