require 'torch'
require 'hdf5'
require 'image'
require 'xlua'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require '../AvgEndPointError'
require '../AvgAngularError'
require '../utils'

local img1 = torch.Tensor(2,384,512)
local img2 = torch.Tensor(2,384,512)
local sampleFile = hdf5.open('../sampleForColorCoding.h5', 'r')
img1:copy(sampleFile:read('/data3'):all())
img2:copy(sampleFile:read('/data4'):all())

sampleFile:close()

local lossFn = nn.AvgAngularError():cuda() -- AvgAngularError AvgEndPointError
local errr = lossFn:forward(img2:cuda(), img1:cuda())
print('Error between prediction and Ground truth  ' .. errr)


