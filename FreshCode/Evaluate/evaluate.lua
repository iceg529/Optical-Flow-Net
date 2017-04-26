require 'torch'
require 'hdf5'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require '../AvgEndPointError'

--local myFile = hdf5.open('../trainData2.h5', 'r')
--local concatData = myFile:read('/data8'):all()
--local im1 = concatData:sub(1,1,1,3)
--local im2 = concatData:sub(1,1,4,6)
--local flow = concatData:sub(1,1,7,8)
--myFile:close()

local img1 = image.load('../../../../FlowNet/dispflownet-release/data/FlyingChairs_examples/0000000-img0.ppm')
local img2 = image.load('../../../../FlowNet/dispflownet-release/data/FlyingChairs_examples/0000000-img1.ppm')

local caffePredFile = hdf5.open('../sampleForColorCoding2.h5', 'r')
local caffePredFlow = caffePredFile:read('/data'):all():cuda()
caffePredFile:close()

local sampleFile = hdf5.open('../sampleForColorCoding.h5', 'r')
local flow2 = sampleFile:read('/data'):all():cuda()
sampleFile:close()

local model2 = torch.load('../logFiles/flownet_50_Model.t7')
model2:evaluate()
local input = torch.cat(img1, img2, 1):cuda() -- change later to im1[1] im2[1]
local pred = model2:forward(input)

local module = nn.SpatialUpSamplingBilinear(4):cuda()
local predFinal = module:forward(pred:cuda())

print(flow2:size())
print(predFinal:size())
local lossFn = nn.AvgEndPointError():cuda()  --AvgEndPointError or MSECriterion
local errr = lossFn:forward(flow2, flow2)
print('Error between prediction and Ground truth  ' .. errr)

-- Save images, ground truth flow and predicted flow ----

--im1:div(255)
--im2:div(255)
--print(flow:size())
--print(pred:size())
--print(predFinal:size())
--print(flow[1][1][1][1])
--print(pred[1][1][1])
--print(pred:type())
--print(input:size())
--image.save('trainIm1.png',im1[1])
--image.save('trainIm2.png',im2[1])



--torch.save('flow_sample2.t7',flow)
--torch.save('flow_sample2_pred.t7',predFinal)
