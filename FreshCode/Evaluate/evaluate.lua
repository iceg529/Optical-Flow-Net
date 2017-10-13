require 'torch'
require 'hdf5'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require '../AvgEndPointError'
require '../AvgAngularError'
require '../utils'
require 'xlua'

local myFile = hdf5.open('../trainData.h5', 'r')
local concatData = myFile:read('/data1'):all()
local im1 = concatData:sub(1,8,1,3):cuda()
local im2 = concatData:sub(1,8,4,6):cuda()
local flow2 = concatData:sub(1,8,7,8):cuda()
myFile:close()


local chns, lnt
local function getWeight(chns, lnt)
   local weights = torch.Tensor(chns,chns,lnt,lnt)
   local accum_weight = 0
   local ii, jj, tmpWt
   local scale = (lnt-1)/2
   ii = 1
   for i = -scale,scale,1 do
     jj = 1
     for j = -scale,scale,1 do
       tmpWt = (1 -(torch.abs(i)/(scale+1))) * (1 -(torch.abs(j)/(scale+1)))
       weights[{ {1},{1},ii,jj }] = tmpWt
       weights[{ {2},{2},ii,jj }] = tmpWt
       weights[{ {1},{2},ii,jj }] = 0
       weights[{ {2},{1},ii,jj }] = 0
       accum_weight = accum_weight + tmpWt
       jj = jj + 1
     end
     ii = ii + 1
   end
  
   weights:div(accum_weight)
   return weights
end

local meanFile = hdf5.open('../meanData.h5','r')
meanData = meanFile:read('/data'):all():cuda()     
meanFile:close()

local img1 = torch.Tensor(1,3,384,512)
local img2 = torch.Tensor(1,3,384,512)
local fflow = torch.Tensor(1,2,384,512)

local caffePredFile = hdf5.open('../sampleForColorCoding2.h5', 'r')
local caffePredFlow = caffePredFile:read('/data'):all():cuda()
caffePredFile:close()

local sampleFile = hdf5.open('../sampleForColorCoding.h5', 'r')
img1[1]:copy(sampleFile:read('/data1'):all()):cuda()
img2[1]:copy(sampleFile:read('/data2'):all()):cuda()
fflow[1]:copy(sampleFile:read('/data3'):all()):cuda() --for residual

--[[img1[1]:copy(im1[1]):cuda()
img2[1]:copy(im2[1]):cuda()
fflow[1]:copy(flow2[1]):cuda()--]] --for residual

local flow = sampleFile:read('/data3'):all():cuda()
local epicflow = sampleFile:read('/data4'):all():cuda()
sampleFile:close()

--uncomment these
--img1[1]:copy(im1[1])
--img2[1]:copy(im2[1])
--flow:copy(flow2[1])

--local model2 = torch.load('../logFiles/flownetLC6_LR3_165_Model.t7') --flownetLC_LR3_200_Model.t7
local model2 = torch.load('../logFiles/residual/flownetLC1_LR3_121_Model.t7')
model2:evaluate()


img1, img2 = normalizeMean(meanData, img1, img2)

local input = torch.cat(img1, img2, 2):cuda() -- change later to im1[1] im2[1]
local pred = model2:forward(input)

local module = nn.SpatialUpSamplingBilinear(4):cuda()
local predFinal = module:forward(pred:cuda())
local predZero = torch.Tensor(2,384,512):fill(0.2):cuda()
print(torch.min(pred))
print(torch.max(pred))

local lossFn = nn.AvgEndPointError():cuda()  --AvgEndPointError or MSECriterion or AvgAngularError

local temp = torch.Tensor(2,384,512):copy(flow)
local down5 =torch.Tensor(2,96,128)
local ti =-3
local tj =-3
for i = 1, 96 ,1 do
  ti = ti+4
  tj = -3
  for j = 1, 128 ,1 do
    tj = tj+4
    down5[1][i][j] =  temp[1][ti][tj]
    down5[2][i][j] =  temp[2][ti][tj]
  end
end

local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) -- nn.SpatialConvolution(2,2,1, 1, 4, 4, 0, 0)
mod.weight = getWeight(2, 7)
mod.bias = torch.Tensor(2):fill(0)
mod = mod:cuda()
local down5 = mod:forward(fflow:cuda()) -- flow fflow
local up5 = nn.SpatialUpSamplingBilinear({oheight=384, owidth=512}):cuda():forward(predFinal:cuda())

local pred2 = torch.CudaTensor(1,2,96,128) --bcos of changes in avg angular error for residual
pred2:copy(pred)
print(pred:size())
print(down5:size())
print(predFinal:size())
local errr = lossFn:forward(pred,down5) --caffePredFlow pred down5 predFinal flow
print('Error between prediction and Ground truth  ' .. errr)
--print(predFinal[{{},{1,3},{1,3}}])

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
torch.save('down.t7',predFinal[1]) --predFinal flow
