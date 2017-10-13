
require 'image'
require 'torch'
require 'xlua'
require 'pl'
require 'trepl'
require 'lfs'
require 'nn'
require 'cunn'
require 'cutorch'
require 'optim'
require 'AvgEndPointError' -- custom cost function
require 'CorrelationLayer'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
  package.path = dir_path .."?.lua;".. package.path
end

local opTtrain = 'trainData_Sintel.h5' --trainData.h5 trainData_16.h5 trainData_Sintel.h5
local opTval = 'testData_SintelClean.h5' --testData.h5 testData_SintelClean.h5
local opTDataMode = 'sintel' -- chair or sintel
local opTshuffle = false 
local opTthreads = 3 -- 3 1
local opTepoch = 1 
local opTsnapshotInterval = 10
local epIncrement = 50 --151, 0 
local opTsave = "logFiles/residual"  -- "logFiles/correlation" "logFiles/finetuning" , "logFiles", "logFiles/newWithoutReg"
local isTrain = true -- true false
local isCorr = false -- true false
profiler = xlua.Profiler(false, true)

require 'logmessage'

-- load utils
require 'utils'
----------------------------------------------------------------------

torch.setnumthreads(opTthreads)

----------------------------------------------------------------------
-- Open Data sources:
-- training database 
-- optionally: validation database 

require 'data'
if isCorr then 
  require 'modelCorr'
else
  require 'modelRes' -- 'model'
  require 'modelResSplit1'
  require 'modelResSplit2'
end

-- DataLoader objects take care of loading data from
-- databases. Data loading and tensor manipulations
-- (e.g. cropping, mean subtraction, mirroring) are
-- performed from separate threads
local trainDataLoader, trainSize, inputTensorShape
local valDataLoader, valSize

local num_threads_data_loader = opTthreads
local valData = {}

local meanFile = hdf5.open('meanDataSintel.h5','r')  --meanData.h5
local meanData2 = meanFile:read('/data'):all()    
meanFile:close()
meanFile = hdf5.open('meanData.h5','r')
local meanData1 = meanFile:read('/data'):all()     
meanFile:close()

if opTDataMode == 'sintel' then
	meanData = fuseMean(meanData1,meanData2) 
	--or following lines if no fuse wanted and mean1 wanted
	--[[meanData = torch.Tensor(4,3,436,1024) 
	meanData[1] = image.scale(meanData1[1],1024,436)
	meanData[2] = image.scale(meanData1[2],1024,436)
	meanData[3] = image.scale(meanData1[3],1024,436)
	meanData[4] = image.scale(meanData1[4],1024,436)--]]
elseif opTDataMode == 'chair' then
	meanData = meanData1 
end

if isTrain then
	-- create data loader for training dataset
	trainDataLoader = DataLoader:new(
	      num_threads_data_loader, -- num threads
	      package.path,
	      opTtrain,
	      true, -- train
	      opTshuffle
	)
	-- retrieve info from train DB (number of records and shape of input tensors)
	trainSize, inputTensorShape = trainDataLoader:getInfo()
	logmessage.display(0,'found ' .. trainSize .. ' images in train db' .. opTtrain)
else
	---- create data loader for validation dataset
	valDataLoader = DataLoader:new(
	      1, -- num threads
	      package.path,
	      opTval,
	      false, -- train
	      opTshuffle
	)
	---- retrieve info from train DB (number of records and shape of input tensors)
	valSize, valInputTensorShape = valDataLoader:getInfo()
	logmessage.display(0,'found ' .. valSize .. ' images in train db' .. opTval)
	
	if valDataLoader:acceptsjob() then
	  valDataLoader:scheduleNextBatch(valSize, 1, valData, true) -- 640 , 137*1   val data size
	end
	valDataLoader:waitNext()
        valData.im1, valData.im2 = normalizeMean(meanData, valData.im1, valData.im2)
end

local downSampleFlowWeights = getWeight(2, 7) -- convolution weights , args: 2 - num of channels , 7 - conv filter size
downSampleFlowWeights:cuda()


-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
--local model = getModel()
--local model = require('weight-init')(getResModel(), 'kaiming')
--local model = torch.load('logFiles/residual/flownetLC1_LR3_' .. epIncrement .. '_Model.t7') -- this is the model before finetuning with sintel
local model = torch.load('logFiles/residual/res2/flownetLC1_LR3_' .. epIncrement .. '_Model.t7') -- for res2 (addition of multiple disp)

model = model
local model2_1 = getResModel1():cuda()
if model then
   --[[
   j = 2
   for i=2,model:size() do
       if i<40 then
	 model2_1:get(i).parameters = model:get(i).parameters
	 model2_1:get(i).accGradParameters = model:get(i).accGradParameters
       end
   end
   --]]

   modelParam,modelGradParam = model:getParameters()
   modelParam2,modelGradParam2 = model2_1:getParameters()
   modelParam2:copy(modelParam[{{1,modelParam2:size()[1]}}])
end



-- if batch size was not specified on command line then check
-- whether the network defined a preferred batch size (there
-- can be separate batch sizes for the training and validation
-- sets)

local trainBatchSize = 8 --8 16
local logging_check = trainSize --11116 -- 8 splits of training data(22232/8)
local next_snapshot_save = 0.3
local snapshot_prefix = 'flownet'

----------------------
-- epoch value will be calculated for every batch size. To maintain unique epoch value between batches, it needs to be rounded to the required number of significant digits.
local epoch_round = 0 -- holds the required number of significant digits for round function.
local tmp_batchsize = trainBatchSize
while tmp_batchsize <= trainSize do
    tmp_batchsize = tmp_batchsize * 10
    epoch_round = epoch_round + 1
end
logmessage.display(0,'While logging, epoch value will be rounded to ' .. epoch_round .. ' significant digits')

------------------------------
local loggerCnt = 0
local epoch = 1
local actualEp = epoch + epIncrement --+ 170

logmessage.display(0,'started training the model')
local config = {learningRate = (0.0001), --0.0001 0.1 0.000001 0.0001/16(for augment, since divided at earlier epochs in below lines)
	           weightDecay = 0.0004, --0.0004 0
	           momentum = 0.9,
	           learningRateDecay = 0 }--3e-5	

while epoch<=opTepoch do
  local time = sys.clock()  
  ------------------------------
  local NumBatches = 0
  local curr_images_cnt = 0
  local loss_sum = 0
  local loss_batches_cnt = 0
  local learningrate = 0
  local im1, im2, flow
  local dataLoaderIdx = 1
  local data = {}
  local input, flowInput
  --trainSize = 22232  -- 22232 , 22224(to be multiple of 16)
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. trainBatchSize .. ']')
  
  --this bit for normal training
  --[[if actualEp == 146 or actualEp == 150 or actualEp == 160 or actualEp == 166 or actualEp == 170 or actualEp == 175 then
    config.learningRate = (config.learningRate)/(2.0)
  end--]] --for without augmentation

  --this bit for augmentation
  --[[if actualEp == 130 or actualEp == 150 or actualEp == 170 or actualEp == 190 then
    config.learningRate = (config.learningRate)/(2.0)
  end--]]
  
  local t = 1
  local cnt = 1
  while t <= trainSize do --trainSize
    --model:training()
    -- disp progress
    xlua.progress(t, trainSize)
    local time2 = sys.clock()
    profiler:start('pre-fetch')
    -- prefetch data thread
    --------------------------------------------------------------------------------
    while trainDataLoader:acceptsjob() do      
      local dataBatchSize = math.min(trainSize-dataLoaderIdx+1,trainBatchSize)
      if dataBatchSize > 0 and dataLoaderIdx < math.floor(trainSize/trainBatchSize)+1 then   --dataLoaderIdx < 2780 .. depends on batch size , (22232/batchSize)+1
	trainDataLoader:scheduleNextBatch(dataBatchSize, dataLoaderIdx, data, true)
	dataLoaderIdx = dataLoaderIdx + 1 --dataBatchSize
      else break end
    end
    NumBatches = NumBatches + 1

    -- wait for next data loader job to complete
    trainDataLoader:waitNext()
    --------------------------------------------------------------------------------
    profiler:lap('pre-fetch')
    -- get data from last load job
    local thisBatchSize = data.batchSize
    im1 = torch.Tensor(data.im1:size())
    im2 = torch.Tensor(data.im2:size())
    flow = torch.Tensor(data.flow:size())
    im1:copy(data.im1)
    im2:copy(data.im2)
    flow:copy(data.flow)
    ----- mean normalization -------------
    im1, im2 = normalizeMean(meanData, im1, im2)
    
    local tmpValImg1 = torch.Tensor(im1:size(1),im1:size(2),448, 1024)
    local tmpValImg2 = torch.Tensor(im2:size(1),im2:size(2),448, 1024)
    local tmpValFlow = torch.Tensor(flow:size(1),flow:size(2),448, 1024)
    if opTDataMode == 'chair' then
      input = torch.cat(im1, im2, 2)
      flowInput = flow
    elseif opTDataMode == 'sintel' then
      for i = 1,im1:size(1) do 
        tmpValImg1[i] = image.scale(im1[i],1024,448)
        tmpValImg2[i] = image.scale(im2[i],1024,448) 
        tmpValFlow[i] = image.scale(flow[i],1024,448)
      end
      midInput = torch.cat(tmpValImg1, tmpValImg2, 2)
      input = model2_1:forward(midInput:cuda())	      
      flowInput =  tmpValFlow
    end
    t = t + thisBatchSize	    
    print('The data loaded till index ' .. data.indx)
    print(input:size())
    torch.save('sintelFeat/sintelFeatures' .. cnt .. '.t7',input) -- cnt+226
    torch.save('sintelFeat/flow' .. cnt .. '.t7',flowInput) -- cnt+226
    cnt = cnt + 1
    
    if math.fmod(NumBatches,10)==0 then
      collectgarbage()
    end
  end
  
  ------------------------------
  -- time taken
  time = sys.clock() - time
  --time = time / trainSize
  print("==> time to learn for 1 epoch = " .. (time) .. 's')
   
  epoch = epoch+1
  actualEp = actualEp+1
end
	

-- enforce clean exit
os.exit(0)
