
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

local opTtrain = 'trainData.h5' --trainData.h5 trainData_16.h5 trainData_Sintel.h5
local opTval = 'testData.h5' --testData.h5 testData_SintelClean.h5
local opTDataMode = 'chair' -- chair or sintel
local opTshuffle = false 
local opTthreads = 3
local opTepoch = 10
local epIncrement =  0-- 5, 0, 90, 240, 260 ...
local opTsave = "logFiles/testTimeTraining"  -- "logFiles/finetuning" , "logFiles", "logFiles/newWithoutReg"
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
  require 'model'
end

-- DataLoader objects take care of loading data from
-- databases. Data loading and tensor manipulations
-- (e.g. cropping, mean subtraction, mirroring) are
-- performed from separate threads
local trainDataLoader, valSize, inputTensorShape
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


local downSampleFlowWeights = getWeight(2, 7) -- convolution weights , args: 2 - num of channels , 7 - conv filter size
downSampleFlowWeights:cuda()

----------------------------------------------------------------------

-- Log results to files
trainLogger = optim.Logger(paths.concat(opTsave, 'train.log'))
trainLogger:setNames{'Training error', 'Validation error'}
trainLogger:style{'+-', '+-'}
trainLogger:display(false)


-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
--local model = getModel()
--local model = require('weight-init')(getModel(), 'kaiming')
--local model = torch.load('logFiles/flownetLC6_LR3_5_Model.t7') -- this is the base model for most
--local model = torch.load('logFiles/newWithoutReg/flownetLC8_LR3_240_Model.t7') -- or LC8_LR3_260_Model , one of the base model of augmented models
local model = torch.load('logFiles/finetuning/flownetLC9_LR3_130_Model(setup6,noAugafter100).t7') -- 'logFiles/flownetLC6_LR3_180_Model.t7' this is the model before finetuning with sintel

model = model:cuda()
local criterion = nn.MSECriterion() --SmoothL1Criterion L1HingeEmbeddingCriterion(
criterion = criterion:cuda()
local criterion2 = nn.AvgEndPointError() --SmoothL1Criterion L1HingeEmbeddingCriterion(
criterion2 = criterion2:cuda()
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------


-- if batch size was not specified on command line then check
-- whether the network defined a preferred batch size (there
-- can be separate batch sizes for the training and validation
-- sets)

local trainBatchSize = 1 --8 16
local logging_check = valSize --11116 -- 8 splits of training data(22232/8)
local next_snapshot_save = 0.3
local snapshot_prefix = 'flownet'

----------------------
-- epoch value will be calculated for every batch size. To maintain unique epoch value between batches, it needs to be rounded to the required number of significant digits.
local epoch_round = 0 -- holds the required number of significant digits for round function.
local tmp_batchsize = trainBatchSize
while tmp_batchsize <= valSize do
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

-- get data from last load job
local thisBatchSize = 1 -- valData.batchSize
local im1, im2, flow
im1 = torch.Tensor(1,valData.im1:size(2),valData.im1:size(3),valData.im1:size(4))
im2 = torch.Tensor(1,valData.im1:size(2),valData.im1:size(3),valData.im1:size(4))
flow = torch.Tensor(1,2,valData.im1:size(3),valData.im1:size(4))
im1[1]:copy(valData.im1[1])
im2[1]:copy(valData.im2[1])
flow[1]:copy(valData.flow[1])
local output, predFinal

while epoch<=opTepoch do
  local time = sys.clock()  
  ------------------------------
  local NumBatches = 0
  local curr_images_cnt = 0
  local loss_sum = 0
  local loss_batches_cnt = 0
  local learningrate = 0
  local dataLoaderIdx = 1
  local data = {}
  valSize = 1  -- comment later
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. trainBatchSize .. ']')
  
  local t = 1
  
  while t <= valSize do --valSize
    --model:training()
    -- disp progress
    xlua.progress(t, valSize)
    local time2 = sys.clock()

    profiler:start('training process')
    ------------------------------------------------------------------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
	parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0
      local tmpIdentityU = torch.Tensor(flow[1]:size())
      profiler:start('feval process')
      -- evaluate function for complete mini batch
      for i = 1,im1:size(1) do
	-- estimate f		
	local input, tmpImg1, tmpImg2
	-- resized for sintel dataset (orig size 1024, 436) shud be multiple of 32 to avoid size issue at nn.JoinTable(), remember to change later
    	if opTDataMode == 'sintel' then
	  tmpImg1 = torch.Tensor(1,im1[i]:size(1),448,im1[i]:size(3)):cuda()
      	  tmpImg2 = torch.Tensor(tmpImg1:size()):cuda()
      	  tmpImg1[1] = image.scale(im1[i],1024,448)
      	  tmpImg2[1] = image.scale(im2[i],1024,448)
	  input = torch.cat(tmpImg1[1], tmpImg2[1], 1)
	elseif opTDataMode == 'chair' then
	  input = torch.cat(im1[i], im2[i], 1)
	end	
	input = input:cuda()		
	if isCorr then
	  output = model:forward({input:sub(1,3), input:sub(4,6)})		
	else
	  output = model:forward(input)
	end

	------------------ usual training with flow data -------------------------------

	--[[local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) -- nn.SpatialConvolution(2,2,1, 1, 4, 4, 0, 0)
	mod.weight = downSampleFlowWeights
	mod.bias = torch.Tensor(2):fill(0)
	mod = mod:cuda()
	local down5 = mod:forward(flow[i]:cuda())
	--print('aft model fwd')
	down5 = down5:cuda()
	local err = criterion:forward(output, down5) --grdTruth
	f = f + err
	-- estimate df/dW
	local df_do = criterion:backward(output, down5) --grdTruth
	model:backward(input, df_do)--]]
	------------------------------------------------------------
	
	local module2 = nn.SpatialUpSamplingBilinear(4):cuda()
        predFinal = module2:forward(output:cuda())
	local warpedImg = (image.warp(im1[i],predFinal:double())):cuda()
		     
	local err = criterion:forward(warpedImg, im2[i]:cuda()) 
	f = f + err
	-- estimate df/dW
	local df_do = criterion:backward(warpedImg, im2[i]:cuda()) 
	-----	
	tmpIdentityU:copy(predFinal)
	tmpIdentityU[1]:add(1)
	local warpedImg2 = (image.warp(im1[i],tmpIdentityU:double())):cuda()
	local dw_du = torch.add(warpedImg2,-warpedImg)
	local df_du_temp = torch.CudaTensor(2,warpedImg:size(1),warpedImg:size(2),warpedImg:size(3))
	df_du_temp[1] = torch.cmul(dw_du,df_do)
	-----	
	tmpIdentityU:copy(predFinal)
	tmpIdentityU[2]:add(1)
	warpedImg2 = (image.warp(im1[i],tmpIdentityU:double())):cuda()
	dw_du = torch.add(warpedImg2,-warpedImg)
	df_du_temp[2] = torch.cmul(dw_du,df_do)
	local df_du = torch.CudaTensor(predFinal:size())
	df_du = torch.add(df_du_temp:select(2,1),df_du_temp:select(2,2),df_du_temp:select(2,3))
	local shiftedGrad = torch.CudaTensor(df_du:size()):copy(df_du)
	local index = torch.Tensor(2)
	for m=1,predFinal:size(2) do
	  for n=1,predFinal:size(3) do
	    local flowField = predFinal[{{},m,n}]
	    index[1] = (m+flowField[1])>0 and (m+flowField[1]) or 0.1
	    index[1] = (m+flowField[1])<=predFinal:size(2) and index[1] or predFinal:size(2)
	    index[2] = (n+flowField[2])>0 and (n+flowField[2]) or 0.1
	    index[2] = (n+flowField[2])<=predFinal:size(3) and index[2] or predFinal:size(3)
	    --print(index[1])
	    --print(index[2])  
	    --print(df_du[{{},m+flowField[1],n+flowField[2]}])
	    shiftedGrad[{{},m,n}]:copy(df_du[{{},index[1],index[2]}])
	  end
	end
	----------------
	local df_du2 = module2:backward(output:cuda(),shiftedGrad)
	model:backward(input, df_du2)
	  
    	local actualErr = criterion2:forward(predFinal,flow[1]:cuda())
    	print('Actual Err ' .. actualErr)
      end

      -- normalize gradients and f(X)
      gradParameters:div(im1:size(1))
      f = f/(im1:size(1))
      print(f)

      profiler:lap('feval process')
      -- return f and df/dX
      return f,gradParameters
    end    		         
    _, train_err = optim.adam(feval, parameters, config)
    -----------------------------------------------------------------------------------------------------------------------------
    profiler:lap('training process')

    -------------------------------------BLOCK TO CHECK LATER---------------------------------------------------------------------    
    profiler:start('logging process')
    -- adding the loss values of each mini batch and also maintaining the counter for number of batches, so that average loss value can be found at the time of logging details
    loss_sum = loss_sum + train_err[1]
    loss_batches_cnt = loss_batches_cnt + 1
    
    local current_epoch = (epoch-1)+round((math.min(t+trainBatchSize-1,valSize))/valSize, epoch_round)
    
    print(current_epoch)
    
    -- log details on first iteration, or when required number of images are processed
    curr_images_cnt = curr_images_cnt + thisBatchSize
    
    -- update logger/plot
        
    local avgLoss = loss_sum / loss_batches_cnt      
    local avgValErr = 0 --validation(model, valData, criterion, downSampleFlowWeights)

    trainLogger:add{avgLoss, avgValErr}
    trainLogger:plot()

    --logmessage.display(0, 'Training (epoch ' .. current_epoch)
    if (epoch==1 and t==1) then 
      curr_images_cnt = thisBatchSize
    else
      curr_images_cnt = 0 -- For accurate values we may assign curr_images_cnt % logging_check to curr_images_cnt, instead of 0
      loss_sum = 0
      loss_batches_cnt = 0
    end	      
  
    -------------------------------------------------------------------------------------------------------------------------------
    
    t = t + thisBatchSize
    profiler:lap('logging process')    
    collectgarbage()
  end
  ------------------------------
  -- time taken
  time = sys.clock() - time
  --time = time / valSize
  print("==> time to learn for 1 epoch = " .. (time) .. 's')
   
  epoch = epoch+1
  actualEp = actualEp+1
end  
torch.save(paths.concat(opTsave, 'down.t7'),predFinal)	

-- close databases
-- trainDataLoader:close() uncomment if needed

-- enforce clean exit
os.exit(0)
