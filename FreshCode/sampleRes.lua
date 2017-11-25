
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
local opTval = 'testData_SintelClean.h5' --testData.h5 testData_SintelClean.h5
local opTDataMode = 'sintel' -- chair or sintel
local opTshuffle = false 
local opTthreads = 3 -- 3 1
local opTepoch = 25
local opTsnapshotInterval = 5
local epIncrement = 20 --130 --85 --110 -- 50 for res2 -- 151 for simple residual before res2 
local opTsave = "logFiles/residual/plain"  -- "logFiles/correlation" "logFiles/finetuning" , "logFiles", "logFiles/newWithoutReg"
local isTrain = false -- true false
local isPlain = true -- true false
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
if isPlain then 
  require 'modelPlain'
else
  require 'modelRes' -- 'model'
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
-- validation function
local function validation(model,valData,criterion,flowWeights,opTDataMode)
  --model:evaluate()
  local valErr = 0
  local input, flInput, tmpValImg1, tmpValImg2, tmpValFlow
  if opTDataMode == 'chair' then
    input = torch.Tensor(1,2*valData.im1:size(2),valData.im1:size(3),valData.im1:size(4))
    flInput = torch.Tensor(1,2,valData.im1:size(3),valData.im1:size(4))
  elseif opTDataMode == 'sintel' then
    input = torch.Tensor(1,2*valData.im1:size(2),448,1024)
    flInput = torch.Tensor(1,2,448,1024)
  end

  for i = 1,valData.im1:size(1) do    
    if opTDataMode == 'chair' then
	input[1] = torch.cat(valData.im1[i], valData.im2[i], 1) -- uncomment for chairs data validation and comment for sintel
	flInput[1] =  valData.flow[i] -- valData.flow[i] for chairs data validation or tmpValFlow for sintel 
    elseif opTDataMode == 'sintel' then
	-- comment for chairs data validation and uncomment for sintel
	tmpValImg1 = image.scale(valData.im1[i],1024,448)
    	tmpValImg2 = image.scale(valData.im2[i],1024,448) 
    	tmpValFlow = image.scale(valData.flow[i],1024,448)
    	input[1] = torch.cat(tmpValImg1, tmpValImg2, 1)
	flInput[1] =  tmpValFlow -- valData.flow[i] for chairs data validation or tmpValFlow for sintel 
    end
    input = input:cuda()
    flInput = flInput:cuda()
    local output = model:forward(input)
    
    local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) -- nn.SpatialConvolution(2,2,1, 1, 4, 4, 0, 0)
    mod.weight = flowWeights
    mod.bias = torch.Tensor(2):fill(0)
    mod = mod:cuda()
    local down5 = mod:forward(flInput)
    down5 = down5:cuda()
    
    local module = nn.SpatialUpSamplingBilinear(4):cuda()
    local predFi = module:forward(output)
    --local down5 = nn.SpatialAdaptiveAveragePooling(128, 96):cuda():forward(flInput)
    local err = criterion:forward(output, down5)
    valErr = valErr + err
    if i == valData.im1:size(1) then    
	print('model validated ' .. i)
    end
  end
  valErr = valErr / valData.im1:size(1)
  collectgarbage()
  return valErr
end
----------------------------------------------------------------------

-- Log results to files
trainLogger = optim.Logger(paths.concat(opTsave, 'train.log'))
trainLogger:setNames{'Training error', 'Validation error'}
trainLogger:style{'+-', '+-'}
trainLogger:display(false)

valLogger = optim.Logger(paths.concat(opTsave, 'validation.log'))
valLogger:setNames{'Validation error'}
valLogger:style{'+-'}
valLogger:display(false)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
--local model = getResModel()
--local model = require('weight-init')(getResModel(), 'kaiming')
--local model = torch.load('logFiles/flownetLC6_LR3_5_Model.t7') -- this is the base model for most
--local model = torch.load('logFiles/newWithoutReg/flownetLC8_LR3_240_Model.t7') -- or LC8_LR3_260_Model , one of the base model of augmented models

--local model = torch.load('logFiles/residual/res2/flownetLC1_LR3_' .. epIncrement .. '_Model.t7') -- res2
--local model = torch.load('logFiles/residual/res3/flownetLC1_LR3_' .. epIncrement .. '_Model.t7') -- res3
--local model = torch.load('logFiles/residual/res4/flownetLC1_LR3_' .. epIncrement .. '_Model.t7') -- res4
local model = torch.load('logFiles/residual/plain/flownetLC1_LR3_' .. epIncrement .. '_Model.t7') -- res4

model = model:cuda()
local criterion = nn.AvgEndPointError() --SmoothL1Criterion AvgEndPointError
criterion = criterion:cuda()
if model then
   --[[for i=1, (model:size() - 10) do
     model:get(i).parameters = function() return nil end -- freezes the layer when using optim 
     model:get(i).accGradParameters = function() end -- overwrite this to reduce computations
   end--]]
   parameters,gradParameters = model:getParameters()
end


----------------------
local function saveModel(model, directory, prefix, epoch)
    local filename
    local modelObjectToSave
    if model.clearState then
        -- save the full model
        filename = paths.concat(directory, prefix .. 'LC1_LR3_' .. epoch + epIncrement .. '_Model.t7')  --epoch + 170 LC:Learning Curve  LC_LR3_
        modelObjectToSave = model:clearState()
    else
        -- this version of Torch doesn't support clearing the model state => save only the weights
        local Weights,Gradients = model:getParameters()
        filename = paths.concat(directory, prefix .. '_' .. epoch .. '_Weights.t7')
        modelObjectToSave = Weights
    end
    logmessage.display(0,'Snapshotting to ' .. filename)
    torch.save(filename, modelObjectToSave)
    logmessage.display(0,'Snapshot saved - ' .. filename)
end

----------------------

if isTrain then
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
	    
	    local tmpValImg1 = torch.Tensor(im1:size(1),im1:size(2),384, 512)
	    local tmpValImg2 = torch.Tensor(im2:size(1),im2:size(2),384, 512)
	    local tmpValFlow = torch.Tensor(flow:size(1),flow:size(2),384, 512)
	    if opTDataMode == 'chair' then
	      input = torch.cat(im1, im2, 2)
	      flowInput = flow
 	    elseif opTDataMode == 'sintel' then
	      for i = 1,im1:size(1) do 
	        tmpValImg1[i] = image.scale(im1[i],512,384)
    	        tmpValImg2[i] = image.scale(im2[i],512,384) 
    	        tmpValFlow[i] = image.scale(flow[i],512,384)
	      end
    	      input = torch.cat(tmpValImg1, tmpValImg2, 2)
	      flowInput =  tmpValFlow
 	    end
	    --[[im1 = im1:cuda()
	    im2 = im2:cuda()
	    flow = flow:cuda()--]]
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
              profiler:start('feval process')
	      local output
	      -- remove from this line till above for loop
	      flowInput = flowInput:cuda()
              input = input:cuda()		
	      --if isCorr then
		--output = model:forward({input:sub(1,3), input:sub(4,6)})		
	      
 	      output = model:forward(input)
		
	      local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) -- nn.SpatialConvolution(2,2,1, 1, 4, 4, 0, 0)
	      mod.weight = downSampleFlowWeights
	      mod.bias = torch.Tensor(2):fill(0)
	      mod = mod:cuda()
	      local down5 = mod:forward(flowInput)
		--print('aft model fwd')
	      down5 = down5:cuda()
		      --local grdTruth = flow[i]:cuda()
	      local err = criterion:forward(output, down5) --grdTruth
	      f = f + err
		-- estimate df/dW
	      local df_do = criterion:backward(output, down5) --grdTruth
	      model:backward(input, df_do)
	      -- evaluate function for complete mini batch
	      --[[for i = 1,im1:size(1) do
		-- estimate f		
		local input, tmpImg1, tmpImg2, tmpFlow, flowInput
		-- resized for sintel dataset (orig size 1024, 436) shud be multiple of 32 to avoid size issue at nn.JoinTable(), remember to change later
	    	if opTDataMode == 'sintel' then
		  tmpImg1 = torch.Tensor(1,im1[i]:size(1),448,im1[i]:size(3)):cuda()
	      	  tmpImg2 = torch.Tensor(tmpImg1:size()):cuda()
		  tmpFlow = torch.Tensor(1,2,448,im1[i]:size(3)):cuda()
	      	  tmpImg1[1] = image.scale(im1[i],1024,448)
              	  tmpImg2[1] = image.scale(im2[i],1024,448)
		  tmpFlow[1] = image.scale(flow[i],1024,448)
		  input = torch.cat(tmpImg1[1], tmpImg2[1], 1)
		  flowInput = tmpFlow[1]:cuda()
		elseif opTDataMode == 'chair' then
 		  input = torch.cat(im1[i], im2[i], 1)
		  flowInput = flow[i]:cuda()
		end	
		input = input:cuda()		
		if isCorr then
		  output = model:forward({input:sub(1,3), input:sub(4,6)})		
		else
 		  output = model:forward(input)
		end
		
		local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) -- nn.SpatialConvolution(2,2,1, 1, 4, 4, 0, 0)
		mod.weight = downSampleFlowWeights
		mod.bias = torch.Tensor(2):fill(0)
		mod = mod:cuda()
		local down5 = mod:forward(flowInput)
		--print('aft model fwd')
		down5 = down5:cuda()
		      --local grdTruth = flow[i]:cuda()
		local err = criterion:forward(output, down5) --grdTruth
		f = f + err
		-- estimate df/dW
		local df_do = criterion:backward(output, down5) --grdTruth
		model:backward(input, df_do)
		--print(torch.min(tmpImg1[1])) --output[1] im1[i]
		--print(torch.max(tmpImg1[1]))
	      end --]]

	      -- normalize gradients and f(X)
	      --gradParameters:div(im1:size(1)) --uncomment 
	      --f = f/(im1:size(1)) --uncomment
	      print(f)
	      profiler:lap('feval process')
	      -- return f and df/dX
	      return f,gradParameters
	    end
	    
	    --if actualEp == 6 then
	     --config.learningRate = 0.0001
	      --config.weightDecay = 0.0004
            --end   
    
            --print(config.learningRate)
	    --print((config.learningRate)/(2.0))
	    -- optimize on current mini-batch
	    		         
	    _, train_err = optim.adam(feval, parameters, config)
	    -----------------------------------------------------------------------------------------------------------------------------
	    profiler:lap('training process')

	    -------------------------------------BLOCK TO CHECK LATER---------------------------------------------------------------------    
	    profiler:start('logging process')
	    -- adding the loss values of each mini batch and also maintaining the counter for number of batches, so that average loss value can be found at the time of logging details
	    loss_sum = loss_sum + train_err[1]
	    loss_batches_cnt = loss_batches_cnt + 1
	    
	    local current_epoch = (epoch-1)+round((math.min(t+trainBatchSize-1,trainSize))/trainSize, epoch_round)
	    
	    print(current_epoch)
	    
	    print(loggerCnt)
	    -- log details on first iteration, or when required number of images are processed
	    curr_images_cnt = curr_images_cnt + thisBatchSize
	    
	    -- update logger/plot
	    if (epoch==1 and t==1) or curr_images_cnt >= logging_check then      
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
	      loggerCnt = loggerCnt + 1
	    end

	    if current_epoch >= next_snapshot_save then
	      saveModel(model, opTsave, snapshot_prefix, current_epoch)
	      next_snapshot_save = (round(current_epoch/opTsnapshotInterval) + 1) * opTsnapshotInterval -- To find next epoch value that exactly divisible by opt.snapshotInterval
	      last_snapshot_save_epoch = current_epoch
	    end
	    -------------------------------------------------------------------------------------------------------------------------------
	    
	    t = t + thisBatchSize
	    profiler:lap('logging process')
	    print('The data loaded till index ' .. data.indx)
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

	-- if required, save snapshot at the end
	saveModel(model, opTsave, snapshot_prefix, opTepoch)
else
  local models = {'residual/res4/flownetLC1_LR3_10_Model','residual/res4/flownetLC1_LR3_20_Model','residual/plain/flownetLC1_LR3_10_Model','residual/plain/flownetLC1_LR3_20_Model','residual/plain/flownetLC1_LR3_25_Model','residual/plain/flownetLC1_LR3_30_Model','residual/plain/flownetLC1_LR3_35_Model','residual/plain/flownetLC1_LR3_40_Model','residual/plain/flownetLC1_LR3_45_Model'}

  --local models = {'residual/res4/flownetLC1_LR3_10_Model','residual/res4/flownetLC1_LR3_20_Model','residual/res4/flownetLC1_LR3_30_Model','residual/res3/flownetLC1_LR3_10_Model','residual/res3/flownetLC1_LR3_20_Model'}	

--local models = {'residual/res4/flownetLC1_LR3_70_Model','residual/res4/flownetLC1_LR3_75_Model','residual/res4/flownetLC1_LR3_80_Model','residual/res4/flownetLC1_LR3_85_Model','residual/res4/flownetLC1_LR3_100_Model','residual/res4/flownetLC1_LR3_120_Model','residual/res4/flownetLC1_LR3_130_Model','residual/res4/flownetLC1_LR3_135_Model','residual/res4/flownetLC1_LR3_140_Model','residual/res4/flownetLC1_LR3_145_Model','residual/res4/flownetLC1_LR3_155_Model','residual/res4/flownetLC1_LR3_160_Model','residual/res4/flownetLC1_LR3_165_Model','residual/res4/flownetLC1_LR3_170_Model','residual/res3/flownetLC1_LR3_60_Model','residual/res3/flownetLC1_LR3_65_Model', 'residual/res3/flownetLC1_LR3_70_Model','residual/res3/flownetLC1_LR3_75_Model','residual/res3/flownetLC1_LR3_130_Model', 'residual/res3/flownetLC1_LR3_130_Model(noAugAft100)'}

--local models = {'residual/res2/flownetLC1_LR3_40_Model','residual/res2/flownetLC1_LR3_50_Model','residual/res3/flownetLC1_LR3_10_Model', 'residual/res3/flownetLC1_LR3_50_Model','residual/res3/flownetLC1_LR3_55_Model','residual/res3/flownetLC1_LR3_60_Model','residual/res3/flownetLC1_LR3_65_Model', 'residual/res3/flownetLC1_LR3_70_Model','residual/res3/flownetLC1_LR3_75_Model','residual/res3/flownetLC1_LR3_95_Model', 'residual/res3/flownetLC1_LR3_100_Model', 'residual/res3/flownetLC1_LR3_105_Model','residual/res3/flownetLC1_LR3_110_Model', 'residual/res3/flownetLC1_LR3_115_Model', 'residual/res3/flownetLC1_LR3_120_Model','residual/res3/flownetLC1_LR3_125_Model','residual/res3/flownetLC1_LR3_130_Model', 'residual/res3/flownetLC1_LR3_130_Model(noAugAft100)'}

--{'residual/res2/flownetLC1_LR3_10_Model', 'residual/res2/flownetLC1_LR3_20_Model','residual/res2/flownetLC1_LR3_30_Model', 'residual/res2/flownetLC1_LR3_40_Model','residual/res2/flownetLC1_LR3_50_Model', 'residual/res2/flownetLC1_LR3_60_Model', 'residual/res2/flownetLC1_LR3_70_Model', 'residual/res2/flownetLC1_LR3_80_Model', 'residual/res2/flownetLC1_LR3_90_Model'}


--{'residual/flownetLC1_LR3_151_Model', 'residual/flownetLC1_LR3_161_Model','residual/flownetLC1_LR3_171_Model', 'residual/flownetLC1_LR3_181_Model','residual/flownetLC1_LR3_191_Model', 'residual/flownetLC1_LR3_201_Model'}

--{'residual/flownetLC1_LR3_10_Model','residual/flownetLC1_LR3_30_Model','residual/flownetLC1_LR3_70_Model','residual/flownetLC1_LR3_80_Model','residual/flownetLC1_LR3_100_Model','residual/flownetLC1_LR3_120_Model','residual/flownetLC1_LR3_100_Model(aug_after70)','residual/flownetLC1_LR3_110_Model(aug_after70)','residual/flownetLC1_LR3_120_Model(aug_after70)','residual/flownetLC1_LR3_121_Model(noAug_aft120)','residual/flownetLC1_LR3_131_Model','residual/flownetLC1_LR3_141_Model','residual/flownetLC1_LR3_151_Model'}

-- ,'residual/flownetLC1_LR3_100_Model(aug_after70)','residual/flownetLC1_LR3_110_Model(aug_after70)','residual/flownetLC1_LR3_120_Model(aug_after70)','residual/flownetLC1_LR3_121_Model(noAug_aft120)'

	--{'finetuning/flownetLC9_LR3_100_Model(setup1)', 'finetuning/flownetLC9_LR3_90_Model(setup4)', 'finetuning/flownetLC9_LR3_100_Model', 'finetuning/flownetLC9_LR3_160_Model', 'finetuning/flownetLC9_LR3_180_Model', 'finetuning/flownetLC9_LR3_220_Model', 'finetuning/flownetLC9_LR3_240_Model', 'finetuning/flownetLC9_LR3_280_Model',  'finetuning/flownetLC9_LR3_110_Model(setup6,noAugafter100)', 'finetuning/flownetLC9_LR3_120_Model(setup6,noAugafter100)', 'finetuning/flownetLC9_LR3_130_Model(setup6,noAugafter100)'}

	--{'flownetLC6_LR3_180','newWithoutReg/flownetLC8_LR3_210','newWithoutReg/flownetLC8_LR3_240','newWithoutReg/flownetLC8_LR3_260','newWithoutReg/flownetLC8_LR3_268','newWithoutReg/flownetLC8_LR3_275','newWithoutReg/flownetLC8_LR3_270','newWithoutReg/flownetLC8_LR3_280'}	
	
	--{'flownetLC6_LR3_55','newWithoutReg/flownetLC7_LR3_30','newWithoutReg/flownetLC7_LR3_55'}
	--{'flownetLC6_LR3_5','flownetLC6_LR3_55','flownetLC6_LR3_95','flownetLC6_LR3_115','flownetLC6_LR3_145','flownetLC6_LR3_155','flownetLC6_LR3_165','flownetLC6_LR3_180'} 
	--{'flownetLC1_LR3_5000','flownetLC2_LR3_1000','flownetLC3_LR3_500','flownetLC4_LR3_400','flownetLC5_LR3_200','flownet_LR3_172'} -
	for i=1,#models do --for i=1,6 do
	  local model1 = torch.load('logFiles/' .. models[i] .. '.t7') -- remove _Model if adding in the model names above
	  local avgValErr = validation(model1, valData, criterion, downSampleFlowWeights,opTDataMode)
      	  valLogger:add{avgValErr}
	  valLogger:plot()
        end
end
-- close databases
-- trainDataLoader:close() uncomment if needed

-- enforce clean exit
os.exit(0)
