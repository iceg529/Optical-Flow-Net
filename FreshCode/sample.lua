
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

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
  package.path = dir_path .."?.lua;".. package.path
end

local opTtrain = 'trainData2.h5'
local opTval = 'testData.h5'
local opTshuffle = false 
local opTthreads = 2
local opTepoch = 1--50
local opTsnapshotInterval = 10
local opTsave = "logFiles/"
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
require 'model'

-- DataLoader objects take care of loading data from
-- databases. Data loading and tensor manipulations
-- (e.g. cropping, mean subtraction, mirroring) are
-- performed from separate threads
local trainDataLoader, trainSize, inputTensorShape
local valDataLoader, valSize

local num_threads_data_loader = 2


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

-- create data loader for validation dataset
valDataLoader = DataLoader:new(
      1, -- num threads
      package.path,
      opTval,
      false, -- train
      opTshuffle
)
-- retrieve info from train DB (number of records and shape of input tensors)
valSize, valInputTensorShape = valDataLoader:getInfo()
logmessage.display(0,'found ' .. valSize .. ' images in train db' .. opTval)

local valData = {}
if valDataLoader:acceptsjob() then
  valDataLoader:scheduleNextBatch(640, 1, valData, true)
end
valDataLoader:waitNext()

-- validation function
local function validation(model,valData,criterion)
  --model:evaluate()
  local valErr = 0
  --local valData = {} 
    
  print(valData.im1:size())
  for i = 1,valData.im1:size(1) do
    -- estimate f
    local input = torch.cat(valData.im1[i], valData.im2[i], 1)
    input = input:cuda()
    local output = model:forward(input)
    local flInput = valData.flow[i]
    flInput = flInput:cuda()
    --local down5 = nn.SpatialAdaptiveAveragePooling(128, 96):cuda():forward(flInput)
    local err = criterion:forward(output, flInput)
    valErr = valErr + err
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
testLogger = optim.Logger(paths.concat(opTsave, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
--local model = getModel()
local model = require('weight-init')(getModel(), 'kaiming')
--local model = torch.load('logFiles/flownet_90_Model.t7')
model = model:cuda()
local criterion = nn.AvgEndPointError() --SmoothL1Criterion
criterion = criterion:cuda()
if model then
   parameters,gradParameters = model:getParameters()
end


----------------------
local function saveModel(model, directory, prefix, epoch)
    local filename
    local modelObjectToSave
    if model.clearState then
        -- save the full model
        filename = paths.concat(directory, prefix .. '_' .. epoch .. '_Model.t7')
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

-- if batch size was not specified on command line then check
-- whether the network defined a preferred batch size (there
-- can be separate batch sizes for the training and validation
-- sets)
local trainBatchSize = 8
local logging_check = 2779 --11116 -- 8 splits of training data(22232/8)
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

logmessage.display(0,'started training the model')

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
  
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. trainBatchSize .. ']')

  local t = 1
  while t <= trainSize do
    --model:training()
    -- disp progress
    xlua.progress(t, trainSize)
    local time2 = sys.clock()
    profiler:start('pre-fetch')
    -- prefetch data thread
    --------------------------------------------------------------------------------
    while trainDataLoader:acceptsjob() do      
      local dataBatchSize = math.min(trainSize-dataLoaderIdx+1,trainBatchSize)
      if dataBatchSize > 0 then
        trainDataLoader:scheduleNextBatch(dataBatchSize, dataLoaderIdx, data, true)
        dataLoaderIdx = dataLoaderIdx + dataBatchSize
      else break end
    end
    NumBatches = NumBatches + 1

    -- wait for next data loader job to complete
    trainDataLoader:waitNext()
    --------------------------------------------------------------------------------
    profiler:lap('pre-fetch')
    -- get data from last load job
    local thisBatchSize = data.batchSize
    im1 = data.im1
    im2 = data.im2
    flow = data.flow
    
    im1 = im1:cuda()
    im2 = im2:cuda()
    flow = flow:cuda()
        
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

      -- evaluate function for complete mini batch
      for i = 1,im1:size(1) do
        -- estimate f
        local input = torch.cat(im1[i], im2[i], 1)
        local output = model:forward(input)
        local down5 = nn.SpatialAdaptiveAveragePooling(128, 96):cuda():forward(flow[i]) -- upsampled the prediction flow
        --print('aft model fwd')
        down5 = down5:cuda()
	local grdTruth = flow[i]:cuda()
        local err = criterion:forward(output, down5) --grdTruth
        f = f + err
        -- estimate df/dW
        local df_do = criterion:backward(output, down5) --grdTruth
        model:backward(input, df_do)
      end

      -- normalize gradients and f(X)
      gradParameters:div(im1:size(1))
      f = f/(im1:size(1))
      print(f)

      -- return f and df/dX
      return f,gradParameters
    end

    -- optimize on current mini-batch

    config = config or {learningRate = 0.0001,
                   weightDecay = 0.0004,
                   momentum = 0.9,
                   learningRateDecay = 3e-5}
                 
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
      local avgValErr = 0 -- validation(model, valData, criterion)
      
      trainLogger:add{avgLoss, avgValErr}
      trainLogger:plot()
      
      --logmessage.display(0, 'Training (epoch ' .. current_epoch)
      curr_images_cnt = 0 -- For accurate values we may assign curr_images_cnt % logging_check to curr_images_cnt, instead of 0
      loss_sum = 0
      loss_batches_cnt = 0
      loggerCnt = loggerCnt + 1
    end

    if current_epoch >= next_snapshot_save or current_epoch == opTepoch then
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
  print("==> time to learn for 1 epoch = " .. (time) .. 'ms')
   
  epoch = epoch+1
  if epoch == opTepoch then
    trainLogger:display(true)
  end
end     

-- if required, save snapshot at the end
if opTepoch > last_snapshot_save_epoch then
  saveModel(model, opTsave, snapshot_prefix, opTepoch)
end

-- close databases
-- trainDataLoader:close() uncomment if needed

-- enforce clean exit
os.exit(0)
