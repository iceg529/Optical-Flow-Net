
require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'lfs'
require 'nn'
require 'cunn'
require 'cutorch'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
    package.path = dir_path .."?.lua;".. package.path
end

require 'Optimizer'
require 'LRPolicy'
require 'logmessage'

-- load utils
local utils = require 'utils'
----------------------------------------------------------------------
opt.shuffle = opt.shuffle == 'yes' or false
opt.visualizeModel = opt.visualizeModel == 'yes' or false

-- Set the seed of the random number generator to the given number.
if opt.seed ~= '' then
    torch.manualSeed(tonumber(opt.seed))
end

torch.setnumthreads(opt.threads)
local nGpus = cutorch.getDeviceCount()

----------------------------------------------------------------------
-- Open Data sources:
-- training database 
-- optionally: validation database 

local data = require 'data'

-- DataLoader objects take care of loading data from
-- databases. Data loading and tensor manipulations
-- (e.g. cropping, mean subtraction, mirroring) are
-- performed from separate threads
local trainDataLoader, trainSize, inputTensorShape
local valDataLoader, valSize

local num_threads_data_loader = 4

if opt.train ~= '' then
    -- create data loader for training dataset
    trainDataLoader = DataLoader:new(
            num_threads_data_loader, -- num threads
            package.path,
            opt.dbbackend, opt.train, opt.train_labels,
            meanTensor,
            true, -- train
            opt.shuffle,
            classes ~= nil -- whether this is a classification task
        )
    -- retrieve info from train DB (number of records and shape of input tensors)
    trainSize, inputTensorShape = trainDataLoader:getInfo()
    logmessage.display(0,'found ' .. trainSize .. ' images in train db' .. opt.train)
    if opt.validation ~= '' then
        local shape
        valDataLoader = DataLoader:new(
                num_threads_data_loader, -- num threads
                package.path,
                opt.dbbackend, opt.validation, opt.validation_labels,
                meanTensor,
                false, -- train
                false, -- shuffle
                classes ~= nil -- whether this is a classification task
            )
        valSize, shape = valDataLoader:getInfo()
        logmessage.display(0,'found ' .. valSize .. ' images in train db' .. opt.validation)
    end
else
    assert(opt.visualizeModel, 'Train DB should be specified')
end





-- if batch size was not specified on command line then check
-- whether the network defined a preferred batch size (there
-- can be separate batch sizes for the training and validation
-- sets)
local trainBatchSize
local valBatchSize
if opt.batchSize==0 then
    local defaultBatchSize = 16
    trainBatchSize = network.trainBatchSize or defaultBatchSize
    valBatchSize = network.validationBatchSize or defaultBatchSize
else
    trainBatchSize = opt.batchSize
    valBatchSize = opt.batchSize
end
logmessage.display(0,'Train batch size is '.. trainBatchSize .. ' and validation batch size is ' .. valBatchSize)




-- Train function
local function Train(epoch, dataLoader)

    model:training()

    local NumBatches = 0
    local curr_images_cnt = 0
    local loss_sum = 0
    local loss_batches_cnt = 0
    local learningrate = 0
    local inputs, targets

    local dataLoaderIdx = 1

    local data = {}

    local t = 1
    while t <= trainSize do

        while dataLoader:acceptsjob() do
            local dataBatchSize = math.min(trainSize-dataLoaderIdx+1,trainBatchSize)
            if dataBatchSize > 0 then
                dataLoader:scheduleNextBatch(dataBatchSize, dataLoaderIdx, data, true)
                dataLoaderIdx = dataLoaderIdx + dataBatchSize
            else break end
        end

        NumBatches = NumBatches + 1

        -- wait for next data loader job to complete
        dataLoader:waitNext()

        -- get data from last load job
        local thisBatchSize = data.batchSize
        inputs = data.inputs
        targets = data.outputs

        if inputs then
            --[=[
            -- print some statistics, show input in iTorch

            if t%1024==1 then
                print(string.format("input mean=%f std=%f",inputs:mean(),inputs:std()))
                for idx=1,thisBatchSize do
                    print(classes[targets[idx]])
                end
                if itorch then
                    itorch.image(inputs)
                end
            end
            --]=]

            if opt.type =='cuda' then
                inputs = inputs:cuda()
                targets = targets:cuda()
            else
                inputs = inputs:float()
                targets = targets:float()
            end

            _,learningrate,_,trainerr = optimizer:optimize(inputs, targets)

            -- adding the loss values of each mini batch and also maintaining the counter for number of batches, so that average loss value can be found at the time of logging details
            loss_sum = loss_sum + trainerr[1]
            loss_batches_cnt = loss_batches_cnt + 1

            if math.fmod(NumBatches,50)==0 then
                collectgarbage()
            end

            local current_epoch = (epoch-1)+utils.round((math.min(t+trainBatchSize-1,trainSize))/trainSize, epoch_round)

            -- log details on first iteration, or when required number of images are processed
            curr_images_cnt = curr_images_cnt + thisBatchSize
            if (epoch==1 and t==1) or curr_images_cnt >= logging_check then
                logmessage.display(0, 'Training (epoch ' .. current_epoch .. '): loss = ' .. (loss_sum/loss_batches_cnt) .. ', lr = ' .. learningrate)
                curr_images_cnt = 0 -- For accurate values we may assign curr_images_cnt % logging_check to curr_images_cnt, instead of 0
                loss_sum = 0
                loss_batches_cnt = 0
            end

            if opt.validation ~= '' and current_epoch >= next_validation then
                Validation(model, loss, current_epoch, valDataLoader, valSize, valBatchSize, valConfusion, labelFunction)

                next_validation = (utils.round(current_epoch/opt.interval) + 1) * opt.interval -- To find next nearest epoch value that exactly divisible by opt.interval
                last_validation_epoch = current_epoch
                model:training() -- to reset model to training
            end

            if current_epoch >= next_snapshot_save then
                saveModel(model, opt.save, snapshot_prefix, current_epoch)
                next_snapshot_save = (utils.round(current_epoch/opt.snapshotInterval) + 1) * opt.snapshotInterval -- To find next nearest epoch value that exactly divisible by opt.snapshotInterval
                last_snapshot_save_epoch = current_epoch
            end

            t = t + thisBatchSize
        else
            -- failed to read from database (possibly due to disabled thread)
            dataLoaderIdx = dataLoaderIdx - data.batchSize
        end

    end

    --xlua.progress(trainSize, trainSize)

end

------------------------------

local epoch = 1

logmessage.display(0,'started training the model')

-- run an initial validation before the first train epoch
if opt.validation ~= '' then
    Validation(model, loss, 0, valDataLoader, valSize, valBatchSize, valConfusion, labelFunction)
end

while epoch<=opt.epoch do
    local ErrTrain = 0
    if trainConfusion ~= nil then
        trainConfusion:zero()
    end
    Train(epoch, trainDataLoader)
    if trainConfusion ~= nil then
        trainConfusion:updateValids()
        --print(trainConfusion)
        ErrTrain = (1-trainConfusion.totalValid)
    end
    epoch = epoch+1
end

-- if required, perform validation at the end
if opt.validation ~= '' and opt.epoch > last_validation_epoch then
    Validation(model, loss, opt.epoch, valDataLoader, valSize, valBatchSize, valConfusion, labelFunction)
end

-- if required, save snapshot at the end
if opt.epoch > last_snapshot_save_epoch then
    saveModel(model, opt.save, snapshot_prefix, opt.epoch)
end

-- close databases
trainDataLoader:close()
if opt.validation ~= '' then
    valDataLoader:close()
end

-- enforce clean exit
os.exit(0)


