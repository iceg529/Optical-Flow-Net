-- data.lua: script to fetch and preprocess batch data from hdf5 datasets
require 'image'
require 'torch' -- torch
require 'nn' -- provides a normalization operator
require 'utils' -- various utility functions
require 'hdf5' -- import HDF5 now as it is unsafe to do it from a worker thread
require 'cunn'
require 'cutorch'

local threads = require 'threads' -- for multi-threaded data loader

--if color transforms are needed : check_require('image') 

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'

local tdsIsInstalled, tds = pcall(function() return check_require('tds') end)

-- enable shared serialization to speed up Tensor passing between threads
threads.Threads.serialization('threads.sharedserialize')

----------------------------------------------------------------------

-- shallow-copy a table
function copy (t) 
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end

-- Meta class
DBSource = {mean = nil, ImChannels = 0, ImHeight = 0, ImWidth = 0, FlowChannels = 2, total=0,
            augOpt={}, subtractMean=false,
            train=false}

-- Derived class method new
-- Creates a new instance of a database
-- Parameters:
-- @param db_path (string): path to database
-- @param labels_db_path (string): path to labels database, or nil
-- @param isTrain (boolean): whether this is a training database (e.g. mirroring not applied to validation database)
-- @param shuffle (boolean): whether samples should be shuffled
function DBSource:new (db_path, isTrain, shuffle)
    local self = copy(DBSource)
    local paths = require('paths')
    check_require('hdf5')
    
    logmessage.display(0,'opening HDF5 database: ' .. db_path)
    self.dbs = {}
    self.total = 0
            
    -- get number of records
    local myFile = hdf5.open(db_path,'r')
    local dim, n_records
    if isTrain then
    	dim = myFile:read('/data1'):dataspaceSize()
    	n_records = 22232 -- 22232 904*2
    else
        dim = myFile:read('/data1'):dataspaceSize()
    	n_records = dim[1] -- 640 137*1
    end
    myFile:close()
        
    self.ImChannels = 3
    self.ImHeight = dim[3]
    self.ImWidth = dim[4]
    self.FlowChannels = 2
    
    -- store DB info
    self.dbs[#self.dbs + 1] = {
        path = db_path,
        records = n_records
    }
    self.total = self.total + n_records
        
    -- which DB is currently being used (initially set to nil to defer
    -- read to first call to self:nextBatch)
    self.db_id = nil
    -- cursor points to current record in current DB
    self.cursor = nil
    -- set pointers to HDF5-specific functions
    self.getSample = self.hdf5_getSample
    self.reset = self.hdf5_reset
    self.batchSize = nil
    logmessage.display(0,'Image channels are ' .. self.ImChannels .. ', Image width is ' .. self.ImWidth .. ' and Image height is ' .. self.ImHeight)

    self.train = isTrain
    self.shuffle = shuffle    

    return self
end

-- Derived class method inputTensorShape
-- This returns the shape of the input samples in the database
-- There is an assumption that all input samples have the same shape
function DBSource:inputTensorShape()
    local shape = torch.Tensor(3)
    shape[1] = self.ImChannels
    shape[2] = self.ImHeight
    shape[3] = self.ImWidth
    return shape
end

-- Derived class method getSample (HDF5 flavour)
function DBSource:hdf5_getSample(shuffle, indx)    
    if not self.db_id or self.cursor>self.dbs[self.db_id].records then
        self.db_id = self.db_id or 0
        assert(self.db_id < #self.dbs, "Trying to read more records than available")
        self.db_id = self.db_id + 1        
        self.cursor = 1
    end     
    --print('id ' .. self.db_id)
    --print('cursor ' .. self.cursor)
    --print('records ' .. self.dbs[self.db_id].records)
    --print('Num of DBs ' .. #self.dbs)
    local idx = indx
--    if shuffle then
--        idx = math.max(1,torch.ceil(torch.rand(1)[1] * (self.dbs[self.db_id].records)*(1/8)))
--    else
--        idx = self.cursor
--    end
    
--    if self.train then
      --idx = idx*8 -- multiplied by 8 because data stored with keyvalues in multiples of 8 , change it if dataset key values are fixed
--      idx = idx + 7 -- similar reason
--    end
    local myFile = hdf5.open(self.dbs[self.db_id].path, 'r')
    local concatData = myFile:read('/data' .. idx):all()
    local im1 = concatData:sub(1,self.batchSize,1,3)
    local im2 = concatData:sub(1,self.batchSize,4,6)
    local flow = concatData:sub(1,self.batchSize,7,8)
    --print('/data' .. idx)
    myFile:close()
    
    self.cursor = self.cursor + 1
    return im1, im2, flow
end

-- Derived class method nextBatch
-- Parameters:
-- @param batchSize (int): Number of samples to load
-- @param indx (int): Current index within database
function DBSource:nextBatch (batchSize, indx)

    local im1Batch, im2Batch, flowBatch

    -- this function creates a tensor that has similar
    -- shape to that of the provided sample plus one
    -- dimension (batch dimension)
    local function createBatchTensor(sample, batchSize)
        local t
        if type(sample) == 'number' then
            t = torch.Tensor(batchSize)
        else
            shape = sample:size():totable()
            -- add 1 dimension (batchSize)
            table.insert(shape, 1, batchSize)
            t = torch.Tensor(torch.LongStorage(shape))
        end
        return t
    end
    self.batchSize = batchSize    
    im1Batch, im2Batch, flowBatch = self:getSample(self.shuffle, indx)
    
    return im1Batch, im2Batch, flowBatch
end

-- Derived class method to reset cursor
function DBSource:hdf5_reset ()
    self.db_id = nil
    self.cursor = nil
end

-- Derived class method to get total number of Records
function DBSource:totalRecords ()
    return self.total;
end


-- Meta class
DataLoader = {}

-- Derived class method new
-- Creates a new instance of a database
-- Parameters:
-- @param numThreads (int): number of reader threads to create
-- @param package_path (string): caller package path
-- @param db_path (string):  path to database
-- @param isTrain (boolean): whether this is a training database (e.g. mirroring not applied to validation database)
-- @param shuffle (boolean): whether samples should be shuffled
function DataLoader:new (numThreads, package_path, db_path, isTrain, shuffle)
    local self = copy(DataLoader)
    
    -- create pool of threads
    self.numThreads = numThreads
    self.threadPool = threads.Threads(
        self.numThreads,
        function(threadid)
            -- inherit package path from main thread
            package.path = package_path
            require('data')
            -- executes in reader thread, variables are local to this thread
            db = DBSource:new(db_path, isTrain, shuffle)
        end
    )
    -- use non-specific mode
    self.threadPool:specific(false)
    return self
end

function DataLoader:getInfo()
    local datasetSize
    local inputTensorShape

    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    -- we need to iterate here as some threads may not have a valid DB
    -- handle (as happens when opening too many concurrent instances)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                   i, -- thread to add job to
                   function()
                       -- executes in reader thread, return values passed to
                       -- main thread through following function
		       print('get info read thread')
                       if db then
                           return db:totalRecords(), db:inputTensorShape()
                       else
                           return nil, nil
                       end
                   end,
                   function(totalRecords, shape)
                       -- executes in main thread
                       datasetSize = totalRecords
                       inputTensorShape = shape
                   end
                   )
        self.threadPool:synchronize()
        if datasetSize then
            break
        end
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
    return datasetSize, inputTensorShape
end

-- Schedule next data loader batch
-- Parameters:
-- @param batchSize (int): Number of samples to load
-- @param dataIdx (int): Current index in database
-- @param dataTable (table): Table to store data into
function DataLoader:scheduleNextBatch(batchSize, dataIdx, dataTable)
    -- send reader thread a request to load a batch from the training DB
    self.threadPool:addjob(
                function()
                    if dataIdx == 1 then
                        db:reset()
                    end
		    -- executes in reader thread
		    --print('data reader in prog...')
                    if db then
                        in1, in2, out =  db:nextBatch(batchSize, dataIdx)
                        return batchSize, in1, in2, out, dataIdx
                    else
                        return batchSize, nil, nil, nil, nil
                    end
                end,
                function(batchSize, in1, in2, out, indx)
                    -- executes in main thread
                    
		    --[[profiler:start('augmentation process')		    
	 	    ----- augmentation ----------	    
		    if torch.bernoulli(0.5) == 1 then  -- 0.5
	 	      print('in affine')
		      local trMax = round(0.2 * in1:size(4))
		      local trY, trX, theta
		      local scaledW, scaledH, scaleRand, actualW, actualH   	      

		      for i = 1,in1:size(1) do
			trY = torch.random(-trMax,trMax)
			trX = torch.random(-trMax,trMax)
			theta = math.rad(torch.random(-17,17))
			scaleRand = ((torch.uniform() * 0.8) + 1.2) -- range conversion "NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin"
			scaledW = round(scaleRand * in1:size(4))
			scaledH = round(scaleRand * in1:size(3))	
			actualW = in1:size(4)
			actualH = in1:size(3)

			in1[i] = image.crop(image.scale(image.rotate(image.translate(in1[i], trX, trY), theta, 'bilinear'), scaledW, scaledH), 'c', actualW, actualH)
			in2[i] = image.crop(image.scale(image.rotate(image.translate(in2[i], trX, trY), theta, 'bilinear'), scaledW, scaledH), 'c', actualW, actualH)
			out[i] = image.crop(image.scale(image.rotate(image.translate(out[i], trX, trY), theta, 'bilinear'), scaledW, scaledH), 'c', actualW, actualH)
						
			local colorAugParams = {addNoiseSig = torch.uniform(0, 0.04), colFac1 = torch.uniform(0.5, 2), colFac2 = torch.uniform(0.5, 2), colFac3 = torch.uniform(0.5, 2), gamma = torch.uniform(0.7, 1.5), brightnessSigma = 0.2, contrast = torch.uniform(-0.8, 0.4)}
						
			in1[i] = colorAugmentation(in1[i],colorAugParams)
			in2[i] = colorAugmentation(in2[i],colorAugParams)
		      end
	  	    end
		    profiler:lap('augmentation process')--]]

		    dataTable.batchSize = batchSize
                    dataTable.im1 = in1
                    dataTable.im2 = in2
                    dataTable.flow = out
		    dataTable.indx = indx
		    --print('after data has set up')
                end
            )
end

-- returns whether data loader is able to accept more jobs
function DataLoader:acceptsjob()
    return self.threadPool:acceptsjob()
end

-- wait until next data loader job completes
function DataLoader:waitNext()
    -- wait for next data loader job to complete
    self.threadPool:dojob()
    -- check for errors in loader threads
    if self.threadPool:haserror() then -- check for errors
        self.threadPool:synchronize() -- finish everything and throw error
    end
end

-- free data loader resources
function DataLoader:close()
    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                    i,
                    function()
                        if db then
                            db:close()
                        end
                    end
                )
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
end

