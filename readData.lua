require 'image'
require 'hdf5'
require 'torch'
local threads = require 'threads'

--local myFile = hdf5.open('trainData.h5', 'r')
--local data = myFile:read('/img1'):partial(unpack({2211,1,380,510}))
--myFile:close()

--print(data)



--local mainfile = hdf5.open('trainData.h5', 'r')
local nthreads = 8
local data = nil

local worker = function(mainfile)
	torch.setnumthreads(1)
	print(__threadid)
	return h5file:read('/img1' .. __threadid):all()
end

local pool = threads.Threads(nthreads, function(threadid) require'torch' require'hdf5'end)
pool:specific(true)
local time = sys.clock()
for i=1,nthreads do
	pool:addjob(i,
	            function() 
			print(__threadid)
		        return hdf5.open('trainData1.h5', 'r'):read('trainData/data' .. __threadid):all() 
            
		    end, 
		    function(_data) data = _data end)
end

for i=1,nthreads do
	pool:dojob()
	print(data:size())
  print(sys.clock() - time)
end
--mainfile:close()
