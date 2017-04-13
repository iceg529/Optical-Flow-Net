
require 'lmdb'

function fetchData()
	local db= lmdb.env{
	    Path = '../../FlowNet/dispflownet-release/data/FlyingChairs_release_lmdb',
	    Name = 'FlyingChairs_release_lmdb'
	}

	db:open()
	print(db:stat()) -- Current status
--	local txn = db:txn()
--	local cursor = txn:cursor()	
	local reader = db:txn(true) --Read-only transaction
	local y = torch.Tensor(10,3,384,512)
--	cnt=0
	cursor = reader:cursor()
--	repeat
--	   k,v = cursor:get(k,lmdb.MDB_NEXT)
	   -- do something with your key/value pair
--	   cnt = cnt + 1
--	   print(k)
--	until (not k)

--	print(cnt)
	-------Read-------
--	for i=1,10 do
--	    y[i] = reader:get(i)
--	end
	data1 = reader:get("00000002_FlyingChairs_release/data/00003_flow.flo")
	print(data1[1990669])

	
	

	reader:abort()

	db:close()
end

fetchData()
