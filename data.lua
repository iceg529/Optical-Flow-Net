-- loading data to hdf5 and preprocessing
require 'image'
require 'hdf5'

function readDataToHdf5(fileName,targetH5File,tensor,targetKeyName)
	-- Opens a file in read
	file = io.open(fileName, "r")
	
	local myFile = hdf5.open(targetH5File, 'w')
	local options = hdf5.DataSetOptions()
	options:setChunked(32, 32)
	options:setDeflate()
	local i=1
	
	for img in file:lines() do 
	    print(targetH5File,i)
	    collectgarbage()
	    tensor[i] = image.load(img)
	    i=i+1
	end
	print(collectgarbage("count"))
	myFile:write(targetKeyName, tensor)
	print(collectgarbage("count"))
	-- closes the open file
	io.close(file)
	myFile:close()
end

--train image1
local t=torch.Tensor(22232,3,384,512)
readDataToHdf5('Train_Img1.list','trainImg.h5',t,'/trainData1')

--train image2
--local t=torch.Tensor(22232,3,384,512)
--readDataToHdf5('Train_Img2.list','trainImg.h5',t,'/trainData2')

--test image2
--local t=torch.Tensor(640,3,384,512)
--readDataToHdf5('Test_Img1.list','testImg.h5',t,'/testData1')

--test image2
--local t=torch.Tensor(640,3,384,512)
--readDataToHdf5('Test_Img2.list','testImg.h5',t,'/testData2')

