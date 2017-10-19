require 'torch'
require 'nn'
require 'nngraph'

function getResModel1()
  
  local outputs = {}
  --table.insert(inputs, nn.Identity()())
  local imgIn = nn.Identity()()
--  local flowIn = nn.Identity()()
  local inputs = {imgIn}

  -- stage 1 : filter bank -> squashing -> filter bank -> squashing
  local h1 = imgIn - nn.SpatialConvolution(6, 64, 7, 7, 2, 2, 3, 3)
                   - nn.ReLU()
                   - nn.SpatialConvolution(64, 64, 5, 5, 2, 2, 2, 2)
                   - nn.ReLU()

  -- stage 2 : filter bank -> squashing -> filter bank -> squashing
  local h2 = h1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                - nn.ReLU()
                - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		- nn.ReLU()

  local h2_1 = nn.CAddTable()({h2, h1})
               

  -- stage 3 : filter bank -> squashing -> filter bank -> squashing
  local h3 = h2_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h3_1 = nn.CAddTable()({h3, h2_1})
                      
  -- stage 4 : filter bank -> squashing -> filter bank -> squashing
  local h4_0 = h3_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  
  local h4 = h4_0 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h4_1 = nn.CAddTable()({h4, h3_1})
       
  -- stage 5 : filter bank -> squashing -> filter bank -> squashing
  local h5 = h4_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h5_1 = nn.CAddTable()({h5, h4_1})

  -- stage 6 : filter bank -> squashing -> filter bank -> squashing
  local h6 = h5_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()
  
  local h6_0 = nn.CAddTable()({h6, h5_1})

  local h6_1 = h6_0 - nn.SpatialMaxPooling(2, 2, 2, 2)
  
  local h6_2 = h6_1 - nn.Copy(cudaTensor, cudaTensor)
		    - nn.MulConstant(0)
  local h6_concat = nn.JoinTable(2)({h6_1, h6_2})

  -- additions Res 2
  --local flowDisp1    = h6_1 - nn.SpatialConvolution(64, 2, 3, 3, 1, 1, 1, 1) --h6_0
    
  --------------------------
  
  -- stage 7 : filter bank -> squashing -> filter bank -> squashing
  local h7 = h6_1 - nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h7_1 = nn.CAddTable()({h7, h6_concat})

  -- additions Res 3
  local catFeat = nn.JoinTable(2)({h1, h4_0, h6_0})
  local flowDisp1    = catFeat - nn.SpatialConvolution(192, 2, 3, 3, 1, 1, 1, 1)
			       - nn.SpatialMaxPooling(2, 2, 2, 2) 

  --local mergeOut = nn.JoinTable(2)({h7_1, flowDisp1})

  --table.insert(outputs, h7_1)
  --table.insert(outputs, mergeOut)
  --outputs = {h6_0, h1, h4_0} 
  outputs = {h7_1, flowDisp1} 
  --table.insert(outputs, h6_1)
  return nn.gModule(inputs, outputs)
end


