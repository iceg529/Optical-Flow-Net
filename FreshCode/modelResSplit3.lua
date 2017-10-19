require 'torch'
require 'nn'
require 'nngraph'

function getResModel3()
  
  local outputs = {}
  --table.insert(inputs, nn.Identity()())
  local imgIn = nn.Identity()()
  
  -- additions Res 2
  --local flowDisp1    = nn.Identity()()

  local h1  = nn.Identity()()
  local h4_0  = nn.Identity()()

  local inputs = {imgIn,h1,h4_0} --{imgIn,flowDisp1}
  --local inputs = {imgIn}
  -- additions Res 3
  local catFeat = nn.JoinTable(2)(inputs)
  local flowDisp1    = catFeat - nn.SpatialConvolution(192, 2, 3, 3, 1, 1, 1, 1)
			       - nn.SpatialMaxPooling(2, 2, 2, 2)
  
  local h6_1 = imgIn - nn.SpatialMaxPooling(2, 2, 2, 2)
  
  --local h6_2 = torch.cudaTensor(h6_1:size()):fill(0)
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
    

  outputs = {h7_1,flowDisp1}

  return nn.gModule(inputs, outputs)
end


