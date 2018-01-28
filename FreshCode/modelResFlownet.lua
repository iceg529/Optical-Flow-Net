require 'torch'
require 'nn'
require 'nngraph'

function getResModel()
  
  local outputs = {}
  --table.insert(inputs, nn.Identity()())
  local imgIn = nn.Identity()()
--  local flowIn = nn.Identity()()
  local inputs = {imgIn}

  -- stage 1 : filter bank -> squashing -> filter bank -> squashing
  local h1 = imgIn - nn.SpatialConvolution(6, 64, 7, 7, 2, 2, 3, 3)
                   - nn.ReLU()
                   - nn.SpatialConvolution(64, 128, 5, 5, 2, 2, 2, 2)
                   - nn.ReLU()

  local h1_1 = h1 - nn.SpatialMaxPooling(2, 2, 2, 2)

  local h1_2 = h1_1 - nn.Copy(cudaTensor, cudaTensor)
		    - nn.MulConstant(0)
  local h1_concat = nn.JoinTable(2)({h1_1, h1_2})

  -- stage 2 : filter bank -> squashing -> filter bank -> squashing
  local h2 = h1_1 - nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h2_0 = nn.CAddTable()({h2, h1_concat})

  -- stage 3 : filter bank -> squashing -> filter bank -> squashing
  local h3 = h2_0 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h3_0 = nn.CAddTable()({h3, h2_0})              

  -- stage 4 : filter bank -> squashing -> filter bank -> squashing
  local h4 = h3_0 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h4_0 = nn.CAddTable()({h4, h3_0})  
 
  local h4_1 = h4_0 - nn.SpatialMaxPooling(2, 2, 2, 2)

  local h4_2 = h4_1 - nn.Copy(cudaTensor, cudaTensor)
		    - nn.MulConstant(0)
  local h4_concat = nn.JoinTable(2)({h4_1, h4_2})             
       
  -- stage 5 : filter bank -> squashing -> filter bank -> squashing
  local h5 = h4_1 - nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h5_0 = nn.CAddTable()({h5, h4_concat})
  
  -- stage 6 : filter bank -> squashing -> filter bank -> squashing
  local h6 = h5_0 - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h6_0 = nn.CAddTable()({h6, h5_0})              

  -- stage 7 : filter bank -> squashing -> filter bank -> squashing
  local h7 = h6_0 - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h7_0 = nn.CAddTable()({h7, h6_0})  
 
  local h7_1 = h7_0 - nn.SpatialMaxPooling(2, 2, 2, 2)

  -- commented bcos we dont need double ,as next layer also has same 512. special case 
  --local h7_2 = h7_1 - nn.Copy(cudaTensor, cudaTensor)
	--	      - nn.MulConstant(0)
  --local h7_concat = nn.JoinTable(2)({h7_1, h7_2})

  -- stage 8 : filter bank -> squashing -> filter bank -> squashing
  local h8 = h7_1 - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h8_0 = nn.CAddTable()({h8, h7_1})
               
  -- stage 9 : filter bank -> squashing -> filter bank -> squashing
  local h9 = h8_0 - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h9_0 = nn.CAddTable()({h9, h8_0})  

  -- stage 10 : filter bank -> squashing -> filter bank -> squashing
  local h10 = h9_0 - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                   - nn.ReLU()
                   - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
		   - nn.ReLU()

  local h10_0 = nn.CAddTable()({h10, h9_0})  
 
  local h10_1 = h10_0 - nn.SpatialMaxPooling(2, 2, 2, 2)

  local h10_2 = h10_1 - nn.Copy(cudaTensor, cudaTensor)
		      - nn.MulConstant(0)
  local h10_concat = nn.JoinTable(2)({h10_1, h10_2}) 


  -- stage 11 : filter bank -> squashing -> filter bank -> squashing
  local h11 = h10_1 - nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h11_0 = nn.CAddTable()({h11, h10_concat})
               
  -- stage 12 : filter bank -> squashing -> filter bank -> squashing
  local h12 = h11_0 - nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h12_0 = nn.CAddTable()({h12, h11_0})

  -- stage 13 : filter bank -> squashing -> filter bank -> squashing
  local h13 = h12_0 - nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h13_0 = nn.CAddTable()({h13, h12_0})  
 
  --[[local h7_1 = h7_0 - nn.SpatialMaxPooling(2, 2, 2, 2)

  local h7_2 = h7_1 - nn.Copy(cudaTensor, cudaTensor)
		    - nn.MulConstant(0)
  local h7_concat = nn.JoinTable(2)({h7_1, h7_2}) --]]

  
  -- Deconvolution and Concatnate stage 1 
  local Con1    = h13_0 - nn.SpatialConvolution(1024, 2, 3, 3, 1, 1, 1, 1)
  local Con1_up = Con1 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)  

  local ConCat1 = nn.JoinTable(2)({h10_0, Con1_up})

  -- Deconvolution and Concatnate stage 2 
  local Con2    = ConCat1 - nn.SpatialConvolution(514, 2, 3, 3, 1, 1, 1, 1)
  local Con2_up = Con2 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)  

  local ConCat2 = nn.JoinTable(2)({h7_0, Con2_up})

  -- Deconvolution and Concatnate stage 3 
  local Con3    = ConCat2 - nn.SpatialConvolution(514, 2, 3, 3, 1, 1, 1, 1)
  local Con3_up = Con3 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)  

  local ConCat3 = nn.JoinTable(2)({h4_0, Con3_up})

  -- Deconvolution and Concatnate stage 4 
  local Con4    = ConCat3 - nn.SpatialConvolution(258, 2, 3, 3, 1, 1, 1, 1)
  local Con4_up = Con4 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)  

  local ConCat4 = nn.JoinTable(2)({h1, Con4_up})

  -- Final Convolution stage
  local Con5    = ConCat4 - nn.SpatialConvolution(130, 2, 3, 3, 1, 1, 1, 1)


  table.insert(outputs, Con5)

  return nn.gModule(inputs, outputs)
end


