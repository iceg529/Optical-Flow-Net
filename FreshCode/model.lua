require 'torch'
require 'nn'
require 'nngraph'

function getModel()
  
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
       
  -- stage 2 : filter bank -> squashing -> filter bank -> squashing
  local h2 = h1 - nn.SpatialConvolution(128, 256, 5, 5, 2, 2, 2, 2)
                - nn.ReLU()
                - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                - nn.ReLU()

  -- stage 3 : filter bank -> squashing -> filter bank -> squashing
  local h3 = h2 - nn.SpatialConvolution(256, 512, 3, 3, 2, 2, 1, 1)
                - nn.ReLU()
                - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                - nn.ReLU()
       
  -- stage 4 : filter bank -> squashing -> filter bank -> squashing
  local h4 = h3 - nn.SpatialConvolution(512, 512, 3, 3, 2, 2, 1, 1)
                - nn.ReLU()
                - nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)
                - nn.ReLU()
       
  -- stage 5 : filter bank -> squashing -> filter bank -> squashing
  local h5 = h4 - nn.SpatialConvolution(512, 1024, 3, 3, 2, 2, 1, 1)
                - nn.ReLU()
                - nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)
                - nn.ReLU()
        
  -- Deconvolution and Concatnate stage 1 
  local Con1    = h5 - nn.SpatialConvolution(1024, 2, 3, 3, 1, 1, 1, 1)
  local Con1_up = Con1 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)

  local DeCon1  = h5 - nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1)
                     - nn.ReLU()

  local ConCat1 = nn.JoinTable(1)({h4, Con1_up, DeCon1})

  -- Deconvolution and Concatnate stage 2 
  local Con2    = ConCat1 - nn.SpatialConvolution(1026, 2, 3, 3, 1, 1, 1, 1)
  local Con2_up = Con2 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)

  local DeCon2  = ConCat1 - nn.SpatialFullConvolution(1026, 256, 4, 4, 2, 2, 1, 1)
                          - nn.ReLU()

  local ConCat2 = nn.JoinTable(1)({h3, Con2_up, DeCon2})

  -- Deconvolution and Concatnate stage 3 
  local Con3    = ConCat2 - nn.SpatialConvolution(770, 2, 3, 3, 1, 1, 1, 1)
  local Con3_up = Con3 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)

  local DeCon3  = ConCat2 - nn.SpatialFullConvolution(770, 128, 4, 4, 2, 2, 1, 1)
                          - nn.ReLU()

  local ConCat3 = nn.JoinTable(1)({h2, Con3_up, DeCon3})

  -- Deconvolution and Concatnate stage 4 
  local Con4    = ConCat3 - nn.SpatialConvolution(386, 2, 3, 3, 1, 1, 1, 1)
  local Con4_up = Con4 - nn.SpatialFullConvolution(2, 2, 4, 4, 2, 2, 1, 1)

  local DeCon4  = ConCat3 - nn.SpatialFullConvolution(386, 64, 4, 4, 2, 2, 1, 1)
                          - nn.ReLU()

  local ConCat4 = nn.JoinTable(1)({h1, Con4_up, DeCon4})

  -- Final Convolution stage
  local Con5    = ConCat4 - nn.SpatialConvolution(194, 2, 3, 3, 1, 1, 1, 1)

--  -- loss(output) stage 1
--  local down1 = {flowIn, Con1} - nn.SpatialAdaptiveAveragePooling(8, 6)
--  local loss1 = {down1, Con1}  - nn.SmoothL1Criterion()
--  --table.insert(outputs, loss1)

--  -- loss(output) stage 2
--  local down2 = {flowIn, Con2} - nn.SpatialAdaptiveAveragePooling(16, 12)
--  local loss2 = {down2, Con2}  - nn.SmoothL1Criterion()
--  --table.insert(outputs, loss2)

--  -- loss(output) stage 3
--  local down3 = {flowIn, Con3} - nn.SpatialAdaptiveAveragePooling(32, 24)
--  local loss3 = {down3, Con3}  - nn.SmoothL1Criterion()
--  --table.insert(outputs, loss3)

--  -- loss(output) stage 4
--  local down4 = {flowIn, Con4} - nn.SpatialAdaptiveAveragePooling(64, 48)
--  local loss4 = {down4, Con4}  - nn.SmoothL1Criterion()
--  --table.insert(outputs, loss4)

--  -- loss(output) stage 5
--  local down5 = flowIn - nn.SpatialAdaptiveAveragePooling(128, 96)
--  local loss5 = nn.SmoothL1Criterion()({down5, Con5})

  table.insert(outputs, Con5)
--  table.insert(outputs, loss5)

  return nn.gModule(inputs, outputs)
end


