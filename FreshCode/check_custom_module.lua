require 'torch'
require 'nn'
require 'CorrelationLayer'
require 'xlua'
require 'cunn'
require 'cutorch'

profiler = xlua.Profiler(false, true)
-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)

local input = {torch.CudaTensor(48,64,256):fill(-2), torch.CudaTensor(48,64,256):fill(5)}
local module = nn.CorrelationLayer():cuda()

-- test backprop, with Jacobian
--[[local err = module:forward(input, input*2)
local errGrad = module:backward(input, input*2)

print('==> error: ' .. err)
print(errGrad[2][1])--]]


-- test backprop, with Jacobian
local grad = torch.CudaTensor(input[1]:size(1),input[1]:size(2),441):fill(0.0034)
profiler:start('fwd-pass')
local corr = module:forward(input)
print(corr:mean())
profiler:lap('fwd-pass')
profiler:start('back-pass')
local gradIn = module:backward(input,grad)
profiler:lap('back-pass')
print(gradIn:max())
print(gradIn:min())
local err = jac.testJacobian(module,input)
print(err:size())
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end
