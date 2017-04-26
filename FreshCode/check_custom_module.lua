require 'AvgEndPointError'


-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)

local input = torch.Tensor(2,2):fill(3)
local module = nn.AvgEndPointError()

-- test backprop, with Jacobian
local err = module:forward(input, input*3)
local errGrad = module:backward(input, input*2)

print('==> error: ' .. err)
print(errGrad[1][1])

