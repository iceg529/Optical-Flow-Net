require 'torch'
require 'nn'
require 'nngraph'

criterion = nn.Sequential()
criterion:add(nn.SpatialAdaptiveAveragePooling(128, 96))
criterion:add(nn.SmoothL1Criterion())