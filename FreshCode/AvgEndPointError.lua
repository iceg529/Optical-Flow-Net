--require 'torch'  -- commented as main file expected with all requirements added
--require 'nn'
local AvgEndPointError, parent = torch.class('nn.AvgEndPointError', 'nn.Criterion')

function AvgEndPointError:__init()
   parent.__init(self)
   self.eucDistance = torch.Tensor()
   self.output_tensor = torch.Tensor()
   self.gradTemp = torch.Tensor()
end

function AvgEndPointError:updateOutput(input, target)
   --print(input:size())
   --print(target:size())
   self.eucDistance = (input-target):pow(2)
   self.output_tensor:add(self.eucDistance[1], self.eucDistance[2]):sqrt()
   --self.output_tensor = self.eucDistance:sum()
   --self.output_tensor = torch.sqrt(self.output_tensor)
   self.output = (self.output_tensor:sum())/(self.output_tensor:numel())
   return self.output
end

function AvgEndPointError:updateGradInput(input, target)
   self.gradTemp:resizeAs(input)
   self.gradTemp[1] = self.output_tensor
   self.gradTemp[2] = self.gradTemp[1]
   self.gradInput = torch.cdiv((input-target), self.gradTemp)
   self.gradInput:div(self.output_tensor:numel())   
   return self.gradInput
end
