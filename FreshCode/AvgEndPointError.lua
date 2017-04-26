--require 'torch'  -- commented as main file expected with all requirements added
--require 'nn'
local AvgEndPointError, parent = torch.class('nn.AvgEndPointError', 'nn.Criterion')

function AvgEndPointError:__init()
   parent.__init(self)
   self.eucDistance = torch.Tensor()
   self.output_tensor = torch.Tensor()
   self.new = torch.Tensor()
end

function AvgEndPointError:updateOutput(input, target)
   print(input:size())
   print(target:size())
   self.new = torch.add(input,-1,target)
   print(self.new:size())
   self.eucDistance = (input-target):pow(2)
   self.output_tensor = self.eucDistance:sum()
   self.output_tensor = torch.sqrt(self.output_tensor)
   self.output = (self.output_tensor)/(self.eucDistance:numel())
   return self.output
end

function AvgEndPointError:updateGradInput(input, target)
   self.gradInput = (input-target):div(self.eucDistance:numel()):div(self.output_tensor)   
   return self.gradInput
end
