--require 'torch'  -- commented as main file expected with all requirements added
--require 'nn'
local AvgAngularError, parent = torch.class('nn.AvgAngularError', 'nn.Criterion')

function AvgAngularError:__init()
   parent.__init(self)
   self.eucDistance1 = torch.Tensor()
   self.eucDistance2 = torch.Tensor()
   self.output_tensor1 = torch.Tensor()
   self.output_tensor2 = torch.Tensor()
   self.output_tensor3 = torch.Tensor()
   self.output_tensor4 = torch.Tensor()
   self.output_tensor5 = torch.Tensor()
   self.gradTemp = torch.Tensor()
end

function AvgAngularError:updateOutput(input, target)
   --print(input:size())
   --print(target:size())
   self.output_tensor5:resizeAs(input[1]):fill(1)
   self.eucDistance1:resizeAs(input):copy(input):pow(2)
   self.output_tensor1:resizeAs(input[1]):copy(self.eucDistance1[1]):add(self.eucDistance1[2]):add(self.output_tensor5):sqrt()
   self.eucDistance2:resizeAs(input):copy(target):pow(2)
   self.output_tensor2:resizeAs(input[1]):copy(self.eucDistance2[1]):add(self.eucDistance2[2]):add(self.output_tensor5):sqrt()
   --self.output_tensor = self.eucDistance:sum()
   --self.output_tensor = torch.sqrt(self.output_tensor)
      
   self.output_tensor3:resizeAs(input):copy(input):cmul(target)
   
   self.output_tensor4:resizeAs(input[1]):copy(self.output_tensor3[1]):add(self.output_tensor3[2]):add(self.output_tensor5)--:acos()
   self.output_tensor4:cdiv(self.output_tensor1):cdiv(self.output_tensor2):clamp(-1,1)
   
   self.output_tensor4:acos()
   self.output = (self.output_tensor4:sum())/(self.output_tensor4:numel())*180/(math.pi)
   return self.output
end

function AvgAngularError:updateGradInput(input, target)
   return 12
end
