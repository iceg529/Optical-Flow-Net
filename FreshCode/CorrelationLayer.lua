require 'torch'  -- commented as main file expected with all requirements added
require 'nn'
local CorrelationLayer, parent = torch.class('nn.CorrelationLayer', 'nn.Module')

function CorrelationLayer:__init(nInputChannels, patchSize)   
   parent.__init(self)
   self.pSize = patchSize
   self.inSize = nInputChannels
   self.patch1 = torch.CudaTensor(1,self.inSize)
   self.patch2 = torch.CudaTensor(self.pSize,self.inSize)
   self.tempIn1 = torch.CudaTensor()
   self.tempIn2 = torch.CudaTensor()
   self.subGrad = torch.CudaTensor(self.pSize, self.inSize):fill(0)
   self.gradPatch = torch.CudaTensor(1, self.pSize):fill(0)
end

function CorrelationLayer:updateOutput(input)   
   self.tempIn1 = torch.CudaTensor((input[1]:size(1))+20,(input[1]:size(2))+20,input[1]:size(3)):fill(0)
   self.tempIn2 = torch.CudaTensor((self.tempIn1):size()):fill(0)
   -- fill the original area with original values
   self.tempIn1:sub(11,10+input[1]:size(1),11,10+input[1]:size(2)):copy(input[1])
   self.tempIn2:sub(11,10+input[2]:size(1),11,10+input[2]:size(2)):copy(input[2]) 
   self.output = torch.CudaTensor(input[1]:size(1),input[1]:size(2),self.pSize)
   for i=1,input[1]:size(1) do
     for j=1,input[1]:size(2) do
	(self.patch1)[1] = input[1][i][j]
	self.patch2 = torch.reshape(((self.tempIn2):sub(i,i+20,j,j+20)), self.pSize, self.inSize)
   	self.output[i][j] = torch.mm(self.patch1, (self.patch2):transpose(1,2))[1]
     end
   end
   return self.output
end

function CorrelationLayer:updateGradInput(input, gradOutput)
   self.tempGrad = torch.CudaTensor((gradOutput:size(1))+20,(gradOutput:size(2))+20,gradOutput:size(3)):fill(0)
   -- fill the original area with original values
   self.tempGrad:sub(11,10+gradOutput:size(1),11,10+gradOutput:size(2)):copy(gradOutput)
   
   self.gradInput = torch.CudaTensor(2,input[1]:size(1),input[1]:size(2),input[1]:size(3))
   for i=1,input[1]:size(1) do
     for j=1,input[1]:size(2) do
  	self.subGrad = torch.reshape(((self.tempIn2):sub(i,i+20,j,j+20)),self.pSize, self.inSize)
	self.gradPatch[1] = gradOutput[i][j]
  	self.gradInput[1][i][j] = ((torch.mm(self.gradPatch, self.subGrad))[1])
  	
	self.subGrad = torch.reshape(((self.tempIn1):sub(i,i+20,j,j+20)),self.pSize, self.inSize)
	self.gradPatch[1] = (torch.diag(torch.reshape(((self.tempGrad):sub(i,i+20,j,j+20)),self.pSize, self.pSize)))
  	self.gradInput[2][i][j] = ((torch.mm(self.gradPatch, self.subGrad))[1])
     end
   end
   local gradOutSize = gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3)
   self.grandInput:div(gradOutSize)  
   return self.gradInput
end

--[[function CorrelationLayer:updateGradInput(input, gradOutput)
   self.gradInput = torch.Tensor(2,input[1]:size(1),input[1]:size(2),input[1]:size(3))
   local input[1] = input[1]  
   for i=1,input[1]:size(1) do
     for j=1,input[1]:size(2) do
       for k=1,input[1]:size(3) do
	  self:resetSubGrad()
	  self.subGrad[i][j] = torch.reshape(((self.tempIn2):sub(i,i+20,j,j+20,k,k)),self.pSize)
	  self.gradInput[1][i][j][k] = (torch.cmul(self.subGrad, gradOutput[1])):mean()

	  self:resetSubGrad()
	  local cnt = 1
	  for l=i,i+20 do
     	    for m=j,j+20 do
	      self.subGrad[l][m][cnt] = (self.tempIn1)[i][j][k]
	      cnt = cnt + 1
	    end
	  end
	  self.gradInput[2][i][j][k] = (torch.cmul(self.subGrad, gradOutput[2])):mean()
       end
     end
   end   
   return self.gradInput
end--]]

function CorrelationLayer:resetSubGrad()
   self.subGrad = torch.CudaTensor(self.pSize, self.inSize):fill(0)
   self.gradPatch = torch.CudaTensor(1, self.pSize):fill(0)
end

