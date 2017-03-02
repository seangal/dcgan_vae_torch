local VAEGANCriterion, parent = torch.class('nn.VAEGANCriterion', 'nn.Criterion')

function VAEGANCriterion:__init()
  self.BCEC1 = nn.BCECriterion():cuda()
  self.BCEC2 = nn.BCECriterion():cuda()
  self.BCEC3 = nn.BCECriterion():cuda()
  self.MSEC = nn.MSECriterion():cuda()
end

function VAEGANCriterion:updateOutput(input, target)
  -- input is {DReal,DGeneratedRandom,GGeneratedRandom, VAEImage, EncoderOutput1,EncoderOutput2}
  
  
  self.BCEC1:updateOutput(input[1],torch.ones(input[1]:size(1)):cuda())
  self.errD = self.BCEC1.output
  self.BCEC2:updateOutput(input[2],torch.zeros(input[2]:size(1)):cuda())
  self.errD = self.errD + self.BCEC2.output
  
  self.BCEC3:updateOutput(input[2],torch.ones(input[2]:size(1)):cuda())
  self.errG = self.BCEC3.output
  
  self.MSEC:updateOutput(input[4],target)
  self.errVAE = self.MSEC.output;
  
  self.output = self.errD --+ self.errG-- + self.errVAE;
  return self.output;
end

function VAEGANCriterion:updateGradInput(input, target)
  self.gradInput = {}
  self.BCEC1:updateGradInput(input[1],torch.ones(input[1]:size(1)):cuda())
  self.gradInput[1] = self.BCEC1.gradInput;
  
  self.BCEC2:updateGradInput(input[2],torch.zeros(input[2]:size(1)):cuda())
  self.gradInput[2] = self.BCEC2.gradInput;
  --print("GGG2",self.gradInput[2] )
  
  self.BCEC3:updateGradInput(input[2],torch.ones(input[2]:size(1)):fill(2):cuda())
  self.gradInput[3] = self.BCEC3.gradInput;
  
  self.MSEC:updateGradInput(input[4],target)
  self.gradInput[4] = self.MSEC.gradInput;
  
  local output = input[4]
  nElements = output:nElement()
  mean, log_var = table.unpack({input[5],input[6]})
  var = torch.exp(log_var)
  KLLoss = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - var)
  KLLoss = KLLoss / nElements
  gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}
  self.gradInput[5] = gradKLLoss[1]
  self.gradInput[6] = gradKLLoss[2]
  --print(self.gradInput[1]:mean(),self.gradInput[2]:mean()) 
  
  return self.gradInput;
end