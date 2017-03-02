local MyMapTable, parent = torch.class('nn.MyMapTable', 'nn.Container')

function MyMapTable:__init(module, shared)
   parent.__init(self)
   self.shared = shared or {'weight', 'bias', 'gradWeight', 'gradBias'}
   self.output = {}
   self.gradInput = {}
   self:add(module)
end

function MyMapTable:_extend(n)
   self.modules[1] = self.module
   for i = 2, n do
      if not self.modules[i] then
         self.modules[i] = self.module:clone(table.unpack(self.shared))
      end
   end
end

function MyMapTable:resize(n)
   self:_extend(n)
   for i = n + 1, #self.modules do
      -- It's not clear why this clearState call is necessary, but it fixes
      -- https://github.com/torch/nn/issues/1141 .
      self.modules[i]:clearState()
      self.modules[i] = nil
   end
end

function MyMapTable:add(module)
   assert(not self.module, 'Single module required')
   self.module = module
   self.modules[1] = self.module
   return self
end

function MyMapTable:updateOutput(input)
   self.output = {}
   self:_extend(#input)
   for i = 1, #input do
      self.output[i] = self:rethrowErrors(self.modules[i], i, 'updateOutput', input[i])
   end
   return self.output
end

function MyMapTable:updateGradInput(input, gradOutput)
   self.gradInput = {}
   self:_extend(#input)
   for i = 1, #input do
      self.gradInput[i] = self:rethrowErrors(self.modules[i], i, 'updateGradInput', input[i], gradOutput[i])
   end
   return self.gradInput
end

function MyMapTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self:_extend(#input)
   for i = 1, #input-1 do
      self:rethrowErrors(self.modules[i], i, 'accGradParameters', input[i], gradOutput[i], scale)
   end
end

function MyMapTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   self:_extend(#input)
   for i = 1, #input do
      self:rethrowErrors(self.modules[i], i, 'accUpdateGradParameters', input[i], gradOutput[i], lr)
   end
end

function MyMapTable:zeroGradParameters()
    if self.module then
        self.module:zeroGradParameters()
    end
end

function MyMapTable:updateParameters(learningRate)
    if self.module then
        self.module:updateParameters(learningRate)
    end
end

function MyMapTable:clearState()
   for i = 2, #self.modules do
      -- It's not clear why this clearState call is necessary, but it fixes
      -- https://github.com/torch/nn/issues/1141 .
      self.modules[i]:clearState()
      self.modules[i] = nil
   end
   parent.clearState(self)
end

function MyMapTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local extlast = '      '
   local str = torch.type(self)
   if self.module then
      str = str .. ' {' .. line .. tab
      str = str .. tostring(self.module):gsub(line, line .. tab .. extlast) .. line .. '}'
   else
      str = str .. ' { }'
   end
   return str
end