require 'image'
require 'xlua'
require 'nn'
require 'dpnn'
require 'optim'
require 'lfs'
require 'nngraph'
require 'VAEGANCriterion'
require 'BlockBP'
require 'MyMapTable'


tnt = require 'torchnet'
local VAE = require 'VAE'
local discriminator = require 'discriminator'
local disp = require 'display'
require 'cunn'

local function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 (Torchnet) Imagenet Training script')
   cmd:text()
   cmd:text('Options:')
   cmd:text('---------- General options ----------------------------------')
   cmd:text()
   cmd:option('-i',        'images',      'Home of images')
   cmd:option('-o',        'output',      'Home of output')
   cmd:option('-GPU',         1,             'Default preferred GPU < only 1 if dpt')
   cmd:option('-nGPU',        1,             'Number of GPUs to use')
   cmd:option('-cudnn',       'fastest',     'Options: fastest | deterministic')
   cmd:option('-manualSeed',  2,             'Manually set RNG seed')
   cmd:text()
   cmd:text('---------- Data options ----------------------------------')
   cmd:text()
   cmd:option('-nDonkeys',   8,    'number of data loading threads')
   cmd:option('-imageSize',  64,  'Smallest side of the resized image')
   cmd:text()
   cmd:text('---------- Training options ----------------------------------')
   cmd:text()
   cmd:option('-nEpochs',    55,   'Number of total epochs to run')
   cmd:option('-saveEpoch',  10,   'save each <VALUE> epochs')
   cmd:option('-batchSize',  128,  'mini-batch size (1 = pure stochastic)')
   cmd:text()
   cmd:text('---------- Optimization options ----------------------------------')
   cmd:text()
   cmd:option('-LR',         0.0002,  'learning rate; 0 - default LR/WD recipe')
   cmd:option('-momentum',   0.9,  'momentum')
   cmd:option('-WD',         5e-4, 'weight decay')
   cmd:text()
   cmd:text('---------- Network options ----------------------------------')
   cmd:text()
   cmd:option('-zDim',       100,  'Z dimension')
   cmd:option('-featureSize',64,   'number of features after first conv')
   cmd:text()
   cmd:text('---------- Resume/Finetune options ----------------------------------')
   cmd:text()
   cmd:option('-network',    'none', 'model to retrain with')
   cmd:option('-epoch',       0,     'epochs completed (for LR Policy)')
   cmd:text()

   local opt = cmd:parse(arg or {})
   return opt
end

local opt = parse(arg)
print(opt)

function getFilenames(path)
    queue = {}
    count = 1
    for file in lfs.dir(path) do
        if file ~= '.' and file ~= '..' then
            queue[count] = file
            count = count + 1
        end
    end
    return queue
end

function getNumber(num)
  length = #tostring(num)
  filename = ""
  for i=1, (6 - length) do
    filename = filename .. 0
  end
  filename = filename .. num
  return filename
end

filenames = getFilenames(opt.i)




RealDataset = tnt.BatchDataset{
  dataset = tnt.ListDataset{
    list = filenames,
    load = function(name)
      local input = image.scale(image.load(name),opt.imageSize,opt.imageSize):cuda()
      return {input=input}
    end,
    path = opt.i
  },
  batchsize=opt.batchSize
}

Ddataset = tnt.TransformDataset{
  dataset = RealDataset,
  transform = function(sample)
    local noise_x = torch.Tensor(sample.input:size(1), opt.zDim, 1, 1)
    noise_x = noise_x:cuda()
    noise_x:normal(0, 0.01)
    --return {input=torch.cat(sample.input, decoder:forward(noise_x),1),target=torch.cat(torch.ones(sample.input:size(1)),torch.ones(sample.input:size(1)):fill(2),1)}
    --return {input={sample.input:cuda(), noise_x:cuda()},target=torch.cat(torch.ones(sample.input:size(1)),torch.zeros(sample.input:size(1)),1):cuda()}
    return {input={sample.input:cuda(), noise_x:cuda()},target=sample.input:cuda()}
  end
}

-- NETWORK STUFF

dNoise = .1

ndf = opt.featureSize
ngf = opt.featureSize
naf = opt.featureSize

if opt.network == 'none' then
  encoder = VAE.get_encoder(3, naf, opt.zDim, opt.imageSize)
  sampler = VAE.get_sampler()
  decoder = VAE.get_decoder(3, ngf, opt.zDim, opt.imageSize)

  netD = discriminator.get_discriminator(3, ndf, opt.imageSize)

  netD = netD:cuda()
  --cudnn.convert(netG, cudnn)
  --cudnn.convert(netD, cudnn)
  --input = nn.Identity()()
  
  
  iInput = - nn.Identity()
  zInput = - nn.Identity()
  --G=netG(input)
  L1 = iInput - encoder
  L2 = {L1 - sampler, zInput} - nn.MapTable():add(decoder)
  L3 = {iInput - nn.WhiteNoise(0, dNoise), L2 - nn.SelectTable(2) - nn.BlockBP(), L2 - nn.SelectTable(2)} - nn.MyMapTable():add(netD)
  L4 = L2 - nn.SelectTable(1)
  O = {L3, L4, L1} - nn.FlattenTable()
  --O=nn.JoinTable(1)(D)
  model = nn.gModule({iInput,zInput}, {O}):cuda()
  
  input = {torch.randn(32,3,opt.imageSize,opt.imageSize):cuda(),torch.randn(32,opt.zDim,1,1):cuda()}
  print(input)
  print(model:forward(input))
  graph.dot(model.fg, 'Big MLP','network')
else
  model = torch.load(opt.network)
  graph.dot(model.bg, 'Big MLP','back')
end
GDcriterion = nn.BCECriterion()
GDcriterion = GDcriterion:cuda()

AEcriterion = nn.MSECriterion()
AEcriterion = AEcriterion:cuda()


--noise to pass through decoder to generate random samples from Z
noise_x = torch.Tensor(opt.batchSize, opt.zDim, 1, 1)
noise_x = noise_x:cuda()
noise_x:normal(0, 0.01)

local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
   state.epoch = opt.epoch or 0
end

engine.hooks.onStartEpoch = function(state)
   timers.epochTimer:reset()
end

engine.hooks.onSample = function(state)
   cutorch.synchronize()
   timers.dataTimer:stop()
end
gdata = {}
meter = tnt.ClassErrorMeter{topk={1}}
engine.hooks.onForwardCriterion = function(state)
  if state.training then
    meter:reset()
    print(('Epoch:%d [%d]   [Data/BatchTime %.3f/%.3f]   LR %.0e   ErrG %.4f  ErrD %.4f  ErrVAE %.4f'):format(state.epoch+1, state.t, timers.dataTimer:time().real,timers.batchTimer:time().real, state.config.learningRate,state.criterion.errG,state.criterion.errD,state.criterion.errVAE))
    
    a = torch.round(nn.JoinTable(1):cuda():forward({state.network.output[1],state.network.output[2]}))
    
    meter:add(torch.cat(a,torch.ones(a:size(1),a:size(2)):cuda() - a,2),torch.cat(torch.ones(a:size(1)/2),torch.zeros(a:size(1)/2):fill(2),1):cuda())
    
    table.insert(gdata,{state.t,meter:value(1)})
    disp.plot(gdata,{title="PLT",win=4,labels = {"epoch", "1"},ylabel = "ratio"})
    disp.image(state.network.output[4],{title="VAE",win=2})
    --disp.image(state.network.forwardnodes[11].data.module.output[2],{title="generations",win=1})
    
    timers.batchTimer:reset() -- cycle can start anywhere
  end
end

engine.hooks.onUpdate = function(state)
   cutorch.synchronize()
   timers.dataTimer:reset()
   timers.dataTimer:resume()
   --graph.dot(model.bg, 'Big MLP','back')
end

engine.hooks.onEndEpoch = function(state)
  print("Total Epoch time (Train):",timers.epochTimer:time().real)
  if state.epoch % 10 == 0 then dNoise = dNoise * 0.99 end
  if state.epoch % opt.saveEpoch == 0 then
    torch.save(paths.concat(opt.o,'model_' .. state.epoch ..'.t7'),state.network)
  end
end

local iterator = tnt.DatasetIterator{
  --nthread = opt.nDonkeys,
  dataset = Ddataset
}

engine:train{
     network = model,
     criterion = nn.VAEGANCriterion():cuda(),
     iterator = iterator,
     optimMethod = optim.sgd,
     config = {
        learningRate = opt.LR,
        momentum = 0.9,
        beta1 = 0.5
     },
  }
