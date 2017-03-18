require 'nn'
require 'dpnn'

local VAE = {}

function VAE.get_encoder(channels, naf, z_dim, imsize)
    encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(channels, naf, 4, 4, 2, 2, 1, 1)) -- 1/2 size
    encoder:add(nn.ReLU())
    
    local num = torch.log(imsize)/torch.log(2) - 4 -- IF anyone has a better way go ahead and contribute!
    print(num)
    for i=0,num do
      j=2^i;
      encoder:add(nn.SpatialConvolution(naf * j, naf * 2 * j, 4, 4, 2, 2, 1, 1)) -- 1/2 size
      encoder:add(nn.SpatialBatchNormalization(naf * 2 * j)):add(nn.ReLU())
    end
    
    -- size here = size/2^(num) should be 4

    zLayer = nn.ConcatTable()
    zLayer:add(nn.SpatialConvolution(naf * 2^(num+1), z_dim, 4, 4))
    zLayer:add(nn.SpatialConvolution(naf * 2^(num+1), z_dim, 4, 4))
    encoder:add(zLayer)
    
    return encoder
end

function VAE.get_sampler()
    epsilonModule = nn.Sequential()
    epsilonModule:add(nn.MulConstant(0))
    epsilonModule:add(nn.WhiteNoise(0, 0.01))

    noiseModule = nn.Sequential()
    noiseModuleInternal = nn.ConcatTable()
    stdModule = nn.Sequential()
    stdModule:add(nn.MulConstant(0.5)) -- Compute 1/2 log σ^2 = log σ
    stdModule:add(nn.Exp()) -- Compute σ
    noiseModuleInternal:add(stdModule) -- Standard deviation σ
    noiseModuleInternal:add(epsilonModule) -- Sample noise ε
    noiseModule:add(noiseModuleInternal)
    noiseModule:add(nn.CMulTable())

    sampler = nn.Sequential()
    samplerInternal = nn.ParallelTable()
    samplerInternal:add(nn.Identity())
    samplerInternal:add(noiseModule)
    sampler:add(samplerInternal)
    sampler:add(nn.CAddTable())
    
    return sampler
end

function VAE.get_decoder(channels, ngf, z_dim, imsize)
    local num = torch.log(imsize)/torch.log(2) - 4 -- IF anyone has a better way go ahead and contribute!
    print(num)
  
    decoder = nn.Sequential()
    decoder:add(nn.SpatialFullConvolution(z_dim, ngf * 2^(num+1), 4, 4))
    decoder:add(nn.SpatialBatchNormalization(ngf * 2^(num+1))):add(nn.ReLU(true))
    
    for i=num,0,-1 do
      j=2^i
      decoder:add(nn.SpatialFullConvolution(ngf * j *2, ngf * j, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(ngf * j)):add(nn.ReLU(true))
    end

    decoder:add(nn.SpatialFullConvolution(ngf, channels, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.Sigmoid())
    
    return decoder
end

return VAE
