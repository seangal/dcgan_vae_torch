require 'nn'

local discriminator = {}

function discriminator.get_discriminator(channels, ndf, imsize)
    netD = nn.Sequential()
    netD:add(nn.SpatialConvolution(channels, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    
    local num = torch.log(imsize)/torch.log(2) - 4 -- IF anyone has a better way go ahead and contribute!
    print(num)
    for i=0,num do
      j=2^i;
      netD:add(nn.SpatialConvolution(ndf * j, ndf * j * 2, 4, 4, 2, 2, 1, 1))
      netD:add(nn.SpatialBatchNormalization(ndf * j * 2)):add(nn.LeakyReLU(0.2, true))
    end
    netD:add(nn.SpatialConvolution(ndf * 2^(num+1), 1, 4, 4))
    netD:add(nn.Sigmoid())
    netD:add(nn.View(1):setNumInputDims(3))
    
    return netD
end

return discriminator
