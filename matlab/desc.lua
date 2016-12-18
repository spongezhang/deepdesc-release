require 'nn'
require 'cunn'
require 'mattorch'

local cmd = torch.CmdLine()
cmd:option( '--model', '../models/CNN3_p8_n8_split4_073000.t7', 'Network model to use' )
local params = cmd:parse(arg)

-- load model and mean
local data = torch.load( params.model )
local desc = data.desc
local mean = data.mean
local std  = data.std

print desc
print mean
print std

desc:cuda()

torch.save('../models/CNN3_p8_n8_split4_073000_model_only'..'.t7',desc)
