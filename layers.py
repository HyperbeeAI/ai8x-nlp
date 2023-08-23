###########################################################################
# NLP demo software by HyperbeeAI.                                        #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
license_statement = "NLP demo software by HyperbeeAI. Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai"
print("imported layers.py")
print(license_statement)
print("")

import torch, sys
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from functions import quantization, clamping_hw, linear_functional

class ai85_base(nn.Module):
    def __init__(
            self,
            operation_module  = None,
            operation_fcnl    = None,
            activation_module = None,
            output_width_30b  = False
    ):
        super().__init__()
        self.op               = operation_module
        self.op_fcn           = operation_fcnl
        self.act              = activation_module
        self.wide             = output_width_30b
        self.quantize_Q_d_8b    = None
        self.quantize_Q_u_wb    = None
        self.quantize_Q_d_wide  = None
        self.clamp_C_hw_8b      = None
        self.clamp_C_hw_wide    = None
        self.output_shift        = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) 
        self.weight_bits         = nn.Parameter(torch.Tensor([ 8 ]), requires_grad=False) 
        self.bias_bits           = nn.Parameter(torch.Tensor([ 8 ]), requires_grad=False) 
        self.quantize_activation = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) 
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) 
        self.shift_quantile      = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False)
        weight_bits      = self.weight_bits
        bias_bits        = self.bias_bits
        shift_quantile   = self.shift_quantile
        self.configure_layer_base( weight_bits, bias_bits, shift_quantile )

    def configure_layer_base(self, weight_bits, bias_bits, shift_quantile):
        self.quantize_Q_d_8b    = quantization(xb = 8,           mode ='down'   , wide=False) # 8 here is activation bits
        self.quantize_Q_u_wb    = quantization(xb = weight_bits, mode ='up'     , wide=False)
        self.quantize_Q_d_wide  = quantization(xb = 8,           mode ='down'   , wide=True)  # 8 here is activation bits, but its wide, so check inside
        self.clamp_C_hw_8b    = clamping_hw(xb = 8,           wide=False) # 8 here is activation bits
        self.clamp_C_hw_wide  = clamping_hw(xb = None,        wide=True)  # None to avoid misleading info on the # of bits, check inside
        self.weight_bits     = nn.Parameter(torch.Tensor([ weight_bits    ]), requires_grad=False)
        self.bias_bits       = nn.Parameter(torch.Tensor([ bias_bits      ]), requires_grad=False)
        self.shift_quantile  = nn.Parameter(torch.Tensor([ shift_quantile ]), requires_grad=False)

    def forward(self, x):
        w = self.op.weight
        b = self.op.bias
        los  = self.output_shift
        s_o  = 2**(los)
        w_q = self.quantize_Q_u_wb(w);
        b_q = self.quantize_Q_u_wb(b);

        x = self.op_fcn(x, w_q, b_q, self.op.stride, self.op.padding) # convolution / linear
        x = x*s_o
        if(self.act is not None):
            x = self.act(x)
        if((self.wide) and (self.act is None)):
            x = self.quantize_Q_d_wide(x)
            x = self.clamp_C_hw_wide(x)
            ### The +5 here is the 5 fractional bits the chip adds to the number in wide mode
            ### we divide the number back here to get it back into range. ai8x-training does not do this for some reason 
            ### until the synthesis/deployment phase, and they do a +1 bit, why?
            x = x / (2**(5)); # this is simulation of chip behavior
            x = x / 128.0     # this is ours, for convenience + this part is done outside the chip since it's the step before table lookup
            x = x / 2.0;      # this is ours, for convenience + this part is done outside the chip since it's the step before table lookup
        else:
            x = self.quantize_Q_d_8b(x)
            x = self.clamp_C_hw_8b(x)

        return x

class ai85_conv1d(ai85_base):
    def __init__(
            self,
            C_in_channels      = None,
            D_out_channels     = None,
            K_kernel_dimension = None,
            padding            = 0,
            activation         = None,
            output_width_30b   = False,
    ):

        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        else:
            print('wrong activation type in model. only {relu} is acceptable. exiting')
            sys.exit()

        operation_mdl  = nn.Conv1d(C_in_channels, D_out_channels, kernel_size=K_kernel_dimension, stride=1, padding=padding, bias=True);
        operation_fcn  = nn.functional.conv1d

        super().__init__(
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            output_width_30b   = output_width_30b,
        )

class ai85_add(nn.Module):
    def __init__(self ):
        super().__init__()
        self.clamp_C_hw_8b    = clamping_hw( xb = 8, wide=False) # 8 here is activation bits

    def forward(self, x, res):
        x = self.clamp_C_hw_8b(x+res)
        return x

class ai85_fullyconnected(ai85_base):
    def __init__(
            self,
            in_features        = None,
            out_features       = None,
            activation         = None,    
            output_width_30b   = False):
 
        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        else:
            print('wrong activation type in model. only {relu} is acceptable. exiting')
            sys.exit()

        operation_mdl  = nn.Linear(in_features, out_features, bias=True);
        operation_fcn  = linear_functional

        super().__init__(
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            output_width_30b   = output_width_30b
        )
        # Define dummy arguments to make Linear and conv compatible in ai85_base.
        # the name "op" here refers to op in super, i.e., in base_layer
        self.op.stride = None
        self.op.padding = None

class lpre(nn.Module):
    def __init__(self):
        super().__init__()
        self.ee1 = nn.Embedding(16384, 64)
        self.ee2 = nn.Embedding(48, 64)
        self.quantize  = quantization(xb = 8, mode ='updown', wide=False)

    def forward(self, x, sp1, sp2, sb):
        pp= torch.arange(sp1, sp2).unsqueeze(0).repeat(sb, 1).to(x.device)
        ee2_d = self.ee2(pp)
        ee1_d = self.ee1(x)
        ed = ee1_d + ee2_d
        min_w = self.ee2.weight.data.min() + self.ee1.weight.data.min()
        max_w = self.ee2.weight.data.max() + self.ee1.weight.data.max()
        t = (ed - min_w) / (max_w - min_w)
        t = t.add(-0.5).mul(2.0)
        t = self.quantize(t)
        t = t.clamp(min= -1.0, max=1.0-(1.0/128.0))
        t = t.mul(2**(8-1)).add(0.5).floor().clamp(min=-128, max=127)
        return t.permute(0, 2, 1) 
