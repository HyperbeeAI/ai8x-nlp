###########################################################################
# NLP demo software by HyperbeeAI.                                        #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
license_statement = "NLP demo software by HyperbeeAI. Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai"
print("imported functions.py")
print(license_statement)
print("")

import torch, sys
import torch.nn as nn
from torch.autograd import Function

class Q_ud(Function):
    @staticmethod
    def forward(_, x, xb):
        factor = 2**(xb-1)
        return x.mul(factor).add(.5).floor().div(factor)

class Q_u(Function):
    @staticmethod
    def forward(_, x, xb):
        factor = 2**(8-xb)
        return x.mul(factor).add(.5).floor()

class Q_d(Function):
    @staticmethod
    def forward(_, x, xb):
        factor = 2**(xb-1)
        return x.div(factor).add(.5).floor()

class quantization(nn.Module):
    def __init__(self, xb = 8, mode='updown', wide=False):
        super().__init__()
        self.xb   = xb
        self.mode = mode
        self.wide = wide

    def forward(self, x):
        if(self.mode=='updown'):
            return Q_ud.apply(x, self.xb)
        elif(self.mode=='down'):
            if(self.wide):
                return Q_d.apply(x, self.xb - 5)
            else:
                return Q_d.apply(x, self.xb)
        elif(self.mode=='up'):
            return Q_u.apply(x, self.xb)
        else:
        	print('wrong quantization mode. exiting')
        	sys.exit()

class clamping_hw(nn.Module):
    def __init__(self, xb = 8, wide=False):
        super().__init__()
        if(wide):
            self.min_val = -2**(30-1)  
            self.max_val =  2**(30-1)-1
        else:
            self.min_val = -2**(xb-1)
            self.max_val =  2**(xb-1)-1

    def forward(self, x):
        return x.clamp(min=self.min_val, max=self.max_val)

###################################################
### Linear layer functional
def linear_functional(x, weight, bias, _stride, _padding):
    # dummy linear function that has same arguments as conv
    return nn.functional.linear(x, weight, bias)
