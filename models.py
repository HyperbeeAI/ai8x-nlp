###########################################################################
# NLP demo software by HyperbeeAI.                                        #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
license_statement = "NLP demo software by HyperbeeAI. Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai"
print("imported models.py")
print(license_statement)
print("")

import torch
import torch.nn as nn
import layers

class encoder_ai85cnn(nn.Module):
    def __init__(
            self,
            device,
            **kwargs
    ):
        super().__init__()
        self.cc0     = layers.ai85_conv1d(  64,      112,       1,           0, activation=None)
        self.cc1     = layers.ai85_conv1d( 112,      112,       3,           1, activation='relu')
        self.res1    = layers.ai85_add()
        self.cc2     = layers.ai85_conv1d( 112,      112,       3,           1, activation='relu')
        self.res2    = layers.ai85_add()
        self.cc3     = layers.ai85_conv1d( 112,      112,       3,           1, activation='relu')
        self.res3    = layers.ai85_add()
        self.cc4     = layers.ai85_conv1d( 112,      112,       3,           1, activation='relu')
        self.res4    = layers.ai85_add()
        self.cc5     = layers.ai85_conv1d( 112,      64 ,       1,           0, activation=None) 
        self.resg    = layers.ai85_add()
        self.device  = device

    def forward(self, x):
        r = self.cc0(x)
        t = self.cc1( r )
        r = self.res1(t, r)
        t = self.cc2( r )
        r = self.res2(t, r)
        t = self.cc3( r )
        r = self.res3(t, r)
        t = self.cc4( r )
        r = self.res4(t, r)
        t = self.cc5(r)
        y = self.resg(t, x)
        return y

class encoder(nn.Module):
    def __init__(
            self,
            device,
            **kwargs
    ):
        super().__init__()
        self.pre       = layers.lpre()
        self.cnn       = encoder_ai85cnn(device = device);
        self.device    = device

    def forward(self, x):
        ssb   = x.shape[0]
        sl    = x.shape[1]
        pre_d = self.pre(x, 0, sl, ssb)
        out   = self.cnn(pre_d)
        return out, pre_d

class decoder_ai85cnn_ccf(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op = layers.ai85_conv1d(   112,    64 ,       1,           0, activation=None, output_width_30b=True)

    def forward(self, x):
        y = self.op(x)
        return y

class decoder_ai85cnn_cpr(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer1   = layers.ai85_conv1d( 64*2,  64,       1,           0,       activation='relu')
        self.layer2   = layers.ai85_conv1d( 64,    64,       1,           0,       activation='relu') 

    def forward(self, x):
        x = self.layer1(x)
        y = self.layer2(x)
        return y

class decoder_ai85cnn_cl1(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op    = layers.ai85_conv1d( 112,     112,       3,           0, activation='relu')

    def forward(self, x):
        y = self.op(x)
        return y

class decoder_ai85cnn_cma(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op = layers.ai85_conv1d(   64,     112,       1,           0, activation=None)
        self.res= layers.ai85_add()

    def forward(self, x, res):
        t = self.op(x)
        y = self.res(t, res)
        return y

class decoder_ai85cnn_claa(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op    = layers.ai85_conv1d( 112,     112,       3,           0, activation='relu')

    def forward(self, x):
        y = self.op(x)
        return y

class decoder_ai85cnn_cl0(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.op = layers.ai85_conv1d(   64,     112,       1,           0, activation=None)

    def forward(self, x):
        y = self.op(x)
        return y

class decoder_ai85cnn_clfa(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op    = layers.ai85_conv1d( 112,     112,       3,           0, activation='relu')

    def forward(self, x):
        y = self.op(x)
        return y

class decoder_ai85cnn_ccac(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op    = layers.ai85_conv1d( 112,     112,       3,           0, activation='relu')

    def forward(self, x):
        y = self.op(x)
        return y

class decoder_ai85cnn_cib(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.op = layers.ai85_conv1d(   112,    64 ,       1,           0, activation=None)

    def forward(self, x):
        y = self.op(x)
        return y

class decoder(nn.Module):
    def __init__(
            self,
            device, 
            tpi, 
            **kwargs
    ):
        super().__init__()

        self.device       = device
        self.tpi          = tpi
        self.pre          = layers.lpre()
        self.fff          = nn.Linear(64, 16384)
        self.fff.weight   = self.pre.ee1.weight    # i.e., fff is not a layer, this is just an easy way of doing reverse embedding on pytorch
        self.cl0          = decoder_ai85cnn_cl0();
        self.ccf          = decoder_ai85cnn_ccf();
        self.cib          = decoder_ai85cnn_cib();
        self.cma          = decoder_ai85cnn_cma();
        self.cpr          = decoder_ai85cnn_cpr();
        self.cl1          = decoder_ai85cnn_cl1();
        self.claa         = decoder_ai85cnn_claa();
        self.clfa         = decoder_ai85cnn_clfa();
        self.ccac         = decoder_ai85cnn_ccac();

    def forward(self, x, ees , pss=0):
        ssb = x.shape[0]
        sst = x.shape[1]
        sl  = ees.shape[2]
        
        pre_d          = self.pre(x, pss, sst + pss, ssb)
        t              = self.cl0(pre_d)
        cl0_out        = t
        ssb, ts1, _    = t.shape
        tp             = torch.zeros(ssb, ts1, 2).fill_(self.tpi).to(t.device)
        t              = torch.cat((tp, t), dim = 2)
        xconv          = self.cl1(t)
        t              = self.cib(xconv)
        ssb, ss_p, sst = t.shape
        x2             = ees.unsqueeze(3).repeat(1, 1, 1, sst).view(ssb, ss_p, -1)
        t              = t.unsqueeze(2).repeat(1, 1, sl, 1).view(ssb, ss_p, -1)
        t              = torch.cat([t, x2], dim=1)
        t              = self.cpr(t)
        t              = t.view(ssb, ss_p, sl, sst)
        t              = torch.max(t, dim=2).values
        t              = self.cma(t, xconv)
        t              = torch.cat((tp, t), dim = 2)
        xconv          = self.claa(t)
        t              = self.cib(xconv)
        t              = t.unsqueeze(2).repeat(1, 1, sl, 1).view(ssb, ss_p, -1)
        t              = torch.cat([t, x2], dim=1)
        t              = self.cpr(t)
        t              = t.view(ssb, ss_p, sl, sst)
        t              = torch.max(t, dim=2).values
        t              = self.cma(t, xconv)
        t              = torch.cat((tp, t), dim = 2)
        xconv          = self.clfa(t)
        t              = self.cib(xconv)
        t              = t.unsqueeze(2).repeat(1, 1, sl, 1).view(ssb, ss_p, -1)
        t              = torch.cat([t, x2], dim=1)
        t              = self.cpr(t)
        t              = t.view(ssb, ss_p, sl, sst)
        t              = torch.max(t, dim=2).values
        t              = self.cma(t, xconv)
        t              = torch.cat((tp, t), dim = 2)
        xconv          = self.ccac(t)
        t              = self.cib(xconv)
        t              = t.unsqueeze(2).repeat(1, 1, sl, 1).view(ssb, ss_p, -1)
        t              = torch.cat([t, x2], dim=1)
        t              = self.cpr(t)
        t              = t.view(ssb, ss_p, sl, sst)
        t              = torch.max(t, dim=2).values
        t              = self.cma(t, xconv)
        pss            = t + sst
        ccf_out        = self.ccf(t)
        output         = self.fff(ccf_out.permute(0, 2, 1))

        return output, pre_d, ccf_out

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        enc_out, _ = self.encoder(src)
        output, _, _  = self.decoder(trg, enc_out)
        return output

