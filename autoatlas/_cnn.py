import torch
import collections

#class DwnPool(torch.nn.Module):
#    def __init__(mode,stride=2,filters=None):
#        if mode == 'maxpool': 
#            self.pool = torch.nn.MaxPool3d(kernel_size=stride,stride=stride,return_indices=True)
#        elif mode == 'cnn': 
#            self.pool = torch.nn.Conv3d(in_channels=filters,out_channels=filters,kernel_size=stride,stride=stride)
#            if filters is None:
#                raise ValueError('Filter/channel length must be specified for conv pool')
#        else:
#            raise ValueError('Only maxpool and cnn pooling are supported')

#    def forward(x):
#        if mode == 'maxpool':
#            x,y = self.pool(x) 
#            return x,y
#        elif mode == 'cnn':
#            return self.pool(x)

#class UpPool(torch.nn.Module):
#    def __init__(mode,stride=2,filters=None):
#        if mode == 'maxpool': 
#            self.pool = torch.nn.MaxUnpool3d(kernel_size=stride,stride=stride)
#        elif mode == 'cnn':
#            self.pool = torch.nn.ConvTranspose3d(in_channels=filters,out_channels=filters,kernel_size=stride,stride=stride,output_padding=out_pad)
#            if filters is None:
#                raise ValueError('Filter/channel length must be specified for conv pool')
#        else:
#            raise ValueError('Only maxpool and cnn pooling are supported')

class _UNet(torch.nn.Module):
    def __init__(self,num_labels,dim=3,data_chan=1,kernel_size=3,filters=32,blocks=7,layers_block=2,batch_norm=False,pad_type='SAME',enc_dev='cuda',dec_dev='cuda'):
        super(_UNet,self).__init__()
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.enc_dev = enc_dev
        self.dec_dev = dec_dev

        if self.blocks % 2 == 0:
            raise ValueError('Number of UNet blocks must be odd')

        self.pad_type = pad_type
        if pad_type == 'VALID':
            pad_width = 0
        elif pad_type == 'SAME':
            pad_width = 1
        else:
            raise ValueError("ERROR: pad_type must be either VALID or SAME")

        if dim == 2:
            self.conv = torch.nn.Conv2d
        elif dim == 3:
            self.conv = torch.nn.Conv3d
        else:
            raise ValueError("ERROR: dim for UNet must be either 2 or 3")
        
        self.encoders = torch.nn.ModuleList([_UNetEncoder(dim=dim,in_filters=data_chan,out_filters=filters,kernel_size=kernel_size,pad_width=pad_width,layers_block=layers_block,batch_norm=batch_norm)])
        for _ in range(blocks//2-1):
            enc = _UNetEncoder(dim=dim,in_filters=filters,out_filters=2*filters,kernel_size=kernel_size,pad_width=pad_width,layers_block=layers_block,batch_norm=batch_norm) 
            self.encoders.append(enc)
            filters *= 2
  
        self.decoders = torch.nn.ModuleList([_UNetDecoder(dim=dim,in_filters=filters,out_filters=filters,kernel_size=kernel_size,pad_width=pad_width,layers_block=layers_block,batch_norm=batch_norm)])
        for _ in range(blocks//2-1):
            dec = _UNetDecoder(dim=dim,in_filters=filters*2,out_filters=filters//2,kernel_size=kernel_size,pad_width=pad_width,layers_block=layers_block,batch_norm=batch_norm)
            self.decoders.append(dec)
            filters = filters//2
            #Division using // ensures return value is integer

        seq_lays = [self.conv(in_channels=filters*2,out_channels=filters,kernel_size=kernel_size,padding=pad_width)]
        seq_lays.append(torch.nn.ReLU(inplace=True)) 
        for i in range(layers_block-1):
            seq_lays.append(self.conv(in_channels=filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width)) 
            seq_lays.append(torch.nn.ReLU(inplace=True)) 
        seq_lays.append(self.conv(in_channels=filters,out_channels=num_labels,kernel_size=1,padding=0)) 
        self.output = torch.nn.Sequential(*seq_lays)

        self.encoders = self.encoders.to(self.enc_dev)
        self.decoders = self.decoders.to(self.dec_dev)
        self.output = self.output.to(self.dec_dev)

    def forward(self,x):
        features = []
        for enc in self.encoders:
            x,z = enc(x)
            features.append(z)

        x = self.decoders[0](x.to(self.dec_dev))
        for i,dec in enumerate(self.decoders[1:]):
            z = features.pop()
            z = z.to(self.dec_dev)
            if self.pad_type == 'VALID':
                cr = [(sz-sx)//2 for sz,sx in zip(z.size(),x.size())]
                #sz-sx is always even since up/down sample factor is 2
                z = z[:,:,cr[2]:-cr[2],cr[3]:-cr[3],cr[4]:-cr[4]] if self.dim==3 else z[:,:,cr[2]:-cr[2],cr[3]:-cr[3]]
            x = torch.cat((x,z),dim=1)
            x = dec(x)

        z = features.pop()
        z = z.to(self.dec_dev)
        if self.pad_type == 'VALID':
            cr = [(sz-sx)//2 for sz,sx in zip(z.size(),x.size())]
            z = z[:,:,cr[2]:-cr[2],cr[3]:-cr[3],cr[4]:-cr[4]] if self.dim==3 else z[:,:,cr[2]:-cr[2],cr[3]:-cr[3]]
        x = torch.cat((x,z),dim=1)
        return self.output(x)

class _UNetEncoder(torch.nn.Module):
    def __init__(self,dim,in_filters,out_filters,kernel_size,pad_width,layers_block,batch_norm=False):
        super(_UNetEncoder,self).__init__()
        
        if dim == 2:
            self.conv = torch.nn.Conv2d
            self.pool = torch.nn.MaxPool2d(2,padding=0)
        elif dim == 3:
            self.conv = torch.nn.Conv3d
            self.pool = torch.nn.MaxPool3d(2,padding=0)
        else:
            raise ValueError("ERROR: dim for UNet must be either 2 or 3")
       
        seq_lays = []
        for i in range(layers_block):
            in_channels = in_filters if i==0 else out_filters
            seq_lays.append(self.conv(in_channels=in_channels,out_channels=out_filters,kernel_size=kernel_size,padding=pad_width))
            seq_lays.append(torch.nn.ReLU(inplace=True))
        self.model = torch.nn.Sequential(*seq_lays)

    def forward(self,x):
        x = self.model(x)
        return self.pool(x),x

class _UNetDecoder(torch.nn.Module):
    def __init__(self,dim,in_filters,out_filters,kernel_size,pad_width,layers_block,batch_norm=False):
        super(_UNetDecoder,self).__init__()
       
        if dim == 2:
            self.conv = torch.nn.Conv2d
            self.conv_tran = torch.nn.ConvTranspose2d
        elif dim == 3:
            self.conv = torch.nn.Conv3d
            self.conv_tran = torch.nn.ConvTranspose3d
        else:
            raise ValueError("ERROR: dim for Decoder must be either 2 or 3")

        seq_lays = []
        for i in range(layers_block):
            in_channels = in_filters if i==0 else 2*out_filters
            seq_lays.append(self.conv(in_channels=in_channels,out_channels=2*out_filters,kernel_size=kernel_size,padding=pad_width))
            seq_lays.append(torch.nn.ReLU(inplace=True))
        seq_lays.append(self.conv_tran(in_channels=2*out_filters,out_channels=out_filters,kernel_size=2,padding=0,stride=2))
        self.model = torch.nn.Sequential(*seq_lays)

    def forward(self,x):
        return self.model(x)

class _AutoEnc(torch.nn.Module):
    def __init__(self,sizes,data_chan=1,kernel_size=3,filters=8,depth=5,pool=2,batch_norm=False,pool_type='conv',pad_type='SAME'):
        super(_AutoEnc,self).__init__()
        self.sizes = sizes
        dim = len(self.sizes)

        if pad_type == 'VALID':
            pad_width = 0
        elif pad_type == 'SAME':
            pad_width = kernel_size//2
        else:
            raise ValueError("ERROR: pad_type must be either VALID or SAME")

        if kernel_size%2==0:
            raise ValueError("ERROR: Kernel size must be odd")

        if depth%2==0:
            raise ValueError("ERROR: Depth parameter must be odd")

        if dim == 2:
            self.conv = torch.nn.Conv2d
            self.conv_tran = torch.nn.ConvTranspose2d
            self.max_pool = torch.nn.MaxPool3d(3,padding=0)
        elif dim == 3:
            self.conv = torch.nn.Conv3d
            self.conv_tran = torch.nn.ConvTranspose3d
            self.max_pool = torch.nn.MaxPool3d(2,padding=0)

        self.encoders = torch.nn.ModuleList([])
        stages = depth//2
        for i in range(stages):
            #in_filters = 2*data_chan if i==0 else filters
            in_filters = data_chan if i==0 else filters
            #if pool_type == 'conv':
            enc = torch.nn.Sequential(
                self.conv(in_channels=in_filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width,stride=pool),
                torch.nn.ReLU(inplace=True))
            #else:
            #    enc = torch.nn.Sequential(
            #        self.conv(in_channels=in_filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width),
            #        torch.nn.ReLU(inplace=True),
            #        self.max_pool)
            self.encoders.append(enc)

        if dim == 2:
            self.lin_features = int(filters*sizes[0]*sizes[1]/(pool**(2*stages)))
        elif dim == 3:
            self.lin_features = int(filters*sizes[0]*sizes[1]*sizes[2]/(pool**(3*stages)))
 
        self.linear_enc = torch.nn.Sequential(torch.nn.Linear(in_features=self.lin_features,out_features=filters),
                                     torch.nn.ReLU(inplace=True))   
        self.linear_dec = torch.nn.Sequential(torch.nn.Linear(in_features=filters,out_features=self.lin_features),
                                     torch.nn.ReLU(inplace=True))   
        
        self.decoders = torch.nn.ModuleList([])
        for i in range(stages):
            if i==stages-1:
                dec = self.conv_tran(in_channels=filters,out_channels=data_chan,kernel_size=kernel_size,padding=pad_width,stride=pool,output_padding=1) 
            else: 
                dec = torch.nn.Sequential(
                    self.conv_tran(in_channels=filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width,stride=pool,output_padding=1),
                    torch.nn.ReLU(inplace=True)) 
            self.decoders.append(dec)
        
    def forward(self,x):
        for enc in self.encoders:
            x = enc(x)
        sz = x.size()
        x = x.view(-1,self.lin_features)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = x.view(sz)
        for dec in self.decoders:
            x = dec(x)
        return x

    def encode(self,x):
        for enc in self.encoders:
            x = enc(x)
        sz = x.size()
        x = x.view(-1,self.lin_features)
        return self.linear_enc(x),sz

    def decode(self,x,sz):
        x = self.linear_dec(x)
        x = x.view(sz)
        for dec in self.decoders:
            x = dec(x)
        return x
        
class EncPred(torch.nn.Module):
    def __init__(self,sizes,data_chan,kernel_size,filters,depth,pool,out_features,batch_norm=False,pool_type='conv',pad_type='SAME'):
        super(EncPred,self).__init__()
        self.sizes = sizes
        dim = len(self.sizes)

        if pad_type == 'VALID':
            pad_width = 0
        elif pad_type == 'SAME':
            pad_width = kernel_size//2
        else:
            raise ValueError("ERROR: pad_type must be either VALID or SAME")

        if kernel_size%2==0:
            raise ValueError("ERROR: Kernel size must be odd")

        if dim == 2:
            self.conv = torch.nn.Conv2d
            self.conv_tran = torch.nn.ConvTranspose2d
        elif dim == 3:
            self.conv = torch.nn.Conv3d
            self.conv_tran = torch.nn.ConvTranspose3d

        self.encoders = torch.nn.ModuleList([])
        for i in range(depth):
            in_filters = data_chan if i==0 else filters
            enc = torch.nn.Sequential(
                self.conv(in_channels=in_filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width,stride=pool),
                torch.nn.ReLU(inplace=True))
            self.encoders.append(enc)

        if dim == 2:
            self.lin_features = int(filters*sizes[0]*sizes[1]/(pool**(2*depth)))
        elif dim == 3:
            self.lin_features = int(filters*sizes[0]*sizes[1]*sizes[2]/(pool**(3*depth)))
 
        self.linear_enc = torch.nn.Linear(in_features=self.lin_features,out_features=out_features)   
        
    def forward(self,x):
        for enc in self.encoders:
            x = enc(x)
        x = x.view(-1,self.lin_features)
        return self.linear_enc(x)

