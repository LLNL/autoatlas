import torch

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

class UNet3D(torch.nn.Module):
    def __init__(self,num_labels,kernel_size=3,filters=32,blocks=7,batch_norm=False,pad_type='SAME'):
        super(UNet3D,self).__init__()
        self.kernel_size = kernel_size
        self.blocks = blocks

        if self.blocks % 2 == 0:
            raise ValueError('Number of UNet blocks must be odd')

        self.pad_type = pad_type
        if pad_type == 'VALID':
            pad_width = 0
        elif pad_type == 'SAME':
            pad_width = 1
        else:
            raise ValueError("ERROR: pad_type must be either VALID or SAME")

        self.encoders = torch.nn.ModuleList([UNetEncoder3D(in_filters=1,out_filters=filters*2,kernel_size=kernel_size,pad_width=pad_width,batch_norm=batch_norm)])
        for _ in range(blocks//2-1):
            filters *= 2
            enc = UNetEncoder3D(in_filters=filters,out_filters=2*filters,kernel_size=kernel_size,pad_width=pad_width,batch_norm=batch_norm) 
            self.encoders.append(enc)
  
        filters *= 2
        self.decoders = torch.nn.ModuleList([UNetDecoder3D(in_filters=filters,out_filters=2*filters,kernel_size=kernel_size,pad_width=pad_width,batch_norm=batch_norm)])
        for _ in range(blocks//2-1):
            dec = UNetDecoder3D(in_filters=filters+filters*2,out_filters=filters,kernel_size=kernel_size,pad_width=pad_width,batch_norm=batch_norm)
            self.decoders.append(dec)
            filters = filters//2
            #Division using // ensures return value is integer

        self.output = torch.nn.Sequential(
                        torch.nn.Conv3d(in_channels=filters+2*filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv3d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv3d(in_channels=filters,out_channels=num_labels,kernel_size=1,padding=0))
                        #torch.nn.Softmax(dim=1))

    def forward(self,x):
        features = []
        for enc in self.encoders:
            x,z = enc(x)
            features.append(z)

        x = self.decoders[0](x)
        for i,dec in enumerate(self.decoders[1:]):
            z = features.pop()
            if self.pad_type == 'VALID':
                cr = [(sz-sx)//2 for sz,sx in zip(z.size(),x.size())]
                #sz-sx is always even since up/down sample factor is 2
                z = z[cr[0]:-cr[0],cr[1]:-cr[1],cr[2],-cr[2]]
            x = torch.cat((x,z),dim=1)
            x = dec(x)

        z = features.pop()
        if self.pad_type == 'VALID':
            cr = [(sz-sx)//2 for sz,sx in zip(z.size(),x.size())]
            z = z[cr[0]:-cr[0],cr[1]:-cr[1],cr[2],-cr[2]]
        x = torch.cat((x,z),dim=1)
        return self.output(x)

class UNetEncoder3D(torch.nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,pad_width,batch_norm=False):
        super(UNetEncoder3D,self).__init__()
        
        self.model = torch.nn.Sequential(
                        torch.nn.Conv3d(in_channels=in_filters,out_channels=out_filters//2,kernel_size=kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv3d(in_channels=out_filters//2,out_channels=out_filters,kernel_size=kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True))
        self.pool = torch.nn.MaxPool3d(2,padding=0)

    def forward(self,x):
        x = self.model(x)
        return self.pool(x),x

class UNetDecoder3D(torch.nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,pad_width,batch_norm=False):
        super(UNetDecoder3D,self).__init__()

        self.model = torch.nn.Sequential(
                            torch.nn.Conv3d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,padding=pad_width),
                            torch.nn.ReLU(inplace=True), 
                            torch.nn.Conv3d(in_channels=out_filters,out_channels=out_filters,kernel_size=kernel_size,padding=pad_width),
                            torch.nn.ReLU(inplace=True), 
                            torch.nn.ConvTranspose3d(in_channels=out_filters,out_channels=out_filters,kernel_size=2,padding=0,stride=2))

    def forward(self,x):
        return self.model(x)

class AutoEnc(torch.nn.Module):
    def __init__(self,kernel_size=3,filters=8,depth=5,pool=2,batch_norm=False,pool_type='conv',pad_type='SAME'):
        super(AutoEnc,self).__init__()
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

        self.encoders,self.dwnpools = torch.nn.ModuleList([]),torch.nn.ModuleList([])
        for i in range(depth//2):
            in_filters = 2 if i==0 else filters
            enc = torch.nn.Sequential(
                        torch.nn.Conv3d(in_channels=in_filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True))
            self.encoders.append(enc)
            #self.dwnpools.append(torch.nn.MaxPool3d(kernel_size=pool,stride=pool,return_indices=True))
            self.dwnpools.append(torch.nn.Conv3d(in_channels=filters,out_channels=filters,kernel_size=pool,stride=pool))
        self.decoders,self.uppools = torch.nn.ModuleList([]),torch.nn.ModuleList([])
        for i in range(depth//2):
            dec = torch.nn.Sequential(
                            torch.nn.Conv3d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,padding=pad_width),
                            torch.nn.ReLU(inplace=True)) 
                            #torch.nn.ConvTranspose3d(filters,out_filters,pool,padding=0,stride=pool))
            self.decoders.append(dec)
            #self.uppools.append(torch.nn.MaxUnpool3d(kernel_size=pool,stride=pool))
            self.uppools.append(torch.nn.ConvTranspose3d(in_channels=filters,out_channels=filters,kernel_size=pool,stride=pool))
        
        self.final = torch.nn.Conv3d(in_channels=filters,out_channels=1,kernel_size=1,padding=0)

    def forward(self,x):
        #indices,shapes = [],[]
        for enc,pl in zip(self.encoders,self.dwnpools):
            x = enc(x)
            #shapes.append(x.size())
            #x,idx = pl(x)
            #indices.append(idx)
            x = pl(x)

        for dec,pl in zip(self.decoders,self.uppools):
            x = dec(x)
            #x = pl(x,indices=indices.pop(),output_size=shapes.pop())
            x = pl(x)

        return self.final(x)

