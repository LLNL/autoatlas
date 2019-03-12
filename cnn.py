import torch

class UNet3D(torch.nn.Module):
    def __init__(self,num_labels,kernel_size=3,filters=32,depth=4,batch_norm=False,pad_type='SAME'):
        super(UNet3D,self).__init__()
        self.kernel_size = kernel_size
        self.depth = depth
        self.pad_type = pad_type
        if pad_type == 'VALID':
            pad_width = 0
        elif pad_type == 'SAME':
            pad_width = 1
        else:
            raise ValueError("ERROR: pad_type must be either VALID or SAME")

        self.encoders = torch.nn.ModuleList([Encoder3D(1,filters*2,kernel_size,pad_width,batch_norm)])
        for _ in range(depth-2):
            filters *= 2
            enc = Encoder3D(filters,2*filters,kernel_size,pad_width,batch_norm) 
            self.encoders.append(enc)
  
        filters *= 2
        self.decoders = torch.nn.ModuleList([Decoder3D(filters,2*filters,kernel_size,pad_width,batch_norm)])
        for _ in range(depth-2):
            dec = Decoder3D(filters+filters*2,filters,kernel_size,pad_width,batch_norm)
            self.decoders.append(dec)
            filters = filters//2
            #Division using // ensures return value is integer

        self.output = torch.nn.Sequential(
                        torch.nn.Conv3d(filters+2*filters,filters,kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv3d(filters,filters,kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv3d(filters,num_labels,1,padding=0),
                        torch.nn.Softmax(dim=1))

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

class Encoder3D(torch.nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,pad_width,batch_norm=False):
        super(Encoder3D,self).__init__()
        
        self.model = torch.nn.Sequential(
                        torch.nn.Conv3d(in_filters,out_filters//2,kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv3d(out_filters//2,out_filters,kernel_size,padding=pad_width),
                        torch.nn.ReLU(inplace=True))
        self.pool = torch.nn.MaxPool3d(2,padding=0)

    def forward(self,x):
        x = self.model(x)
        return self.pool(x),x

class Decoder3D(torch.nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,pad_width,batch_norm=False):
        super(Decoder3D,self).__init__()

        self.model = torch.nn.Sequential(
                            torch.nn.Conv3d(in_filters,out_filters,kernel_size,padding=pad_width),
                            torch.nn.ReLU(inplace=True), 
                            torch.nn.Conv3d(out_filters,out_filters,kernel_size,padding=pad_width),
                            torch.nn.ReLU(inplace=True), 
                            torch.nn.ConvTranspose3d(out_filters,out_filters,2,padding=0,stride=2))

    def forward(self,x):
        return self.model(x)
