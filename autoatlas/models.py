import torch

class SegmRecon(torch.nn.Module):
    def __init__(self,segm,autoencs,aenc_devs):
        super(SegmRecon,self).__init__()
        self.segm = segm
        self.autoencs = autoencs
        self.devs = aenc_devs 
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self,x):
        seg = self.segm(x)
        seg = self.softmax(seg)
        recons,codes = [],[]
        for i,auto in enumerate(self.autoencs):
            z = (x.to(self.devs[i]))*(seg[:,i:i+1].to(self.devs[i]))
            #i:i+1 ensures singleton dimensions are retained
            z,sz = auto.encode(z)
            codes.append(z)
            recons.append(auto.decode(z,sz))
        codes = torch.stack(codes,dim=1)
        return seg,recons,codes

