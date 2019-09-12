import torch

class SegmRecon(torch.nn.Module):
    def __init__(self,segm,autoencs):
        super(SegmRecon,self).__init__()
        self.segm = segm
        self.autoencs = autoencs
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self,x):
        seg = self.segm(x)
        y = self.softmax(seg)
#        emb = torch.sum(x*y,dim=(2,3,4),keepdim=True)
#        emb = emb/(torch.sum(y,dim=(2,3,4),keepdim=True)+self.eps)
#        return y,torch.sum(y*emb,dim=1)
        recons = []
        for i,auto in enumerate(self.autoencs):
            z = [x*y[:,i:i+1],x*(1-y[:,i:i+1])]
            z = torch.cat(z,dim=1)
            recons.append(y[:,i:i+1]*auto(z)) #i:i+1 ensures singleton dimensions are retained
        return seg,y,torch.sum(torch.stack(recons,dim=-1),dim=-1)

        
          
