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

        #bg = torch.FloatTensor([1]+[0]*(y.size(1)-1)).to(y.device)
        #bg = bg.view(1,y.size(1),1,1,1) if y.dim()==5 else bg.view(1,y.size(1),1,1)
#        y = torch.where(mask,y,bg)
#        emb = torch.sum(x*y,dim=(2,3,4),keepdim=True)
#        emb = emb/(torch.sum(y,dim=(2,3,4),keepdim=True)+self.eps)
#        return y,torch.sum(y*emb,dim=1)

        recons = []
        for i,auto in enumerate(self.autoencs):
            z = [x*y[:,i:i+1],x*(1-y[:,i:i+1])]
            z = torch.cat(z,dim=1)
            recons.append(auto(z)) #i:i+1 ensures singleton dimensions are retained
        #return seg,y,torch.sum(torch.stack(recons,dim=-1),dim=-1)
        #recons = torch.cat(recons,dim=1)
        #assert recons.size(1)==y.size(1)
        return seg,y,recons

    def segcode(self,x):
        seg = self.segm(x)
        y = self.softmax(seg)
        rec_encs = []
        for i,auto in enumerate(self.autoencs):
            z = [x*y[:,i:i+1],x*(1-y[:,i:i+1])]
            z = torch.cat(z,dim=1)
            rec_encs.append(auto.encode(z))
        return y,torch.stack(rec_encs,dim=1)
         
          
