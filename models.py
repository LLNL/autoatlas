import torch

class SegmRecon(torch.nn.Module):
    def __init__(self,cnn,eps=0.0):
        super(SegmRecon,self).__init__()
        self.cnn = cnn
        self.eps = eps   
 
    def forward(self,x):
        y = self.cnn(x)
        emb = torch.sum(x*y,dim=(2,3,4),keepdim=True)
        emb = emb/(torch.sum(y,dim=(2,3,4),keepdim=True)+self.eps)
        return torch.sum(y*emb,dim=1)
          
