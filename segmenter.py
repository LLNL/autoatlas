import torch
from cnn import UNet3D
from models import SegmRecon
from torch.utils.data import DataLoader
import progressbar
import numpy as np

class AutoSegmenter:
    def __init__(self,num_labels,batch=16,lr=1e-3,eps=0,device='cpu'):
        self.lr = lr
        self.batch = batch
        self.device = device

        self.cnn = UNet3D(num_labels,kernel_size=3,filters=32,depth=4,batch_norm=False,pad_type='SAME')  
        self.model = SegmRecon(self.cnn,eps=eps)
        self.model = self.model.to(self.device)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr) 

        self.widgets = [' [', progressbar.Percentage(), '] ',progressbar.Bar(),
            ' [', progressbar.DynamicMessage('batch_loss'), '] ',
            ' [', progressbar.DynamicMessage('avg_loss'), '] ',
            ' [', progressbar.Timer(), '] ',
            ' [', progressbar.ETA(), '] ']

    def train(self,dataset):
        train_loader = DataLoader(dataset,batch_size=self.batch,shuffle=True)

        self.model.train()
        
        total_loss = 0.0
        widgets = ['Train:']+self.widgets
        with progressbar.ProgressBar(max_value=len(train_loader),widgets=widgets) as bar:
            for idx,data_in in enumerate(train_loader):
                data_in = data_in.to(self.device)
                self.optimizer.zero_grad()
                data_out = self.model(data_in)
                
                loss = self.criterion(data_in,data_out)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss/(idx+1)

                bar.update(idx,avg_loss=avg_loss,batch_loss=loss.item())

        return avg_loss

    def test(self,dataset):
        test_loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        
        self.model.eval()
        
        total_loss = 0.0
        widgets = ['Test:']+self.widgets
        with torch.no_grad():
            with progressbar.ProgressBar(max_value=len(test_loader),widgets=widgets) as bar:
                for idx,data_in in enumerate(test_loader):
                    data_in = data_in.to(self.device)
                    data_out = self.model(data_in)

                    loss = self.criterion(data_in,data_out)
                    total_loss += loss.item()
                    avg_loss = total_loss/(idx+1)

                    bar.update(idx,avg_loss=avg_loss,batch_loss=loss.item())

        return avg_loss

    def segment(self,dataset):
        loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        self.model.eval()
        
        widgets = ['Segment:']+self.widgets
        inp,segm = [],[] 
        with torch.no_grad():
            with progressbar.ProgressBar(max_value=len(loader),widgets=widgets) as bar:
                for idx,data_in in enumerate(loader):
                    inp.append(data_in.numpy())
                    data_in = data_in.to(self.device)
                    data_out = self.cnn(data_in).cpu().numpy()
                    segm.append(data_out)
        return np.concatenate(segm,axis=0),np.concatenate(inp,axis=0)

