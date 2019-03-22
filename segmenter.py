import torch
import os
from cnn import UNet3D
from models import SegmRecon
from torch.utils.data import DataLoader
import progressbar
import numpy as np

class AutoSegmenter:
    def __init__(self,num_labels,batch=16,lr=1e-3,eps=0,device='cpu',checkpoint_dir='./checkpoints/',load_checkpoint_epoch=None):
        self.lr = lr
        self.batch = batch
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.cnn = UNet3D(num_labels,kernel_size=3,filters=32,depth=4,batch_norm=False,pad_type='SAME')  
        self.model = SegmRecon(self.cnn,eps=eps)
        self.model = self.model.to(self.device)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr) 

        if load_checkpoint_epoch is not None:
            checkpoint_path = self.get_checkpoint_path(load_checkpoint_epoch)
            try:
                checkpoint = torch.load(checkpoint_path)
            except FileNotFoundError:
                raise ValueError('Checkpoint path does not exist: {}'.format(checkpoint_path),flush=True)
            else:
                self.model.load_state_dict(checkpoint['model'])
                self.start_epoch = checkpoint['epoch']
                self.curr_train_loss = checkpoint['train_loss']
                self.curr_val_loss = checkpoint['val_loss']
                print('Loaded model from epoch: {}'.format(self.start_epoch),flush=True)
                print('==> Model stats: train_loss={:.3e}, val_loss={:.3e}'.format(
                    self.curr_train_loss,self.curr_val_loss),flush=True)
        else:
            self.start_epoch = 0
            self.curr_train_loss = float('inf')
            self.curr_val_loss = float('inf')

        self.widgets = [' [', progressbar.Percentage(), '] ',progressbar.Bar(),
            ' [', progressbar.DynamicMessage('batch_loss'), '] ',
            ' [', progressbar.DynamicMessage('avg_loss'), '] ',
            ' [', progressbar.Timer(), '] ',
            ' [', progressbar.ETA(), '] ']
    
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

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

        self.curr_train_loss = avg_loss
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

        self.curr_val_loss = avg_loss
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

    def get_checkpoint_path(self, epoch):
        return os.path.join(self.checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))

    def checkpoint(self, epoch):
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'train_loss': self.curr_train_loss,
            'val_loss': self.curr_val_loss,
        }
        checkpoint_path = self.get_checkpoint_path(epoch)
        if os.path.exists(checkpoint_path):
            print('WARNING: Overwriting existing checkpoint path: {}'.format(checkpoint_path), flush=True)
        torch.save(state_dict, checkpoint_path)
