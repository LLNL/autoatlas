import torch
import os
from cnn import UNet3D,AutoEnc
from models import SegmRecon
from torch.utils.data import DataLoader
import progressbar
import numpy as np

class CustomLoss:
    def __init__(self,smooth_reg=0.0,unif_reg=0.0,entr_reg=0.0):
        self.smooth_reg = smooth_reg
        self.unif_reg = unif_reg
        self.neighs = [[0,0,1],  [0,1,-1],[0,1,0], [0,1,1],
                       [1,-1,-1],[1,-1,0],[1,-1,1],
                       [1,0,-1], [1,0,0], [1,0,1],
                       [1,1,-1], [1,1,0], [1,1,1]]

    def mse_loss(self,X,Y):
        mse_loss = torch.mean((X-Y)*(X-Y))
        return mse_loss

#    def smooth_loss(self,Z):
#        smooth_losses = []
#        for (Nz,Ny,Nx) in self.neighs:
#            H = torch.cat((Z[:,:,-Nz:],    Z[:,:,:-Nz]),    dim=2)
#            H = torch.cat((H[:,:,:,-Ny:],  H[:,:,:,:-Ny]),  dim=3)
#            H = torch.cat((H[:,:,:,:,-Nx:],H[:,:,:,:,:-Nx]),dim=4)
#            H = Z-H
#            smooth_losses.append(torch.mean(H*H))   
#        return self.smooth_reg*torch.mean(torch.stack(smooth_losses))

#    def unif_loss(self,Z): #To enforce an equal number of voxels for each class/label
#        return -self.unif_reg*torch.mean(torch.log(torch.mean(Z,dim=(2,3,4))))
    
#    def entr_loss(self,Z): #Must reduce entropy at each voxel. Force each voxel to be a single class.
#        return -self.entr_reg*torch.mean(Z*torch.log(Z))

    def loss(self,X,Z,Y):
        #return self.mse_loss(X,Y)+self.smooth_loss(Z)+self.unif_loss(Z)+self.entr_loss(Z)
        return self.mse_loss(X,Y)

class AutoSegmenter:
    def __init__(self,num_labels,smooth_reg=0.0,unif_reg=0.0,entr_reg=0.0,batch=16,lr=1e-3,device='cpu',checkpoint_dir='./checkpoints/',load_checkpoint_epoch=None):
        self.lr = lr
        self.batch = batch
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.cnn = UNet3D(num_labels,kernel_size=3,filters=32,depth=4,batch_norm=False,pad_type='SAME')
   
        self.autoencs = torch.nn.ModuleList([])
        for _ in range(num_labels):
            self.autoencs.append(AutoEnc(kernel_size=5,filters=8,depth=4,pool=4,batch_norm=False,pad_type='SAME')) 
        self.model = SegmRecon(self.cnn,self.autoencs)
        self.model = self.model.to(self.device)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
            #torch.backends.cudnn.benchmark = False #True results in cuDNN error: CUDNN_STATUS_INTERNAL_ERROR on pascal
       
        self.criterion = CustomLoss(smooth_reg,unif_reg,entr_reg)
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
                self.curr_test_loss = checkpoint['val_loss']
                print('Loaded model from epoch: {}'.format(self.start_epoch),flush=True)
                print('==> Model stats: train_loss={:.3e}, test_loss={:.3e}'.format(
                    self.curr_train_loss,self.curr_test_loss),flush=True)
        else:
            self.start_epoch = 0
            self.curr_train_loss = float('inf')
            self.curr_test_loss = float('inf')

#        self.widgets = [' [', progressbar.Percentage(), '] ',progressbar.Bar(),
#            ' [', progressbar.DynamicMessage('avg_loss'), '] ',
#            ' [', progressbar.DynamicMessage('batch_loss'), '] ',
#            ' [', progressbar.DynamicMessage('batch_mse'), '] ',
#            ' [', progressbar.DynamicMessage('batch_smooth'), '] ',
#            ' [', progressbar.DynamicMessage('batch_unif'), '] ',
#            ' [', progressbar.DynamicMessage('batch_entr'), '] ',
#            ' [', progressbar.Timer(), '] ',
#            ' [', progressbar.ETA(), '] ']
    
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self,dataset):
        train_loader = DataLoader(dataset,batch_size=self.batch,shuffle=True)

        self.model.train()
        
        total_loss = 0.0
#        widgets = ['Train:']+self.widgets
#        with progressbar.ProgressBar(max_value=len(train_loader),widgets=widgets) as bar:
        for idx,data_in in enumerate(train_loader):
            data_in = data_in.to(self.device)
            self.optimizer.zero_grad()
            seg,data_out = self.model(data_in)
            
            mse_loss = self.criterion.mse_loss(data_in,data_out)
            #smooth_loss = self.criterion.smooth_loss(seg)
            #unif_loss = self.criterion.unif_loss(seg)
            #entr_loss = self.criterion.entr_loss(seg)
            loss = mse_loss#+smooth_loss+unif_loss+entr_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss/(idx+1)

            #print("TRAIN: avg loss {:.2e}, batch loss {:.2e}, batch mse {:.2e}, batch smooth {:.2e}, batch unif {:.2e}, batch entr {:.2e}".format(avg_loss,loss.item(),mse_loss.item(),smooth_loss.item(),unif_loss.item(),entr_loss.item()))
            print("TRAIN: avg loss {:.2e}, batch loss {:.2e}, batch mse {:.2e}".format(avg_loss,loss.item(),mse_loss.item()))
#                bar.update(idx,avg_loss=avg_loss,batch_loss=loss.item(),batch_mse=mse_loss.item(),batch_smooth=smooth_loss.item(),batch_unif=unif_loss.item(),batch_entr=entr_loss.item())

        self.curr_train_loss = avg_loss
        return avg_loss

    def test(self,dataset):
        test_loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        
        self.model.eval()
        
        total_loss = 0.0
#        widgets = ['Test:']+self.widgets
        with torch.no_grad():
#            with progressbar.ProgressBar(max_value=len(test_loader),widgets=widgets) as bar:
            for idx,data_in in enumerate(test_loader):
                data_in = data_in.to(self.device)
                seg,data_out = self.model(data_in)

                mse_loss = self.criterion.mse_loss(data_in,data_out)
#                smooth_loss = self.criterion.smooth_loss(seg)
#                unif_loss = self.criterion.unif_loss(seg)
#                entr_loss = self.criterion.entr_loss(seg)
                loss = mse_loss#+smooth_loss+unif_loss+entr_loss
                
                total_loss += loss.item()
                avg_loss = total_loss/(idx+1)

                #print("TEST: avg loss {:.2e}, batch loss {:.2e}, batch mse {:.2e}, batch smooth {:.2e}, batch unif {:.2e}, batch entr {:.2e}".format(avg_loss,loss.item(),mse_loss.item(),smooth_loss.item(),unif_loss.item(),entr_loss.item()))
                print("TEST: avg loss {:.2e}, batch loss {:.2e}, batch mse {:.2e}".format(avg_loss,loss.item(),mse_loss.item()))
                    
#                    bar.update(idx,avg_loss=avg_loss,batch_loss=loss.item(),batch_mse=mse_loss.item(),batch_smooth=smooth_loss.item(),batch_unif=unif_loss.item(),batch_entr=entr_loss.item())

        self.curr_test_loss = avg_loss
        return avg_loss

    def segment(self,dataset):
        loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        self.model.eval()
        
        #widgets = ['Segment:']+self.widgets
        inp,segm,rec = [],[],[]
        with torch.no_grad():
            #with progressbar.ProgressBar(max_value=len(loader),widgets=widgets) as bar:
            for idx,data_in in enumerate(loader):
                inp.append(data_in.numpy())
                data_in = data_in.to(self.device)
                s,r = self.model(data_in)
                segm.append(s.cpu().numpy())
                rec.append(r.cpu().numpy())
        return np.concatenate(segm,axis=0),np.concatenate(rec,axis=0),np.concatenate(inp,axis=0)

    def classrec(self,dataset):
        loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        self.model.eval()
        inp,recs = [],[]
        with torch.no_grad():
            for idx,data_in in enumerate(loader):
                inp.append(data_in.numpy())
                data_in = data_in.to(self.device)
                clrecs = []
                segm = self.cnn(data_in)
                for i,aenc in enumerate(self.autoencs):
                    z = [data_in*segm[:,i:i+1],data_in*(1-segm[:,i:i+1])]
                    z = torch.cat(z,dim=1)
                    r = aenc(z)
                    clrecs.append(r.cpu().numpy())
                recs.append(np.concatenate(clrecs,axis=1))
        return np.concatenate(recs,axis=0),np.concatenate(inp,axis=0)

    def get_checkpoint_path(self, epoch):
        return os.path.join(self.checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))

    def checkpoint(self, epoch):
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'train_loss': self.curr_train_loss,
            'test_loss': self.curr_test_loss,
        }
        checkpoint_path = self.get_checkpoint_path(epoch)
        if os.path.exists(checkpoint_path):
            print('WARNING: Overwriting existing checkpoint path: {}'.format(checkpoint_path), flush=True)
        torch.save(state_dict, checkpoint_path)
