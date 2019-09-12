import torch
import os
from cnn import UNet3D,AutoEnc
from models import SegmRecon
from torch.utils.data import DataLoader
import progressbar
import numpy as np

class CustomLoss:
    def __init__(self,smooth_reg=0.0,unif_reg=0.0,entr_reg=0.0,entr_norm=1.0):
        self.smooth_reg = smooth_reg
        self.unif_reg = unif_reg
        self.entr_reg = entr_reg
        self.entr_norm = entr_norm
        self.neighs = [[0,0,1],  [0,1,-1],[0,1,0], [0,1,1],
                       [1,-1,-1],[1,-1,0],[1,-1,1],
                       [1,0,-1], [1,0,0], [1,0,1],
                       [1,1,-1], [1,1,0], [1,1,1]]
        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        
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
    
    def entr_loss(self,Z,normZ): #Must reduce entropy at each voxel. Force each voxel to be a single class.
        return -self.entr_reg*torch.mean(normZ*self.logsoftmax(Z))/self.entr_norm

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
       
        self.criterion = CustomLoss(smooth_reg,unif_reg,entr_reg=entr_reg,entr_norm=np.log(num_labels))
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
                self.train_tot_loss = checkpoint['train_tot_loss']
                self.train_mse_loss = checkpoint['train_mse_loss']
                self.train_entr_loss = checkpoint['train_entr_loss']
                self.test_tot_loss = checkpoint['test_tot_loss']
                self.test_mse_loss = checkpoint['test_mse_loss']
                self.test_entr_loss = checkpoint['test_entr_loss']
                print('Loaded model from epoch: {}'.format(self.start_epoch),flush=True)
                print('Model stats: train loss={:.3e}, test loss={:.3e}'.format(self.train_tot_loss,self.test_tot_loss),flush=True)
        else:
            self.start_epoch = 0
            self.train_tot_loss = float('inf')
            self.train_mse_loss = float('inf')
            self.train_entr_loss = float('inf')
            self.test_tot_loss = float('inf')
            self.test_mse_loss = float('inf')
            self.test_entr_loss = float('inf')

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self,dataset):
        train_loader = DataLoader(dataset,batch_size=self.batch,shuffle=True)

        self.model.train()
       
        num_batch = 0 
        avg_tot_loss,avg_mse_loss,avg_entr_loss = 0.0,0.0,0.0
        for data_in in train_loader:
            data_in = data_in.to(self.device)
            self.optimizer.zero_grad()
            seg,norm_seg,data_out = self.model(data_in)
            
            mse_loss = self.criterion.mse_loss(data_in,data_out)
            entr_loss = self.criterion.entr_loss(seg,norm_seg)
            tot_loss = mse_loss+entr_loss

            tot_loss.backward()
            self.optimizer.step()

            batch_mse_loss = mse_loss.item()
            batch_entr_loss = entr_loss.item()
            batch_tot_loss = tot_loss.item()

            avg_mse_loss += batch_mse_loss
            avg_entr_loss += batch_entr_loss
            avg_tot_loss += batch_tot_loss
            num_batch += 1
            print("TRAIN: batch losses: tot {:.2e}, mse {:.2e}, entr {:.2e}".format(batch_tot_loss,batch_mse_loss,batch_entr_loss))

        avg_tot_loss /= num_batch
        avg_mse_loss /= num_batch
        avg_entr_loss /= num_batch
        print("TRAIN: average losses: tot {:.2e}, mse {:.2e}, entr {:.2e}".format(avg_tot_loss,avg_mse_loss,avg_entr_loss))

        self.train_tot_loss = avg_tot_loss
        self.train_mse_loss = avg_mse_loss
        self.train_entr_loss = avg_entr_loss
        return avg_tot_loss

    def test(self,dataset):
        test_loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        
        self.model.eval()
        
        num_batch = 0 
        avg_tot_loss,avg_mse_loss,avg_entr_loss = 0.0,0.0,0.0
        with torch.no_grad():
            for data_in in test_loader:
                data_in = data_in.to(self.device)
                seg,norm_seg,data_out = self.model(data_in)

                mse_loss = self.criterion.mse_loss(data_in,data_out)
                entr_loss = self.criterion.entr_loss(seg,norm_seg)
                tot_loss = mse_loss+entr_loss
                
                batch_mse_loss = mse_loss.item()
                batch_entr_loss = entr_loss.item()
                batch_tot_loss = tot_loss.item()

                avg_mse_loss += batch_mse_loss
                avg_entr_loss += batch_entr_loss
                avg_tot_loss += batch_tot_loss
                num_batch += 1
                print("TEST: batch losses: tot {:.2e}, mse {:.2e}, entr {:.2e}".format(batch_tot_loss,batch_mse_loss,batch_entr_loss))
                    
        avg_tot_loss /= num_batch
        avg_mse_loss /= num_batch
        avg_entr_loss /= num_batch
        print("TEST: average losses: tot {:.2e}, mse {:.2e}, entr {:.2e}".format(avg_tot_loss,avg_mse_loss,avg_entr_loss))
        
        self.test_tot_loss = avg_tot_loss
        self.test_mse_loss = avg_mse_loss
        self.test_entr_loss = avg_entr_loss
        return avg_tot_loss

    def segment(self,dataset):
        loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        self.model.eval()
        
        inp,segm,rec = [],[],[]
        with torch.no_grad():
            for idx,data_in in enumerate(loader):
                inp.append(data_in.numpy())
                data_in = data_in.to(self.device)
                _,s,r = self.model(data_in)
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
                _,segm,_ = self.model(data_in)
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
            'train_tot_loss': self.train_tot_loss,
            'train_mse_loss': self.train_mse_loss,
            'train_entr_loss': self.train_entr_loss,
            'test_tot_loss': self.test_tot_loss,
            'test_mse_loss': self.test_mse_loss,
            'test_entr_loss': self.test_entr_loss,
        }
        checkpoint_path = self.get_checkpoint_path(epoch)
        if os.path.exists(checkpoint_path):
            print('WARNING: Overwriting existing checkpoint path: {}'.format(checkpoint_path), flush=True)
        torch.save(state_dict, checkpoint_path)
