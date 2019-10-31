import torch
import os
from cnn import UNet,AutoEnc
from models import SegmRecon
from torch.utils.data import DataLoader
import progressbar
import numpy as np
import multiprocessing as mp

class CustomLoss:
    def __init__(self,dim=3,smooth_reg=0.0,devr_reg=0.0,entr_reg=0.0,entr_norm=1.0,min_freqs=0.01,npow=1):
        self.smooth_reg = smooth_reg
        self.devr_reg = devr_reg
        self.entr_reg = entr_reg
        self.entr_norm = entr_norm
        self.min_freqs = min_freqs
        self.npow = npow
        self.dim = dim
        self.dimlist = [2+i for i in range(self.dim)]
        if dim==3:
            self.neighs = [[0,0,1],  [0,1,-1],[0,1,0], [0,1,1],
                       [1,-1,-1],[1,-1,0],[1,-1,1],
                       [1,0,-1], [1,0,0], [1,0,1],
                       [1,1,-1], [1,1,0], [1,1,1]]
        elif dim==2:
            self.neighs = [[0,1,np.nan],[1,-1,np.nan],[1,0,np.nan],[1,1,np.nan]]
        else:
            raise ValueError('dim must be either 2 or 3')
        self.softmax = torch.nn.Softmax(dim=1)
        #self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.eps = 1e-10
        
    def mse_loss(self,gtruth,recs,seg,mask):
        mse_losses = []
        den = torch.sum(mask,dim=self.dimlist)
        for i,r in enumerate(recs):
            num = torch.mean(torch.abs(gtruth-r)**self.npow,dim=1,keepdim=True)
            num = torch.sum(num*seg[:,i:i+1]*mask,dim=self.dimlist)
            mse_losses.append(num/den)
        return torch.mean(torch.stack(mse_losses))

    def smooth_loss(self,seg,mask):
        nums,dens = [],[]
        for (Nz,Ny,Nx) in self.neighs:
            H = torch.cat((seg[:,:,-Nz:],  seg[:,:,:-Nz]),  dim=2)
            H = torch.cat((H[:,:,:,-Ny:],  H[:,:,:,:-Ny]),  dim=3)
            if self.dim==3:
                H = torch.cat((H[:,:,:,:,-Nx:],H[:,:,:,:,:-Nx]),dim=4)
            H = seg*H
            W = torch.cat((mask[:,:,-Nz:],  mask[:,:,:-Nz]),  dim=2)
            W = torch.cat((W[:,:,:,-Ny:],  W[:,:,:,:-Ny]),  dim=3)
            if self.dim==3:
                W = torch.cat((W[:,:,:,:,-Nx:],W[:,:,:,:,:-Nx]),dim=4)
            W = mask*W
            assert torch.max(W)==1.0
            assert torch.min(W)==0.0
            nums.append(torch.sum(H*W,dim=self.dimlist))
            dens.append(torch.sum(W,dim=self.dimlist))
        smooth_loss = torch.sum(torch.stack(nums,dim=-1),dim=-1)/torch.sum(torch.stack(dens,dim=-1),dim=-1)   
        smooth_loss = -torch.log(torch.sum(smooth_loss,dim=1))
        return self.smooth_reg*torch.mean(smooth_loss)

    def devr_loss(self,seg,mask):
        clp = torch.sum(seg*mask,dim=self.dimlist)/torch.sum(mask,dim=self.dimlist)
        clp = -torch.log((clp+self.eps*self.min_freqs)/self.min_freqs)
        clp = torch.clamp(clp,min=0)
        return self.devr_reg*torch.mean(clp)
    
    #def entr_loss(self,Z,normZ): #Must reduce entropy at each voxel. Force each voxel to be a single class.
    #    return -self.entr_reg*torch.mean(normZ*self.logsoftmax(Z))/self.entr_norm

class AutoSegmenter:
    def __init__(self,num_labels,sizes,data_chan=1,smooth_reg=0.0,devr_reg=0.0,entr_reg=0.0,min_freqs=0.01,batch=16,lr=1e-3,unet_chan=32,unet_blocks=9,aenc_chan=16,aenc_depth=8,re_pow=1,device='cpu',checkpoint_dir='./checkpoints/',load_checkpoint_epoch=None):
        dim = len(sizes)
        self.lr = lr
        self.batch = batch
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.data_chan = data_chan
        self.smooth_reg = smooth_reg
        self.devr_reg = devr_reg
        self.entr_reg = entr_reg

        self.cnn = UNet(num_labels,dim=dim,data_chan=data_chan,kernel_size=3,filters=unet_chan,blocks=unet_blocks,batch_norm=False,pad_type='SAME')
   
        self.autoencs = torch.nn.ModuleList([])
        for _ in range(num_labels):
            self.autoencs.append(AutoEnc(sizes,data_chan=data_chan,kernel_size=7,filters=aenc_chan,depth=aenc_depth,pool=2,batch_norm=False,pad_type='SAME')) 
            #self.autoencs.append(AutoEnc(kernel_size=7,filters=8,depth=4,pool=4,batch_norm=False,pad_type='SAME')) 
        self.model = SegmRecon(self.cnn,self.autoencs)
        self.model = self.model.to(self.device)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
            #torch.backends.cudnn.benchmark = False #True results in cuDNN error: CUDNN_STATUS_INTERNAL_ERROR on pascal
       
        self.criterion = CustomLoss(dim=dim,smooth_reg=smooth_reg,devr_reg=devr_reg,entr_reg=entr_reg,entr_norm=np.log(num_labels),min_freqs=min_freqs,npow=re_pow)
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
                self.train_smooth_loss = checkpoint['train_smooth_loss']
                self.train_entr_loss = checkpoint['train_entr_loss']
                self.train_devr_loss = checkpoint['train_devr_loss']
                self.test_tot_loss = checkpoint['test_tot_loss']
                self.test_mse_loss = checkpoint['test_mse_loss']
                self.test_smooth_loss = checkpoint['test_smooth_loss']
                self.test_entr_loss = checkpoint['test_entr_loss']
                self.test_devr_loss = checkpoint['test_devr_loss']
                print('Loaded model from epoch: {}'.format(self.start_epoch),flush=True)
                print('Model stats: train loss={:.3e}, test loss={:.3e}'.format(self.train_tot_loss,self.test_tot_loss),flush=True)
        else:
            self.start_epoch = 0
            self.train_tot_loss = float('inf')
            self.train_mse_loss = float('inf')
            self.train_smooth_loss = float('inf')
            self.train_entr_loss = float('inf')
            self.train_devr_loss = float('inf')
            self.test_tot_loss = float('inf')
            self.test_mse_loss = float('inf')
            self.test_smooth_loss = float('inf')
            self.test_entr_loss = float('inf')
            self.test_devr_loss = float('inf')

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self,dataset):
        num_workers = min(self.batch,mp.cpu_count())
        print("Using {} number of workers to load data for training".format(num_workers))
        train_loader = DataLoader(dataset,batch_size=self.batch,shuffle=True,num_workers=num_workers)

        self.model.train()
       
        num_batch = 0 
        avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_entr_loss,avg_devr_loss = 0.0,0.0,0.0,0.0,0.0
        for data_in,mask_in in train_loader:
            data_in = data_in.to(self.device)
            mask_in = mask_in.to(self.device)

            self.optimizer.zero_grad()
            seg,norm_seg,data_out = self.model(data_in)
            
            mse_loss = self.criterion.mse_loss(data_in,data_out,norm_seg,mask_in)
            if self.smooth_reg>0:
                smooth_loss = self.criterion.smooth_loss(norm_seg,mask_in)
            else:
                smooth_loss = torch.FloatTensor([0]).to(self.device)
            if self.entr_reg>0:
                entr_loss = self.criterion.entr_loss(seg,norm_seg)
            else:
                entr_loss = torch.FloatTensor([0]).to(self.device)
            if self.devr_reg>0:
                devr_loss = self.criterion.devr_loss(norm_seg,mask_in)
            else:
                devr_loss = torch.FloatTensor([0]).to(self.device)
            tot_loss = mse_loss+smooth_loss+entr_loss+devr_loss

            tot_loss.backward()
            self.optimizer.step()

            batch_mse_loss = mse_loss.item()
            batch_smooth_loss = smooth_loss.item()
            batch_entr_loss = entr_loss.item()
            batch_devr_loss = devr_loss.item()
            batch_tot_loss = tot_loss.item()

            avg_mse_loss += batch_mse_loss
            avg_smooth_loss += batch_smooth_loss
            avg_entr_loss += batch_entr_loss
            avg_devr_loss += batch_devr_loss
            avg_tot_loss += batch_tot_loss
            num_batch += 1
            print("TRAIN: batch losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, entr {:.2e}, devr {:.2e}".format(batch_tot_loss,batch_mse_loss,batch_smooth_loss,batch_entr_loss,batch_devr_loss))

        avg_tot_loss /= num_batch
        avg_mse_loss /= num_batch
        avg_smooth_loss /= num_batch
        avg_entr_loss /= num_batch
        avg_devr_loss /= num_batch
        print("TRAIN: average losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, entr {:.2e}, devr {:.2e}".format(avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_entr_loss,avg_devr_loss))

        self.train_tot_loss = avg_tot_loss
        self.train_mse_loss = avg_mse_loss
        self.train_smooth_loss = avg_smooth_loss
        self.train_entr_loss = avg_entr_loss
        self.train_devr_loss = avg_devr_loss
        return avg_tot_loss

    def test(self,dataset):
        num_workers = min(self.batch,mp.cpu_count())
        print("Using {} number of workers to load data for testing".format(num_workers))
        test_loader = DataLoader(dataset,batch_size=self.batch,shuffle=False,num_workers=num_workers)
        
        self.model.eval()
        
        num_batch = 0 
        avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_entr_loss,avg_devr_loss = 0.0,0.0,0.0,0.0,0.0
        with torch.no_grad():
            for data_in,mask_in in test_loader:
                data_in = data_in.to(self.device)
                mask_in = mask_in.to(self.device)
                seg,norm_seg,data_out = self.model(data_in)

                mse_loss = self.criterion.mse_loss(data_in,data_out,norm_seg,mask_in)
                if self.smooth_reg>0:
                    smooth_loss = self.criterion.smooth_loss(norm_seg,mask_in)
                else:
                    smooth_loss = torch.FloatTensor([0]).to(self.device)
                if self.entr_reg>0:
                    entr_loss = self.criterion.entr_loss(seg,norm_seg)
                else:
                    entr_loss = torch.FloatTensor([0]).to(self.device)
                if self.devr_reg>0:
                    devr_loss = self.criterion.devr_loss(norm_seg,mask_in)
                else:
                    devr_loss = torch.FloatTensor([0]).to(self.device)
                tot_loss = mse_loss+smooth_loss+entr_loss+devr_loss
                
                batch_mse_loss = mse_loss.item()
                batch_smooth_loss = smooth_loss.item()
                batch_entr_loss = entr_loss.item()
                batch_devr_loss = devr_loss.item()
                batch_tot_loss = tot_loss.item()

                avg_mse_loss += batch_mse_loss
                avg_smooth_loss += batch_smooth_loss
                avg_entr_loss += batch_entr_loss
                avg_devr_loss += batch_devr_loss
                avg_tot_loss += batch_tot_loss
                num_batch += 1
                print("TEST: batch losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, entr {:.2e}, devr {:.2e}".format(batch_tot_loss,batch_mse_loss,batch_smooth_loss,batch_entr_loss,batch_devr_loss))
                    
        avg_tot_loss /= num_batch
        avg_mse_loss /= num_batch
        avg_smooth_loss /= num_batch
        avg_entr_loss /= num_batch
        avg_devr_loss /= num_batch
        print("TEST: average losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, entr {:.2e}, devr {:.2e}".format(avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_entr_loss,avg_devr_loss))
        
        self.test_tot_loss = avg_tot_loss
        self.test_mse_loss = avg_mse_loss
        self.test_smooth_loss = avg_smooth_loss
        self.test_entr_loss = avg_entr_loss
        self.test_devr_loss = avg_devr_loss
        return avg_tot_loss

    def segment(self,dataset,masked=False):
        #num_workers = min(self.batch,mp.cpu_count())
        #print("Using {} number of workers to load data for segmentation".format(num_workers))
        loader = DataLoader(dataset,batch_size=self.batch,shuffle=False)
        self.model.eval()
        
        inp,segm,rec = [],[],[]
        with torch.no_grad():
            for idx,(data_in,mask_in) in enumerate(loader):
                inp.append(data_in.numpy())
                data_in = data_in.to(self.device)
                _,segtemp,rectemp = self.model(data_in)
                s = segtemp.cpu().numpy()[:,:,np.newaxis] #3rd axis is channels (like RGB)
                r = np.stack([r.cpu().numpy() for r in rectemp],axis=1) #2nd axis is seg label 
                if masked:
                    mk = mask_in.cpu().numpy()[:,np.newaxis] #insert an axis at 2nd or 3rd positions
                    segm.append(s*mk)
                    rec.append(r*mk)
                else:
                    segm.append(s)
                    rec.append(r)
        return np.concatenate(segm,axis=0),np.concatenate(rec,axis=0),np.concatenate(inp,axis=0)

#    def classrec(self,dataset,masked=False):
#        num_workers = min(self.batch,mp.cpu_count())
#        print("Using {} number of workers to load data for class reconstruction".format(num_workers))
#        loader = DataLoader(dataset,batch_size=self.batch,shuffle=False,num_workers=num_workers)
#        self.model.eval()
#        inp,recs = [],[]
#        with torch.no_grad():
#            for idx,(data_in,mask_in) in enumerate(loader):
#                inp.append(data_in.numpy())
#                data_in = data_in.to(self.device)
#                clrecs = []
#                _,segm,_ = self.model(data_in)
#                segm_np = segm.cpu().numpy()
#                mk = mask_in.cpu().numpy()
#                for i,aenc in enumerate(self.autoencs):
#                    z = [data_in*segm[:,i:i+1],data_in*(1-segm[:,i:i+1])]
#                    z = torch.cat(z,dim=1)
#                    r = aenc(z).cpu().numpy()
#                    if masked:
#                        clrecs.append(r*mk)
#                    else:
#                        clrecs.append(r)
#                recs.append(np.concatenate(clrecs,axis=1)*segm_np)
#        return np.concatenate(recs,axis=0),np.concatenate(inp,axis=0)

    def get_checkpoint_path(self, epoch):
        return os.path.join(self.checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))

    def checkpoint(self, epoch):
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'train_tot_loss': self.train_tot_loss,
            'train_mse_loss': self.train_mse_loss,
            'train_smooth_loss': self.train_smooth_loss,
            'train_entr_loss': self.train_entr_loss,
            'train_devr_loss': self.train_devr_loss,
            'test_tot_loss': self.test_tot_loss,
            'test_mse_loss': self.test_mse_loss,
            'test_smooth_loss': self.test_smooth_loss,
            'test_entr_loss': self.test_entr_loss,
            'test_devr_loss': self.test_devr_loss,
        }
        checkpoint_path = self.get_checkpoint_path(epoch)
        if os.path.exists(checkpoint_path):
            print('WARNING: Overwriting existing checkpoint path: {}'.format(checkpoint_path), flush=True)
        torch.save(state_dict, checkpoint_path)
