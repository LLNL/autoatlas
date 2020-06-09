import torch
import os
from autoatlas.cnn import UNet,AutoEnc
from autoatlas.models import SegmRecon
from torch.utils.data import DataLoader
import progressbar
import numpy as np
import multiprocessing as mp
import h5py

def partition_encode(seg,mask):
    neighs = [[0,0,1],  [0,1,-1],[0,1,0], [0,1,1],
          [1,-1,-1],[1,-1,0],[1,-1,1],
          [1,0,-1], [1,0,0], [1,0,1],
          [1,1,-1], [1,1,0], [1,1,1]]
    nums,dens = [],[]
    mask = mask[np.newaxis]
    for (Nz,Ny,Nx) in neighs:
        H = np.concatenate((seg[:,-Nz:],seg[:,:-Nz]),axis=1)
        H = np.concatenate((H[:,:,-Ny:],  H[:,:,:-Ny]),   axis=2)
        H = np.concatenate((H[:,:,:,-Nx:],H[:,:,:,:-Nx]), axis=3)
        H = seg*H
        W = np.concatenate((mask[:,-Nz:],  mask[:,:-Nz]), axis=1)
        W = np.concatenate((W[:,:,-Ny:],  W[:,:,:-Ny]),   axis=2)
        W = np.concatenate((W[:,:,:,-Nx:],W[:,:,:,:-Nx]), axis=3)
        W = mask*W
        nums.append(np.sum(H*W,axis=(1,2,3)))
        dens.append(np.sum(W,axis=(1,2,3)))
    area_meas = np.sum(np.stack(nums,axis=-1),axis=-1)/np.sum(np.stack(dens,axis=-1),axis=-1) 
    vol_meas = np.sum(seg*mask,axis=(1,2,3))/np.sum(mask,axis=(1,2,3))
    return vol_meas,area_meas

class CustomLoss:
    def __init__(self,num_labels,sizes,rel_reg,smooth_reg,devr_reg,roi_reg,norm_pow,devr_mult,roi_mult,device):
        print('CustomLoss: num_labels={},sizes={},rel_reg={},smooth_reg={},devr_reg={},roi_reg={},norm_pow={},devr_mult={},roi_mult={},device={}'.format(num_labels,sizes,rel_reg,smooth_reg,devr_reg,roi_reg,norm_pow,devr_mult,roi_mult,device))
        self.dim = len(sizes)
        self.dimlist = [2+i for i in range(self.dim)]
        
        self.rel_reg = rel_reg
        self.smooth_reg = smooth_reg
        self.devr_reg = devr_reg
        self.roi_reg = roi_reg

        self.devr_mult = devr_mult
        self.min_freqs = self.devr_mult/num_labels

        self.roi_mult = roi_mult
        if self.dim==3:
            self.roi_radft = self.roi_mult*((3.0/(4.0*np.pi))**(1.0/3))
        elif self.dim==2:
            self.roi_radft = self.roi_mult*((1.0/np.pi)**(1.0/2))
        else:
            raise ValueError('Only 3D and 2D inputs are supported.')
        print('CustomLoss: min_freqs={},roi_radft={}'.format(self.min_freqs,self.roi_radft))

        self.norm_pow = norm_pow
        if self.dim==3:
            self.neighs = [[0,0,1],  [0,1,-1],[0,1,0], [0,1,1],
                       [1,-1,-1],[1,-1,0],[1,-1,1],
                       [1,0,-1], [1,0,0], [1,0,1],
                       [1,1,-1], [1,1,0], [1,1,1]]
        elif self.dim==2:
            self.neighs = [[0,1,np.nan],[1,-1,np.nan],[1,0,np.nan],[1,1,np.nan]]
        else:
            raise ValueError('dim must be either 2 or 3')
        self.softmax = torch.nn.Softmax(dim=1)
        #self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.eps = 1e-10
        
        z = torch.arange(0,sizes[0],1)
        y = torch.arange(0,sizes[1],1)
        if self.dim == 3:
            x = torch.arange(0,sizes[2],1)

        if self.dim == 3:
            grid_z,grid_y,grid_x = torch.meshgrid(z,y,x)
        else:
            grid_z,grid_y = torch.meshgrid(z,y)
           
        self.zero_tensor = torch.FloatTensor([0.0]).to(device) 
        self.two_tensor = torch.FloatTensor([2.0]).to(device) 
        self.grid_z = torch.unsqueeze(torch.unsqueeze(grid_z,dim=0),dim=0).type(torch.FloatTensor).to(device)
        self.grid_y = torch.unsqueeze(torch.unsqueeze(grid_y,dim=0),dim=0).type(torch.FloatTensor).to(device)
        if self.dim == 3:
            self.grid_x = torch.unsqueeze(torch.unsqueeze(grid_x,dim=0),dim=0).type(torch.FloatTensor).to(device)

    def mse_loss(self,gtruth,recs,seg,mask):
        assert not torch.isnan(gtruth).any()
        assert not torch.isnan(mask).any()
        assert not torch.isnan(seg).any()

        mse_losses = []
        den = torch.sum(mask,dim=self.dimlist)
        for i,r in enumerate(recs):
            assert not torch.isnan(r).any()
            num = torch.mean(torch.abs(gtruth-r)**self.norm_pow,dim=1,keepdim=True)
            num = torch.sum(num*seg[:,i:i+1]*mask,dim=self.dimlist)
            mse_losses.append(num/den)
        #return torch.mean(torch.stack(mse_losses))
        mse_losses = torch.sum(torch.stack(mse_losses,dim=-1),dim=-1)

        assert (not torch.isnan(mse_losses).any()),torch.isnan(mse_losses)
        return self.rel_reg*torch.mean(mse_losses)

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
        assert (not torch.isnan(smooth_loss).any()),torch.isnan(smooth_loss)
        return self.smooth_reg*torch.mean(smooth_loss)

    def devr_loss(self,seg,mask):
        clp = torch.sum(seg*mask,dim=self.dimlist)/torch.sum(mask,dim=self.dimlist)
        clp = -torch.log((clp+self.eps*self.min_freqs)/self.min_freqs)
        clp = torch.clamp(clp,min=0)
        assert (not torch.isnan(clp).any()),torch.isnan(clp)
        return self.devr_reg*torch.mean(clp)

    def roi_loss(self,seg,mask):
        seg = seg*mask
        if self.dim==3:
            roi_rad = self.roi_radft*(torch.sum(seg,dim=self.dimlist,keepdim=True)**(1.0/3))
        else:
            roi_rad = self.roi_radft*(torch.sum(seg,dim=self.dimlist,keepdim=True)**(1.0/2))
        roi_rad = torch.where(roi_rad>self.two_tensor,roi_rad,self.two_tensor) 
 
        seg_sum = torch.sum(seg,dim=self.dimlist,keepdim=True)
        mask_sum = torch.sum(mask,dim=self.dimlist,keepdim=True)
        seg = seg/(seg_sum+self.eps)
        
        centr_z = torch.sum(seg*self.grid_z,dim=self.dimlist,keepdim=True)  
        centr_y = torch.sum(seg*self.grid_y,dim=self.dimlist,keepdim=True)  
        if self.dim == 3:
            centr_x = torch.sum(seg*self.grid_x,dim=self.dimlist,keepdim=True)  
    
        dist = (self.grid_z-centr_z)*(self.grid_z-centr_z)
        dist = dist + (self.grid_y-centr_y)*(self.grid_y-centr_y)
        if self.dim == 3:
            dist = dist + (self.grid_x-centr_x)*(self.grid_x-centr_x)
        dist = torch.sqrt(dist)       

        lhood = torch.sum(torch.sigmoid(4.6*(roi_rad-dist))*seg,dim=self.dimlist,keepdim=True)
        cond = seg_sum/mask_sum>0.1*self.min_freqs
        lhood = torch.where(cond,torch.log(lhood),self.zero_tensor)
        #print('Total seg_sum/mask_sum <= 0.1 is {}. seg_sum size is {}'.format(torch.sum(torch.bitwise_not(cond)),seg_sum.size()))
        #lhood = torch.where(seg_sum>self.eps,torch.log(lhood),self.zero_tensor)
        #lhood = torch.log(lhood)
        assert (not torch.isnan(lhood).any()),torch.isnan(lhood)
        return -self.roi_reg*torch.mean(lhood) 

#    def entr_loss(self,seg,mask)
#        if self.dim == 3:
#            seg_den = torch.sum(seg,dim=(2,3,4),keepdim=True)
#        else:
#            seg_den = torch.sum(seg,dim=(2,3),keepdim=True)
#
#        seg_norm = seg/seg_den 
         
    
class AutoAtlas:
    def __init__(self,num_labels=None,sizes=None,data_chan=None,rel_reg=None,smooth_reg=None,devr_reg=None,roi_reg=None,devr_mult=None,roi_mult=None,batch=None,lr=None,unet_chan=None,unet_blocks=None,unet_layblk=None,aenc_chan=None,aenc_depth=None,re_pow=None,distr=None,device=None,load_ckpt_epoch=None,ckpt_file='model_epoch_{}.pth'):
        self.ARGS = {}
        self.ARGS['ckpt_file'] = ckpt_file
        self.ARGS['load_ckpt_epoch'] = load_ckpt_epoch

        ckpt_path = None
        if load_ckpt_epoch is not None:
            ckpt_path = self.get_ckpt_path(load_ckpt_epoch,ckpt_file)
            try:
                ckpt = torch.load(ckpt_path)
            except FileNotFoundError:
                raise ValueError('Checkpoint path does not exist: {}'.format(ckpt_path))
            else:
                for key in ckpt['ARGS'].keys():
                    if key not in self.ARGS.keys():
                        self.ARGS[key] = ckpt['ARGS'][key]
                self.start_epoch = ckpt['epoch']
                self.train_tot_loss = ckpt['train_tot_loss']
                self.train_mse_loss = ckpt['train_mse_loss']
                self.train_smooth_loss = ckpt['train_smooth_loss']
                self.train_devr_loss = ckpt['train_devr_loss']
                self.train_roi_loss = ckpt['train_roi_loss']
                self.test_tot_loss = ckpt['test_tot_loss']
                self.test_mse_loss = ckpt['test_mse_loss']
                self.test_smooth_loss = ckpt['test_smooth_loss']
                self.test_devr_loss = ckpt['test_devr_loss']
                self.test_roi_loss = ckpt['test_roi_loss']
                print('Loaded model from epoch: {}'.format(self.start_epoch),flush=True)
                print('Model stats: train loss={:.3e}, test loss={:.3e}'.format(self.train_tot_loss,self.test_tot_loss),flush=True)
        else:
            self.start_epoch = 0
            self.train_tot_loss = float('inf')
            self.train_mse_loss = float('inf')
            self.train_smooth_loss = float('inf')
            self.train_devr_loss = float('inf')
            self.test_tot_loss = float('inf')
            self.test_mse_loss = float('inf')
            self.test_smooth_loss = float('inf')
            self.test_devr_loss = float('inf')
            self.ARGS.update({'num_labels':num_labels,'sizes':sizes,'data_chan':data_chan,
                        'rel_reg':rel_reg,'smooth_reg':smooth_reg,'devr_reg':devr_reg,'roi_reg':roi_reg,
                        'devr_mult':devr_mult,'roi_mult':roi_mult,
                        'batch':batch,'lr':lr,'unet_chan':unet_chan,
                        'unet_blocks':unet_blocks,'unet_layblk':unet_layblk,
                        'aenc_chan':aenc_chan,'aenc_depth':aenc_depth,'re_pow':re_pow,
                        'distr':distr,'device':device})
        
        dim = len(self.ARGS['sizes'])
        self.dev_list = None
        device = self.ARGS['device']
        if self.ARGS['distr']==True and device=='cuda':
            self.acc_dev = '{}:0'.format(device)
            dev_count = torch.cuda.device_count()
            assert dev_count>=1 and dev_count<=4
            self.cnnenc_dev,self.cnndec_dev = '{}:0'.format(device),'{}:0'.format(device)
            if dev_count>=3:
                self.cnndec_dev = '{}:1'.format(device)
            self.aenc_devs = [] 
            for i in range(self.ARGS['num_labels']):
                if dev_count <= 3:
                    devid = dev_count-1
                else:
                    devid = 2 if i%2==0 else 3
                self.aenc_devs.append('{}:{}'.format(device,devid))
        elif self.ARGS['distr']==False and device=='cuda':
            self.acc_dev = 'cuda'
            self.cnnenc_dev,self.cnndec_dev = 'cuda','cuda'
            self.aenc_devs = ['cuda' for i in range(self.ARGS['num_labels'])] 
        else:
            raise ValueError('Combination of distr={} and device={} is not supported'.format(self.ARGS['distr'],device))
        assert self.acc_dev==self.cnnenc_dev

        #print("Devices: Accumulator {}, CNN encoder {}, CNN decoder {}, autoencoders {}".format(self.acc_dev,self.cnnenc_dev,self.cnndec_dev,self.aenc_devs))
        self.cnn = UNet(self.ARGS['num_labels'],dim=dim,data_chan=self.ARGS['data_chan'],kernel_size=3,filters=self.ARGS['unet_chan'],blocks=self.ARGS['unet_blocks'],layers_block=self.ARGS['unet_layblk'],batch_norm=False,pad_type='SAME',enc_dev=self.cnnenc_dev,dec_dev=self.cnndec_dev)
   
        self.autoencs = torch.nn.ModuleList([])
        for i in range(self.ARGS['num_labels']):
            self.autoencs.append(AutoEnc(self.ARGS['sizes'],data_chan=self.ARGS['data_chan'],kernel_size=7,filters=self.ARGS['aenc_chan'],depth=self.ARGS['aenc_depth'],pool=2,batch_norm=False,pad_type='SAME').to(self.aenc_devs[i])) 
        
        self.model = SegmRecon(self.cnn,self.autoencs,self.aenc_devs)
        if self.ARGS['distr']==False and device == 'cuda':
            print('Using torch.nn.DataParallel for parallel processing')
            devids = list(range(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model,device_ids=devids)
            torch.backends.cudnn.benchmark = True
       
        self.criterion = CustomLoss(num_labels=self.ARGS['num_labels'],sizes=self.ARGS['sizes'],rel_reg=self.ARGS['rel_reg'],smooth_reg=self.ARGS['smooth_reg'],devr_reg=self.ARGS['devr_reg'],roi_reg=self.ARGS['roi_reg'],norm_pow=self.ARGS['re_pow'],devr_mult=self.ARGS['devr_mult'],roi_mult=self.ARGS['roi_mult'],device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.ARGS['lr']) 
        if ckpt_path is not None:
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])

    def train(self,dataset):
        num_workers = min(self.ARGS['batch'],mp.cpu_count())
        print("Using {} number of workers to load data for training".format(num_workers))
        train_loader = DataLoader(dataset,batch_size=self.ARGS['batch'],shuffle=True,num_workers=num_workers)

        self.model.train()
       
        num_batch = 0 
        avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_devr_loss,avg_roi_loss = 0.0,0.0,0.0,0.0,0.0
        for data_in,mask_in,_,_ in train_loader:
            data_in = data_in.to(self.cnnenc_dev)
            mask_in = mask_in.to(self.acc_dev)

            self.optimizer.zero_grad()
            norm_seg,data_out,_ = self.model(data_in)
            norm_seg = norm_seg.to(self.acc_dev)           
            data_out = [d.to(self.acc_dev) for d in data_out]         
 
            if self.ARGS['rel_reg']>0:
                mse_loss = self.criterion.mse_loss(data_in,data_out,norm_seg,mask_in)
            else:
                mse_loss = torch.FloatTensor([0]).to(self.acc_dev)
            if self.ARGS['smooth_reg']>0:
                smooth_loss = self.criterion.smooth_loss(norm_seg,mask_in)
            else:
                smooth_loss = torch.FloatTensor([0]).to(self.acc_dev)
            if self.ARGS['devr_reg']>0:
                devr_loss = self.criterion.devr_loss(norm_seg,mask_in)
            else:
                devr_loss = torch.FloatTensor([0]).to(self.acc_dev)
            if self.ARGS['roi_reg']>0:
                roi_loss = self.criterion.roi_loss(norm_seg,mask_in)
            else:
                roi_loss = torch.FloatTensor([0]).to(self.acc_dev)
            tot_loss = mse_loss+smooth_loss+devr_loss+roi_loss

            #import pdb; pdb.set_trace()
            tot_loss.backward()
            self.optimizer.step()

            batch_mse_loss = mse_loss.item()
            batch_smooth_loss = smooth_loss.item()
            batch_devr_loss = devr_loss.item()
            batch_roi_loss = roi_loss.item()
            batch_tot_loss = tot_loss.item()

            avg_mse_loss += batch_mse_loss
            avg_smooth_loss += batch_smooth_loss
            avg_devr_loss += batch_devr_loss
            avg_roi_loss += batch_roi_loss
            avg_tot_loss += batch_tot_loss
            num_batch += 1
            print("TRAIN: batch losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, devr {:.2e}, roi {:.2e}".format(batch_tot_loss,batch_mse_loss,batch_smooth_loss,batch_devr_loss,batch_roi_loss))

        avg_tot_loss /= num_batch
        avg_mse_loss /= num_batch
        avg_smooth_loss /= num_batch
        avg_devr_loss /= num_batch
        avg_roi_loss /= num_batch
        print("TRAIN: average losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, devr {:.2e}, roi {:.2e}".format(avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_devr_loss,avg_roi_loss))

        self.train_tot_loss = avg_tot_loss
        self.train_mse_loss = avg_mse_loss
        self.train_smooth_loss = avg_smooth_loss
        self.train_devr_loss = avg_devr_loss
        self.train_roi_loss = avg_roi_loss
        return avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_devr_loss,avg_roi_loss

    def test(self,dataset):
        num_workers = min(self.ARGS['batch'],mp.cpu_count())
        print("Using {} number of workers to load data for testing".format(num_workers))
        test_loader = DataLoader(dataset,batch_size=self.ARGS['batch'],shuffle=False,num_workers=num_workers)
        
        self.model.eval()
        
        num_batch = 0 
        avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_devr_loss,avg_roi_loss = 0.0,0.0,0.0,0.0,0.0
        with torch.no_grad():
            for data_in,mask_in,_,_ in test_loader:
                data_in = data_in.to(self.cnnenc_dev)
                mask_in = mask_in.to(self.acc_dev)

                norm_seg,data_out,_ = self.model(data_in)
                norm_seg = norm_seg.to(self.acc_dev)
                data_out = [d.to(self.acc_dev) for d in data_out]         

                if self.ARGS['rel_reg']>0:
                    mse_loss = self.criterion.mse_loss(data_in,data_out,norm_seg,mask_in)
                else:
                    mse_loss = torch.FloatTensor([0]).to(self.acc_dev)
                if self.ARGS['smooth_reg']>0:
                    smooth_loss = self.criterion.smooth_loss(norm_seg,mask_in)
                else:
                    smooth_loss = torch.FloatTensor([0]).to(self.acc_dev)
                if self.ARGS['devr_reg']>0:
                    devr_loss = self.criterion.devr_loss(norm_seg,mask_in)
                else:
                    devr_loss = torch.FloatTensor([0]).to(self.acc_dev)
                if self.ARGS['roi_reg']>0:
                    roi_loss = self.criterion.roi_loss(norm_seg,mask_in)
                else:
                    roi_loss = torch.FloatTensor([0]).to(self.acc_dev)
                tot_loss = mse_loss+smooth_loss+devr_loss+roi_loss
                
                batch_mse_loss = mse_loss.item()
                batch_smooth_loss = smooth_loss.item()
                batch_devr_loss = devr_loss.item()
                batch_roi_loss = roi_loss.item()
                batch_tot_loss = tot_loss.item()

                avg_mse_loss += batch_mse_loss
                avg_smooth_loss += batch_smooth_loss
                avg_devr_loss += batch_devr_loss
                avg_roi_loss += batch_roi_loss
                avg_tot_loss += batch_tot_loss
                num_batch += 1
                print("TEST: batch losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, devr {:.2e}, roi {:.2e}".format(batch_tot_loss,batch_mse_loss,batch_smooth_loss,batch_devr_loss,batch_roi_loss))
                    
        avg_tot_loss /= num_batch
        avg_mse_loss /= num_batch
        avg_smooth_loss /= num_batch
        avg_devr_loss /= num_batch
        avg_roi_loss /= num_batch
        print("TEST: average losses: tot {:.2e}, mse {:.2e}, smooth {:.2e}, devr {:.2e}, roi {:.2e}".format(avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_devr_loss,avg_roi_loss))
        
        self.test_tot_loss = avg_tot_loss
        self.test_mse_loss = avg_mse_loss
        self.test_smooth_loss = avg_smooth_loss
        self.test_devr_loss = avg_devr_loss
        self.test_roi_loss = avg_roi_loss
        return avg_tot_loss,avg_mse_loss,avg_smooth_loss,avg_devr_loss,avg_roi_loss

    def process(self,dataset,ret_input=True):
        loader = DataLoader(dataset,batch_size=self.ARGS['batch'],shuffle=False)
        self.model.eval()
        with torch.no_grad():
            segs,recs,inps,masks,codes,din_files,mk_files = [],[],[],[],[],[],[]
            for idx,(din,mk,din_fl,mk_fl) in enumerate(loader):
                din = din.to(self.cnnenc_dev)
                sg,rc,cd = self.model(din)
                if ret_input:
                    din = np.squeeze(din.cpu().numpy(),axis=1)
                    din = np.split(din,len(din_fl),axis=0) #List of ndarrays with singleton axis 0
                    inps.extend([np.squeeze(sl,axis=0) for sl in din])
                sg = np.split(sg.cpu().numpy(),len(din_fl),axis=0)
                segs.extend([np.squeeze(sl,axis=0) for sl in sg])
                rc = [np.squeeze(sl.cpu().numpy(),axis=1) for sl in rc]
                rc = np.split(np.stack(rc,axis=1),len(din_fl),axis=0)
                recs.extend([np.squeeze(sl,axis=0) for sl in rc])
                mk = np.split(np.squeeze(mk.cpu().numpy(),axis=1),len(din_fl),axis=0)
                masks.extend([np.squeeze(sl,axis=0) for sl in mk])
                cd = np.split(cd.cpu().numpy(),len(din_fl),axis=0) 
                codes.extend([np.squeeze(sl,axis=0) for sl in cd])
                din_files.extend(list(din_fl))
                mk_files.extend(list(mk_fl))
 
        if ret_input:
            return segs,recs,masks,codes,din_files,mk_files,inps
        else:
            return segs,recs,masks,codes,din_files,mk_files

    def get_ckpt_path(self, epoch, filen):
        return filen.format(epoch)

    def ckpt(self, epoch, filen):
        state_dict = {
            'ARGS':self.ARGS,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_tot_loss': self.train_tot_loss,
            'train_mse_loss': self.train_mse_loss,
            'train_smooth_loss': self.train_smooth_loss,
            'train_devr_loss': self.train_devr_loss,
            'train_roi_loss': self.train_roi_loss,
            'test_tot_loss': self.test_tot_loss,
            'test_mse_loss': self.test_mse_loss,
            'test_smooth_loss': self.test_smooth_loss,
            'test_devr_loss': self.test_devr_loss,
            'test_roi_loss': self.test_roi_loss,
        }
        ckpt_path = self.get_ckpt_path(epoch,filen)
        if os.path.exists(ckpt_path):
            print('WARNING: Overwriting existing ckpt path: {}'.format(ckpt_path), flush=True)
        torch.save(state_dict, ckpt_path)
