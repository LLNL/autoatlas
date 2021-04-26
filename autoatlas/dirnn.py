import torch
import os
from autoatlas._cnn import EncPred
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing as mp
import h5py

class DirPredNN:
    def __init__(self,sizes=None,data_chan=None,batch=None,lr=None,wdcy=None,cnn_chan=None,cnn_depth=None,device=None,load_ckpt_epoch=None,ckpt_file=None,task=None,num_labels=None):
        self.ARGS = {}
        self.ARGS['ckpt_file'] = ckpt_file
        self.ARGS['load_ckpt_epoch'] = load_ckpt_epoch

        ckpt_path = None
        if load_ckpt_epoch is not None:
            ckpt_path = self.get_ckpt_path(load_ckpt_epoch,ckpt_file)
            try:
                ckpt = torch.load(ckpt_path,map_location=torch.device(device))
            except FileNotFoundError:
                raise ValueError('Checkpoint path does not exist: {}'.format(ckpt_path))
            else:
                for key in ckpt['ARGS'].keys():
                    if key not in self.ARGS.keys():
                        self.ARGS[key] = ckpt['ARGS'][key]
                self.start_epoch = ckpt['epoch']
                self.train_loss = ckpt['train_loss']
                self.test_loss = ckpt['test_loss']
                print('Loaded model from epoch: {}'.format(self.start_epoch),flush=True)
                print('Model stats: train loss={:.3e}, test loss={:.3e}'.format(self.train_loss,self.test_loss),flush=True)
        else:
            self.start_epoch = 0
            self.train_loss = float('inf')
            self.test_loss = float('inf')
            self.ARGS.update({'sizes':sizes,'data_chan':data_chan,'task':task,'num_labels':num_labels,
                        'batch':batch,'lr':lr,'wdcy':wdcy,'cnn_chan':cnn_chan,'cnn_depth':cnn_depth,'device':device})
        
        dim = len(self.ARGS['sizes'])
        self.dev = self.ARGS['device']

        out_features = 1 if self.ARGS['task'] == 'regression' else self.ARGS['num_labels']
        self.model = EncPred(self.ARGS['sizes'],data_chan=self.ARGS['data_chan'],kernel_size=7,filters=self.ARGS['cnn_chan'],depth=self.ARGS['cnn_depth'],pool=2,out_features=out_features,batch_norm=False,pad_type='SAME').to(self.dev) 
        
        print('Using torch.nn.DataParallel for parallel processing')
        devids = list(range(torch.cuda.device_count()))
        self.model = torch.nn.DataParallel(self.model,device_ids=devids)
        torch.backends.cudnn.benchmark = True
      
        if task == 'regression':
            self.criterion = torch.nn.MSELoss(reduction='mean').to(self.dev)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.dev)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.ARGS['lr'],weight_decay=self.ARGS['wdcy']) 
        if ckpt_path is not None:
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])

    def train(self,dataset):
        num_workers = min(self.ARGS['batch'],mp.cpu_count())
        print("Using {} number of workers to load data for training".format(num_workers))
        train_loader = DataLoader(dataset,batch_size=self.ARGS['batch'],shuffle=True,num_workers=num_workers)

        self.model.train()
       
        num_batch = 0 
        avg_loss = 0.0
        for data_in,mask_in,gt_out,_,_ in train_loader:
            data_in = data_in.to(self.dev)
            mask_in = mask_in.to(self.dev)
            data_in[mask_in == False] = 0
            gt_out = gt_out.to(self.dev)

            self.optimizer.zero_grad()
            data_out = self.model(data_in)

            batch_loss = self.criterion(data_out,gt_out)
            #import pdb; pdb.set_trace()
            batch_loss.backward()
            self.optimizer.step()

            batch_loss = batch_loss.item()
            avg_loss += batch_loss
            num_batch += 1
            print("TRAIN: batch loss: {:.2e}".format(batch_loss))

        avg_loss /= num_batch
        print("TRAIN: average loss: {:.2e}".format(avg_loss))
        self.train_loss = avg_loss
        return avg_loss

    def test(self,dataset):
        num_workers = min(self.ARGS['batch'],mp.cpu_count())
        print("Using {} number of workers to load data for testing".format(num_workers))
        test_loader = DataLoader(dataset,batch_size=self.ARGS['batch'],shuffle=False,num_workers=num_workers)
        
        self.model.eval()
        
        num_batch = 0 
        avg_loss = 0.0
        with torch.no_grad():
            for data_in,mask_in,gt_out,_,_ in test_loader:
                data_in = data_in.to(self.dev)
                mask_in = mask_in.to(self.dev)
                data_in[mask_in == False] = 0
                gt_out = gt_out.to(self.dev)

                data_out = self.model(data_in)
                batch_loss = self.criterion(data_out,gt_out)
                batch_loss = batch_loss.item()

                avg_loss += batch_loss
                num_batch += 1
                print("TEST: batch loss: {:.2e}".format(batch_loss))
                    
        avg_loss /= num_batch
        print("TEST: average loss: {:.2e}".format(avg_loss))
        
        self.test_loss = avg_loss
        return avg_loss

    def process(self,dataset,ret_input=True):
        loader = DataLoader(dataset,batch_size=self.ARGS['batch'],shuffle=False)
        self.model.eval()
        with torch.no_grad():
            data_in,data_out,dfile_in,mfile_in = [],[],[],[]
            for idx,(din,mk,gt,din_fl,mk_fl) in enumerate(loader):
                din = din.to(self.dev)
                mk = mk.to(self.dev)
                din[mk == False] = 0.0
                data_in.extend(list(din))
                dout = self.model(din).cpu().numpy().squeeze()
                data_out.extend(dout.tolist())
                dfile_in.extend(list(din_fl))
                mfile_in.extend(list(mk_fl))
 
        if ret_input:
            return data_out,dfile_in,mfile_in,data_in
        else:
            return data_out,dfile_in,mfile_in

    def get_ckpt_path(self, epoch, filen):
        return filen.format(epoch)

    def ckpt(self, epoch, filen):
        state_dict = {
            'ARGS':self.ARGS,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
        }
        ckpt_path = self.get_ckpt_path(epoch,filen)
        if os.path.exists(ckpt_path):
            print('WARNING: Overwriting existing ckpt path: {}'.format(ckpt_path), flush=True)
        torch.save(state_dict, ckpt_path)
