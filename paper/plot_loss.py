import numpy as np
import matplotlib.pyplot as plt
import os

achan = 8

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr{}_freqs0.05'

FONTSZ = 12
LINEIND_WIDTH = 3.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title

out_dir = os.path.join(log_dir.format(achan,1.0),'paper')
train_mse_loss = np.load(os.path.join(out_dir,'train_mse_loss.npy')) 
train_smooth_loss = np.load(os.path.join(out_dir,'train_smooth_loss.npy')) 
train_devr_loss = np.load(os.path.join(out_dir,'train_devr_loss.npy'))
train_tot_loss = np.load(os.path.join(out_dir,'train_tot_loss.npy'))

plt.figure(figsize=(5,2))
plt.plot(train_mse_loss)
plt.plot(train_smooth_loss)
plt.plot(train_devr_loss)
plt.plot(train_tot_loss)
plt.yscale('log')
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.legend(['REL','NSSL','ADL','Total'],loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.2),fancybox=True,shadow=True)
plt.savefig(os.path.join(out_dir,'losses_aenc{}.png'.format(achan)),tight_layout=True)
plt.close() 

train_tot_devr1 = np.load(os.path.join(log_dir.format(achan,1.0),'paper','train_tot_loss.npy'))
test_tot_devr1 = np.load(os.path.join(log_dir.format(achan,1.0),'paper','test_tot_loss.npy'))
train_tot_devr0 = np.load(os.path.join(log_dir.format(achan,0.0),'paper','train_tot_loss.npy'))
test_tot_devr0 = np.load(os.path.join(log_dir.format(achan,0.0),'paper','test_tot_loss.npy'))

plt.figure(figsize=(5,2))
plt.plot(train_tot_devr1)
plt.plot(test_tot_devr1)
plt.plot(train_tot_devr0)
plt.plot(test_tot_devr0)
plt.yscale('log')
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.legend(['Train ADL 1.0','Test ADL 1.0','Train ADL 0.0','Test ADL 0.0'],loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.2),fancybox=True,shadow=True)
plt.savefig(os.path.join(out_dir,'lossadl_aenc{}.png'.format(achan)),tight_layout=True)
plt.close() 

