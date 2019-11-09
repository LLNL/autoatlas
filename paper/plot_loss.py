import numpy as np
import matplotlib.pyplot as plt
import os

log_dirs = ['/p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc16_11_labels16_smooth0.1_devr0.0_freqs0.05','/p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05','/p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc256_11_labels1_smooth0.2_devr1.0_freqs0.05']

for d in log_dirs:
    mse_loss = np.load(os.path.join(d,'paper','train_mse_loss.npy')) 
    #smooth_loss = np.load(os.path.join(d,'paper','train_smooth_loss.npy')) 
    #devr_loss = np.load(os.path.join(d,'paper','train_devr_loss.npy'))
    plt.plot(mse_loss)
    #plt.plot(smooth_loss)
    #plt.plot(devr_loss)
    plt.yscale('log')
    plt.gca().set_ylim([0.001,100])
    plt.xlabel('Epochs') 
    plt.ylabel('REL Loss')
    #plt.legend(['REL','NSSL','ADL'])
    plt.savefig(os.path.join(d,'paper','train_losses.png'))
    plt.close() 
