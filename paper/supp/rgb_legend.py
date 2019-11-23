import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

achan = 4
num_labels = 16

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
out_folder = os.path.join(log_dir.format(achan),'supp','seg')
if not os.path.exists(out_folder):
    os.makedirs(out_folder) 

rgb_list = np.array([[128,0,0],[170,110,40],[128,128,0],[0,128,128],[0,0,128],[255,255,255],[230,25,75],[245,130,48],[255,255,25],[210,245,60],[60,180,75],[70,240,240],[0,130,200],[145,30,180],[240,50,230],[128,128,128]]).astype(np.uint8)[:num_labels]

for i in range(rgb_list.shape[0]):
    fig = plt.figure(figsize=(5,1))
    ax = fig.add_axes([0,0,1,1])
    rgb = rgb_list[i]    
    rect = matplotlib.patches.Rectangle((0,0),1,1,color=(rgb.astype(float)/255))
    ax.add_patch(rect)
    plt.xticks([]),plt.yticks([])
    plt.savefig(os.path.join(out_folder,'rgb_{}.png'.format(i)),bbox_inches='tight')
    plt.close()
