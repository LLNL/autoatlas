import numpy as np
import h5py
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import os

achan = 4
test_start = 0
test_num = 24

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
num_labels = 16

FONTSZ = 16
LINEIND_WIDTH = 3.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title


out_folder = os.path.join(log_dir.format(achan),'supp','seg')
if not os.path.exists(out_folder):
    os.makedirs(out_folder) 

test_folder = os.path.join(log_dir.format(achan),'test_aa')
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
test_files.sort()

vals = []
for i in range(test_start,test_start+test_num):
    print(test_files[i])
    with h5py.File(test_files[i],'r') as f:
        seg = np.array(f['segmentation'])
        mk = np.array(f['mask']).astype(bool)
    vals.append(seg[:,mk].ravel())
    
vals = np.concatenate(vals)

plt.figure(figsize=(10,4))
plt.hist(vals,bins=50,density=True)
plt.yscale('log')
plt.xlabel(r'Segmentation probabilities $y_{i,j}$')
plt.ylabel(r'Density')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(out_folder,'prob_hist.png'),tight_layout=True)
plt.close()
