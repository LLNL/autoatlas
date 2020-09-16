import numpy as np
import csv
import os
import nibabel as nib
import matplotlib.pyplot as plt
#from scipy.stats import entropy as scipy_entropy

tag_list = ['mxvol_labs16_relr0_0_smooth0_005_devrr0_1_devrm0_9_emb16','mxvol_labs16_smooth0_0_devrr0_1_devrm0_9_emb16','mxvol_labs16_smooth0_005_devrr0_0_devrm0_9_emb16','mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb16']
rel_reg = [0.0,1.0,1.0,1.0]
smooth_reg = [0.005,0.0,0.005,0.005]
devr_reg = [0.1,0.1,0.0,0.1]
plot_colors = ['b','r','g','k']
plot_legends = [r'$\lambda_{RE}=0,\lambda_{NSS}=0.005,\lambda_{AD}=0.1$',
                r'$\lambda_{RE}=1,\lambda_{NSS}=0,\lambda_{AD}=0.1$',
                r'$\lambda_{RE}=1,\lambda_{NSS}=0.005,\lambda_{AD}=0$',
                r'$\lambda_{RE}=1,\lambda_{NSS}=0.005,\lambda_{AD}=0.1$']

smpl_list = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/subjects.txt'

def ret_losses(log_dir):
    epochs,tot_loss,rel_loss,smooth_loss,devr_loss = [],[],[],[],[]
    with open(os.path.join(log_dir,'losses.csv'),'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            epochs.append(int(row['Epoch']))
            tot_loss.append(float(row['Tot Loss']))
            rel_loss.append(float(row['RE Loss']))
            smooth_loss.append(float(row['NSS Loss']))
            devr_loss.append(float(row['ADL Loss']))
    return epochs,tot_loss,rel_loss,smooth_loss,devr_loss

FONTSZ = 12
LINEIND_WIDTH = 3.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title

def plot_losses(filen,epochs,losses):
    for i,loss in enumerate(losses):
        plt.plot(epochs,loss,plot_colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(plot_legends,loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.2),fancybox=True,shadow=True)
    plt.tight_layout()
    plt.savefig(filen)
    plt.close()

tot_losses,rel_losses,smooth_losses,devr_losses = [],[],[],[]
for tag in tag_list:
    epochs,tloss,rloss,sloss,dloss = ret_losses(os.path.join(os.path.join('/p/gpfs1/mohan3/Data/TBI/HCP/2mm/',tag),'test_mx'))
    tot_losses.append(tloss)
    rel_losses.append(rloss)
    smooth_losses.append(sloss)
    devr_losses.append(dloss)

sdir = os.path.join('figs','abl_losses')
if not os.path.exists(sdir):
    os.makedirs(sdir,exist_ok=True)

plot_losses(os.path.join(sdir,'abl_rel_loss.png'),epochs,rel_losses)  
plot_losses(os.path.join(sdir,'abl_smooth_loss.png'),epochs,smooth_losses)  
plot_losses(os.path.join(sdir,'abl_devr_loss.png'),epochs,devr_losses)  
plot_losses(os.path.join(sdir,'abl_tot_loss.png'),epochs,tot_losses)  

 
