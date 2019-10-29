import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def stack_plot(vols,filename,title='',sldim=None,nrows=2,cmap=None):
    dims = len(vols)
    sh = vols[0].shape

    ncols = int(np.ceil(dims/nrows))
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(5*ncols,10)) 
    ax = ax.ravel()

    for i in range(dims):
        if sldim=='z':
            im = ax[i].imshow(vols[i][sh[0]//2],cmap=cmap)
        elif sldim=='y':
            im = ax[i].imshow(vols[i][:,sh[1]//2],cmap=cmap)
        elif sldim=='x':
            im = ax[i].imshow(vols[i][:,:,sh[2]//2],cmap=cmap)
        else:
            im = ax[i].imshow(vols[i],cmap=cmap)
            #assumes 2d with or without color
 
        fig.colorbar(im,ax=ax[i])

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def write_nifti(vol,filename):
    vol = np.swapaxes(vol,0,-1)
    img = nib.Nifti1Image(vol,np.eyes(4))
    img.to_filename(filename) 

