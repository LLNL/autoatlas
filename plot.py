import matplotlib.pyplot as plt
import numpy as np

def stack_plot(vols,filename,title='',sldim='z',nrows=2):
    dims = len(vols)
    sh = vols[0].shape

    ncols = int(np.ceil(dims/nrows))
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(5*ncols,10)) 
    ax = ax.ravel()

    for i in range(dims):
        if sldim=='z':
            im = ax[i].imshow(vols[i][sh[0]//2],cmap='gray')
        elif sldim=='y':
            im = ax[i].imshow(vols[i][:,sh[1]//2],cmap='gray')
        else:
            im = ax[i].imshow(vols[i][:,:,sh[2]//2],cmap='gray')
            
        fig.colorbar(im,ax=ax[i])

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
