import matplotlib.pyplot as plt

def stack_plot(vols,segs,filen,title=''):
    dims = vols[0].shape
 
    fig,ax = plt.subplots(nrows=2,ncols=4,figsize=(30,10)) 

    for i in range(2):
        im = ax[i][0].imshow(vols[i][dims[0]//2],cmap='gray')
        ax[i][0].set_title('Vol z-slice')
        fig.colorbar(im,ax=ax[i][0])

        im = ax[i][1].imshow(segs[i][dims[0]//2],cmap='gray')
        ax[i][1].set_title('Seg z-slice')
        fig.colorbar(im,ax=ax[i][1])

        im = ax[i][2].imshow(vols[i][:,dims[1]//2],cmap='gray')
        ax[i][2].set_title('Vol y-slice')
        fig.colorbar(im,ax=ax[i][2])

        im = ax[i][3].imshow(segs[i][:,dims[1]//2],cmap='gray')
        ax[i][3].set_title('Seg y-slice')
        fig.colorbar(im,ax=ax[i][3])

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filen)
    plt.close()
