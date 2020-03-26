import matplotlib.pyplot as plt
import nibabel as nib

fixa_ptr = nib.load('/p/lustre1/mohan3/Data/TBI/2mm/debug/test/618952-tissue_2mm.nii.gz')
fixa_vol = fixa_ptr.get_fdata()

autoa_ptr = nib.load('/p/lustre1/mohan3/Data/TBI/2mm/debug/test/618952_T1w_brain_2_aaparts.nii.gz')
autoa_vol = autoa_ptr.get_fdata()

inp_ptr = nib.load('/p/lustre1/mohan3/Data/TBI/2mm/debug/test/618952_T1w_brain_2_aainput.nii.gz')
inp_vol = inp_ptr.get_fdata()

sh = fixa_vol.shape

def save_img(img,filen,cmap=None):
    plt.imshow(img,cmap=cmap)
    plt.colorbar()
    plt.savefig(filen)
    plt.close()

save_img(fixa_vol[sh[0]//2],'/g/g90/mohan3/temp/fixa_z.png','tab20')
save_img(fixa_vol[:,sh[1]//2],'/g/g90/mohan3/temp/fixa_y.png','tab20')
save_img(fixa_vol[:,:,sh[2]//2],'/g/g90/mohan3/temp/fixa_x.png','tab20')

save_img(autoa_vol[sh[0]//2],'/g/g90/mohan3/temp/autoa_z.png','tab20')
save_img(autoa_vol[:,sh[1]//2],'/g/g90/mohan3/temp/autoa_y.png','tab20')
save_img(autoa_vol[:,:,sh[2]//2],'/g/g90/mohan3/temp/autoa_x.png','tab20')

save_img(inp_vol[sh[0]//2],'/g/g90/mohan3/temp/inp_z.png')
save_img(inp_vol[:,sh[1]//2],'/g/g90/mohan3/temp/inp_y.png')
save_img(inp_vol[:,:,sh[2]//2],'/g/g90/mohan3/temp/inp_x.png')

