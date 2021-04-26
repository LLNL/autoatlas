from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
import glob
import os
import csv
import matplotlib.pyplot as plt

train_num = 80
in_folder = '/p/gpfs1/mohan3/Data/TBI/HCP/fMRI_flat_images/768x384'
out_folder = '/p/gpfs1/mohan3/Data/TBI/HCP/fMRI_flat_images/proc_768x384'

in_files = np.array(glob.glob(os.path.join(in_folder, '*_rfMRI_REST1_LR_avg.png')))
perm_idx = np.random.permutation(np.arange(0, len(in_files), dtype=int))
train_num = int(np.round(len(in_files)*train_num/100))
train_files = in_files[perm_idx[:train_num]]
test_files = in_files[perm_idx[train_num:]]
   
def write_data(write_folder, write_files): 
    if os.path.exists(write_folder):
        raise ValueError('{} exsits'.format(write_folder))        
    
    os.makedirs(write_folder)

    csv_file = open(os.path.join(write_folder, 'subjects.csv'), mode='w')   
    csv_writer = csv.writer(csv_file, delimiter=',')

    for filen in write_files:
        filen = os.path.split(filen)[-1] 
        ID = filen.split('_')[0]
        csv_writer.writerow([ID])

        print(filen)
        img = np.asarray(Image.open(os.path.join(in_folder, filen)))
        img = np.mean(img, axis=-1)
        
        thresh = threshold_otsu(img)
        mask = img > thresh/2
        img = (img-np.mean(img[mask]))/np.std(img[mask])
        img[np.bitwise_not(mask)] = 0
        img = Image.fromarray(img.astype(np.float)) 
        img.save(os.path.join(write_folder, filen[:-4]+'.tif'))
 
        mask = Image.fromarray(mask.astype(np.uint8))
        filen = filen[:-4]+'_mask.png'
        mask.save(os.path.join(write_folder, filen))

    csv_file.close()

write_data(os.path.join(out_folder, 'train_zn'), train_files) 
write_data(os.path.join(out_folder, 'test_zn'), test_files) 
