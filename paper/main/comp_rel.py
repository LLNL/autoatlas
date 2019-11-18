import numpy as np
import os
import h5py

num_vols = None

achans = [4,8,16]

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'

for ac in achans:
    test_folder = os.path.join(log_dir.format(ac),'test_aa')
    test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
    test_files.sort()

    foreg_err = 0
    backg_err = 0
    if num_vols is None:
        num_vols = len(test_files)
    print('Number of volumes is {}'.format(num_vols))

    for i in range(num_vols):
        with h5py.File(test_files[i],'r') as f:
            gt = np.array(f['ground_truth'])[np.newaxis]
            seg = np.array(f['segmentation'])
            rec = np.array(f['reconstruction'])
            mask = np.array(f['mask']).astype(float)[np.newaxis]
       
        assert np.min(mask)==0
        assert np.max(mask)==1

        foreg_mask = seg*mask 
        backg_mask = (1-seg)*mask
     
        foreg_err += np.sum((gt-rec)*foreg_mask*(gt-rec))/np.sum(foreg_mask)
        backg_err += np.sum((gt-rec)*backg_mask*(gt-rec))/np.sum(backg_mask)

    foreg_err /= num_vols
    backg_err /= num_vols

    print('{} & {:.3f} & {:.3f}'.format(ac,foreg_err,backg_err))
