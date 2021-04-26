from autoatlas.data import NibData, ImageData
import csv
import os

def get_samples(smpl_list):
    samples = []
    with open(smpl_list,'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            samples.append(row[0])
    return samples

def get_dataset(samples,ndim,data_filename,mask_filename,targets=None,task=None,labels=None):
    #smpl_dirs = [os.path.join(data_dir,f) for f in smpl_dirs][:10] #FIX. ONLY FOR DEBUG
    data_files,mask_files = [],[]
    for smpl in samples:
        data_files.append(data_filename.format(smpl))
    for smpl in samples:
        mask_files.append(mask_filename.format(smpl))

    if ndim==3: 
        return NibData(data_files,mask_files,targets=targets,task=task,labels=labels),data_files,mask_files
    elif ndim==2:
        return ImageData(data_files,mask_files),data_files,mask_files
    else:
        raise ValueError('Only 3D data readable using nibabel and 2D data readable using PIL are supported.') 
#TEMPORARY: SHOULD INCLUDE CODE TO CHECK SAME METADATA FOR MASK AND DATA FILES IN DATA READER

