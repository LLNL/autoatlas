from autoatlas.data import NibData
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

def get_dataset(samples,data_filename,mask_filename):
    #smpl_dirs = [os.path.join(data_dir,f) for f in smpl_dirs][:10] #FIX. ONLY FOR DEBUG
    data_files,mask_files = [],[]
    for smpl in samples:
        data_files.append(data_filename.format(smpl))
    for smpl in samples:
        mask_files.append(mask_filename.format(smpl))
    return NibData(data_files,mask_files),data_files,mask_files 
#TEMPORARY: SHOULD INCLUDE CODE TO CHECK SAME METADATA FOR MASK AND DATA FILES IN DATA READER

