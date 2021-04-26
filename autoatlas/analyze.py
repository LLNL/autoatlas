import nibabel as nib
import numpy as np
from autoatlas._utils import adjust_dims
import os

def overlap_coeff(atlas1,atlas2,mask=None,norm_type=None):
#    mask = np.bitwise_and(mask,atlas2!=3) #HACK: Should be removed

    if atlas1.ndim == 3:
        atlas1 = atlas1[np.newaxis]

    if atlas2.ndim == 3:
        atlas2 = atlas2[np.newaxis]

    assert atlas1.ndim==4,'Number of dimensions of atlas1 must be 4'
    assert atlas2.ndim==4,'Number of dimensions of atlas2 must be 4'
    assert atlas1.shape == atlas2.shape
    assert mask.dtype==bool

    if norm_type is None:
        norm_func = lambda a,b: 1.0
    elif norm_type == 'min':
        norm_func = lambda a,b: min(a,b)
    elif norm_type == 'max':
        norm_func = lambda a,b: max(a,b)
    elif norm_type == 'sum':
        norm_func = lambda a,b: a+b
    else:
        raise ValueError('norm_type must be either None, min, or max.')

    atlas1 = np.round(atlas1).astype(int)
    atlas2 = np.round(atlas2).astype(int)
    volsh = atlas1.shape

    a1_min,a1_max = int(atlas1.min()),int(atlas1.max())
    a2_min,a2_max = int(atlas2.min()),int(atlas2.max())

    overlap = np.zeros((volsh[0],a1_max-a1_min+1,a2_max-a2_min+1),dtype=np.float32,order='C')
    for i in range(volsh[0]):
        for idx1,lab1 in enumerate(range(a1_min,a1_max+1,1)):
            for idx2,lab2 in enumerate(range(a2_min,a2_max+1,1)):
                temp1 = np.bitwise_and(mask,atlas1[i]==lab1)
                temp2 = np.bitwise_and(mask,atlas2[i]==lab2)
                overlap[i,idx1,idx2] = np.sum(np.bitwise_and(temp1,temp2))  
                den = norm_func(np.sum(temp1),np.sum(temp2))
                if den > 0: 
                    overlap[i,idx1,idx2] /= den
    return overlap

