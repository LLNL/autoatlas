import numpy as np

def adjust_dims(vol,dims):
    dims = np.array(dims,dtype=int)
    iB = (np.array(vol.shape,dtype=int)-dims)//2
    iE = dims+iB
    if iB[0]>=0: 
        vol = vol[iB[0]:iE[0]]
    if iB[1]>=0: 
        vol = vol[:,iB[1]:iE[1]]
    if iB[2]>=0: 
        vol = vol[:,:,iB[2]:iE[2]]
    sh = np.array(vol.shape,dtype=int)
    wid = dims-sh
    wid[wid<0] = 0
    return np.pad(vol,((0,wid[0]),(0,wid[1]),(0,wid[2])),mode='constant')

