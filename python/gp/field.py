import h5py
import numpy as np
from netCDF4 import Dataset

def load( file ):
    f = h5py.File(file, 'r')
    field=np.array(f["field"][:,:,:,:])
    if len(field.shape) == 5:
        field=np.array(field[:,:,:,:,0]) + 1j * np.array(field[:,:,:,:,1])
    field=np.transpose(field)
    #if (field.shape[3]==1):
    #    field=field.reshape(np.array(field.shape)[:-1])

    return field

def save(file,y):
    y=np.array(y)
    if len(y.shape)==3:
        y=y.reshape( y.shape + (1,)  )
    

    with h5py.File(file, "w") as f:
        yt=y.transpose()
        newShape= yt.shape + (2,)
        dtype=h5py.h5t.array_create(h5py.h5t.NATIVE_DOUBLE, (2,))
        f.create_dataset("field", yt.shape , dtype=dtype)
        raw=np.zeros( newShape )
        raw[:,:,:,:,0]=np.real(yt)
        raw[:,:,:,:,1]=np.imag(yt)
        f["field"][:,:,:,:]=raw


def saveNetCDF(psi,filename):
    '''
    Saves the density (rho) and the phase (phi) of the field psi on a file.
    '''

    rho=np.abs(psi)**2
    rootgrp = Dataset(filename, "w")
    xDim=rootgrp.createDimension("X", psi.shape[0])
    yDim=rootgrp.createDimension("Y", psi.shape[1])
    zDim=rootgrp.createDimension("Z", psi.shape[2])

    for c in range(psi.shape[3]):
        rhoVar= rootgrp.createVariable("rho{:d}".format(c),"f8",("Z","Y","X"))
        rhoVar[:,:,:]=rho[:,:,:,c]
        rhoVar.units="K"
    
    phi=np.angle(psi)
    for c in range(psi.shape[3]):
        
        phiVar= rootgrp.createVariable("phi{:d}".format(c),"f8",("Z","Y","X"))
        phiVar[:,:,:]=phi[:,:,:,c]
        phiVar.units="K"
    rootgrp.close()



