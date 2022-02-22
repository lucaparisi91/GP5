import numpy as np
import h5py


def load( file ):
    f = h5py.File(file, 'r')
    field=np.array(f["field"][:,:,:,:])
    if len(field.shape) == 5:
        field=np.array(field[:,:,:,:,0]) + 1j * np.array(field[:,:,:,:,1])
    field=np.transpose(field)
    if (field.shape[3]==1):
        field=field.reshape(np.array(field.shape)[:-1])

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


def generateGrid( meshShape, domain ):
    D=len(meshShape)
    lBox = [ domain[d][1] - domain[d][0]  for d in  range(D) ]
    deltax = [ lBox[d]/meshShape[d]  for d in  range(D) ]
    axes= [ (np.arange(0,meshShape[d]) + 0.5)* deltax[d] + domain[d][0]  for d in range(D)  ]
    X,Y,Z=np.meshgrid(*axes,indexing="ij")
    return X,Y,Z


