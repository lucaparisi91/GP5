import gpCpp
import mpi4py
from mpi4py  import  MPI
import numpy as np
import os
from pathlib import Path
from netCDF4 import Dataset
import re


class geometry:

    def __init__( self,shape,left=[-1,-1,-1],right=[1,1,1]):
        self._geometryCpp=gpCpp.geometry( left, right,shape)
    
    @property
    def left(self):
        return self._geometryCpp.getLeft()
    
    @property
    def right(self):
        return self._geometryCpp.getRight()
    @property
    def shape(self):
        return self._geometryCpp.getShape()
    @property
    def spaceStep(self):
        return [ (self.right[d] - self.left[d] )/ self.shape[d] for d in range(len(self.shape)) ]
    


class field:
    def __init__(self,
    geo,
    nComponents=1 ,
    processorGrid=[1,1,1],
    fftOrdering=[2,0,1]
    ):
        self._nComponents=nComponents
        self._comm=MPI.COMM_WORLD
        self._geometry=geometry
        self._nComponents=nComponents
        self._geometry=geo
        self._fieldCpp=gpCpp.field( geo._geometryCpp, nComponents , processorGrid , fftOrdering)
    
    @property
    def localShape(self):
        return self._fieldCpp.getLocalShape()
    
    @property
    def array(self):
        return self._fieldCpp.getTensor()
    
    @array.setter
    def array(self,x):
        self._fieldCpp.setTensor(x)
    @property
    def nComponents(self):
        return self._nComponents


    def norm(self, component=None):

        if component is not None:
            return self._fieldCpp.getNorm(component)
        else:
            return [ self.norm(c) for c in range(self.nComponents) ]



    def normalize(self, N, component=None ):
        if component is not None:
            self._fieldCpp.normalize(component,N)
        else:
            for Nc,c in zip(N, range( self.nComponents) ):
                self.normalize(Nc,component=c)
    
    @property
    def offset(self):
        return self._fieldCpp.getOffset()
    
    def grid( self ):
        meshShape=self.localShape 
        left=self._geometry.left
        right=self._geometry.right
        D=len(meshShape)
        
        deltax = self._geometry.spaceStep
        axes= [ (np.arange(0,meshShape[d]) + 0.5)* deltax[d] + left[d] + self.offset[d] * deltax[d] for d in range(D)  ]


        axes.append( [1.0 for c in range(self._nComponents)]  )
        
        grids=np.meshgrid(*axes,indexing="ij")
        X=grids[0]
        Y=grids[1]
        Z=grids[2]

        return X,Y,Z

    def save(self,filename,format=None):
        if format is None:
            if re.match(".*\.nc$",filename) is not None:
                format="netCDF4"
            else:
                if re.match(".*\.hdf5$",filename) is not None:
                    format="hdf5"
                else:
                    if re.match(".*\.h5$",filename) is not None:
                        format="hdf5"                 

        if format is None :
            return RuntimeError("Could not guess format.")
        else:
            if format == "netCDF4":
                self.saveNetCDF( filename )
            else:
                if format == "hdf5":
                    self.saveHDF5( filename )
                else:
                    return RuntimeError("Unkown format {}".format(filename) )
        

        
        

    def saveHDF5(self,filename):
        dir=Path(os.path.dirname(filename) )
        dir.mkdir(parents=True,exist_ok=True)
        
        self._fieldCpp.save(filename)

    def saveNetCDF(self,filename):
        '''
        Saves the density (rho) and the phase (phi) of the field psi on a file.
        '''
        dir=Path(os.path.dirname(filename) )
        dir.mkdir(parents=True,exist_ok=True)

        psi=self.array
        rho=np.abs(psi)**2
        rootgrp = Dataset(filename, "w")
        xDim=rootgrp.createDimension("X", psi.shape[0])
        yDim=rootgrp.createDimension("Y", psi.shape[1])
        zDim=rootgrp.createDimension("Z", psi.shape[2])

        for c in range(psi.shape[3]):
            rhoVar= rootgrp.createVariable("rho{:d}".format(c),"f8",("X","Y","Z"))
            rhoVar[:,:,:]=rho[:,:,:,c]
            rhoVar.units="K"

        phi=np.angle(psi)
        for c in range(psi.shape[3]):
            
            phiVar= rootgrp.createVariable("phi{:d}".format(c),"f8",("X","Y","Z"))
            phiVar[:,:,:]=phi[:,:,:,c]
            phiVar.units="K"
        rootgrp.close()

    
    def load(self,filename):
        self._fieldCpp.load( filename)
    
   


class LHY:
    def __init__(self,psi):
        if psi.nComponents != 1:
            raise  ValueError("Field should have one component")
        self._funcCpp=gpCpp.LHY(psi._fieldCpp)

    def apply(self,psi1,psi2,t):
        self._funcCpp.apply(psi1._fieldCpp,psi2._fieldCpp,t)

    @property
    def nComponents(self ):
        return self._funcCpp.nComponents()


class stepper:
    def __init__(self,model, timeStep,name="RK4",imaginary=True,renormalize=False):
        """
        name: str, allowed : RK4, Eulero
        """
        self._timeStep=timeStep
        self._name=name
        self._imaginary=imaginary
        self._stepperCpp=gpCpp.timeStepper( timeStep,name,imaginary)
        self._model=model
        self._stepperCpp.setFunctional(model._funcCpp)
        self._renormalize=False
        self._stepperCpp.unsetRenormalization()
    

    def setRenormalization(self,N):
        self._stepperCpp.setRenormalization(N)


    def advance( self,psi, nSteps=1):
                self._stepperCpp.advance(psi._fieldCpp,nSteps)
    @property
    def time(self):
        return self._stepperCpp.getTime()
    @property
    def timeStep(self):
        return self._timeStep


class maxDensity:
    def __init__(self):
        pass 
    def __call__(self,psi,root=0):
        comm=MPI.COMM_WORLD
        maxDensityLocal=np.array([np.max(np.abs(psi.array))**2])
        maxDensity=np.array([maxDensityLocal])

        comm.Reduce( [maxDensityLocal, MPI.DOUBLE],[maxDensity, MPI.DOUBLE] , op=MPI.MAX,root=root )


        if comm.Get_rank() == root:
            return maxDensity[0]
        




