from sys import settrace
from time import time
import numpy as np
import scipy as sp
import json as j
from gp import io
import os
import re
import tqdm
import pandas as pd
from gp import field


class analysis:
    def __init__( self,settings, runFolder=".", observables=[] ):
        self.runFolder=runFolder
        self.settings=settings

        self.outputFolder=os.path.join(self.runFolder,self.settings.output.folder)
        self.observables=observables

    @property
    def files(self):
        _files=os.listdir(self.outputFolder)
        _files= [os.path.join(self.outputFolder,file) for file in _files]
        return _files
    


    @property
    def times(self):
        return self._getTimes(self.files)


    def _getTimes(self,files):
        times=[]
        for file in files:
            match=re.match(".*_([0-9]+).hdf5$",file )
            if match is not None:
                iteration=int( match[1] ) * int(self.settings.output.nIterations)
                t=float(iteration*float(self.settings.evolution.timeStep) )
            times.append(t)
        

        return np.array(times)


    def collect(self):
        files=self.files
        times=self._getTimes(files)
        times=pd.DataFrame({"times":times})

        estimates=[]
        for i,file in tqdm.tqdm(enumerate(files) ):
            y=field.load(file)
            estimate=[]
            for ob in self.observables:
                estimate.append(ob(y,key=i))
            estimates.append(pd.concat(estimate,axis=1) )
        
        estimates=pd.concat(estimates)

        return pd.merge(times,estimates,left_index=True,right_index=True).sort_values(by="times")


class width:
    def __init__(self,settings):
        self.settings=settings
        X,Y,Z=self.settings.discretization.grid
        self.R2=X**2 + Y**2 + Z**2
        self.deltaV=self.settings.discretization.cellVolume
        
    def __call__(self,field,key=0):
        res=self.deltaV*np.sum(np.abs(field)**2 * self.R2 )
        return pd.DataFrame({ "width" : [res]} , index=[key])



class centerOfMass:
    def __init__(self,settings):
        self.settings=settings
        self.X,self.Y,self.Z=self.settings.discretization.grid
        self.deltaV=self.settings.discretization.cellVolume

    def __call__(self,psi,key=0):
        field2=np.abs(psi)**2
        Xm=self.deltaV*np.sum( field2 * self.X )
        Ym=self.deltaV*np.sum( field2 * self.Y )
        Zm=self.deltaV*np.sum( field2 * self.Z )

        return pd.DataFrame({ "cmX" : [Xm] , "cmY" : [Ym] , "cmZ" : [Zm]  } , index=[key])
        
    


