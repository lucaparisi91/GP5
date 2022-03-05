from fileinput import filename
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
    def iterations(self):
        return self._getIterations(self.files)
    
    @property
    def times(self):
        return self._getTimes(self.iterations)

    def _getIterations(self,files):
        iterations=[]
        for file in files:
            match=re.match(".*_([0-9]+).hdf5$",file )
            if match is not None:
                iteration=int( match[1] ) * int(self.settings.output.nIterations)
                iterations.append(iteration)
        return np.array(iterations)

    def _getTimes(self,iterations):
        return iterations*float(self.settings.evolution.timeStep)
    

    def collect(self):
        files=self.files
        iterations=self.iterations        
        times=pd.DataFrame({"times":self.times})
        times.index=iterations

        estimates=[]
        for i,file in tqdm.tqdm( zip(iterations,files) ):
            try:
                y=field.load(file)
            except OSError as e:
                print( str(e) )
            else:
                estimate=[]
                for ob in self.observables:
                    est=ob(y,key=i)
                    if est is not None:
                        estimate.append(est)
                if len(estimate) != 0:
                    estimates.append(pd.concat(estimate,axis=1) )
        
        if len(estimates) != 0:
            estimates=pd.concat(estimates)
            return pd.merge(times,estimates,left_index=True,right_index=True).sort_values(by="times")



class width:
    def __init__(self,settings,component=0,label="width"):
        self.settings=settings
        X,Y,Z=self.settings.discretization.grid
        self.R2=X**2 + Y**2 + Z**2
        self.deltaV=self.settings.discretization.cellVolume
        self.component=component
        self.label=label

    def __call__(self,field,key=0):
        
        res=self.deltaV*np.sum(np.abs(field[:,:,:,self.component])**2 * self.R2 )
        return pd.DataFrame({ self.label : [res]} , index=[key])



class maxDensity:
    def __init__(self,settings,component=0,label="width"):
        self.settings=settings
        self.component=component
        self.label=label

    def __call__(self,field,key=0):
        
        res=np.max(np.abs(field[:,:,:,self.component])**2 )
        return pd.DataFrame({ self.label : [res]} , index=[key])


class centerOfMass:
    def __init__(self,settings,component=0,label="cm"):
        self.settings=settings
        self.X,self.Y,self.Z=self.settings.discretization.grid
        self.deltaV=self.settings.discretization.cellVolume
        self.component=component
        self.label=label

    def __call__(self,psi,key=0):
        field2=np.abs(psi[:,:,:,self.component])**2
        N=np.sum(field2)
        Xm=np.sum( field2 * self.X )/N
        Ym=np.sum( field2 * self.Y )/N
        Zm=np.sum( field2 * self.Z )/N

        return pd.DataFrame({ "{}X".format(self.label) : [Xm] , "{}Y".format(self.label) : [Ym] , "{}Z".format(self.label) : [Zm]  } , index=[key])
        
    

class netCDFConverter:
    def __init__(self,outdir="outputVis"):
        self.outdir=outdir
    
    def __call__(self,psi, key):
        if ( not os.path.exists(self.outdir)):
            os.makedirs(self.outdir)
        filename=os.path.join( self.outdir, "psi{:d}.nc".format(key))
        field.saveNetCDF(psi,filename)

