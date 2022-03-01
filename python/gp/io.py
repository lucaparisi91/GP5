import numpy as np
import yaml
from gp import field

def generateGrid( meshShape, domain ):
    D=len(meshShape)
    lBox = [ domain[d][1] - domain[d][0]  for d in  range(D) ]
    deltax = [ lBox[d]/meshShape[d]  for d in  range(D) ]
    axes= [ (np.arange(0,meshShape[d]) + 0.5)* deltax[d] + domain[d][0]  for d in range(D)  ]
    X,Y,Z=np.meshgrid(*axes,indexing="ij")
    return X,Y,Z


class discretization:
    def __init__(self,shape,domain):
        self.shape=shape
        self.domain=domain
    
    @property
    def domain(self):
        return self._domain
    @property
    def left(self):
        return self._left
    @property
    def right(self):
        return self._right
    
    @property
    def grid(self):
        return generateGrid(self.shape,domain=self.domain)

    @property
    def grid(self):
        return generateGrid(self.shape,domain=self.domain)

    @property
    def cellWidth(self):
        return self._cellWidth

    @property
    def cellVolume(self):
        return np.prod(self._cellWidth)
    
    
    @domain.setter
    def domain(self,domain):
        self._domain=domain
        self._left=[ xRange[0]  for xRange in self._domain]
        self._right=[ xRange[1]  for xRange in self._domain]
        self._cellWidth = [ (self._right[d] - self._left[d] )/self.shape[d]   for d in range(len(self.shape))]


class output:
    def __init__(self, nIterations = 100, folder = "output"):
        self.nIterations=nIterations
        self.folder=folder
    
    def load(self,settings):
        self.folder=settings["folder"]
        self.nIterations=settings["nIterations"]


class evolution:
    def __init__( self, timeStep=0, algorithm="eulero",imaginaryTime=True):
        self.timeStep=timeStep
        self.algorithm=algorithm
        self.imaginaryTime=imaginaryTime
    

    def load(self,settings):
        self.timeStep=settings["timeStep"]
        self.algorithm=settings["stepper"]
        self.imaginaryTime=settings["imaginaryTime"]


class simulation:
    def __init__(self, discretization=None,output=output() ,evolution=None):
        self.discretization=discretization
        self.output=output
        self.evolution=evolution


def load( yamlFile ):
    with open( yamlFile) as f:
        settings=yaml.safe_load(f)
    shape= settings["mesh"]["shape"]
    left= settings["domain"]["left"]
    right= settings["domain"]["right"]
    domain=[ (left[d] , right[d] )       for d in range(len(shape))  ]

    dis = discretization(shape,domain )

    simOutput=output()
    simOutput.load(settings["output"])

    simEvolution=evolution()
    simEvolution.load(settings["evolution"])

    return simulation(discretization=dis,output=simOutput,evolution=simEvolution)

