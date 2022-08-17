



class geometry:
    def __init__(self, size, left=[-1,-1,-1],right=[1,1,1]  ):
        """
        size: array of length 3, global mesh
        left: array of length 3,left-most edge position in real space
        right: array of length 3,right-most edge position in real space
        """
        self._size=size
        self._left=left
        self._right=right
        

# map to C++ implementation. Contains the MPI communicator and the local array



class field:
    def __init__(self,geo,nComponents=1,processorGrid=[1,1,1] ):
        self._geo=geo

class euleroStepper:
    def __init__(self,timeStep,imaginaryTime=True):
        self._timeStep=timeStep

class RK4Stepper:
    def __init__(self,timeStep,imaginaryTime=False):
        self._timeStep=timeStep

class LHY:
    def __init__(self):
        self.nComponents=1

class simulation:
    def __init__(self,geometry,model, stepper):
        self._geometry=geometry
        self._model=model
        self._stepper=stepper
        self._time=0

    def propagate( field , nSteps):
        """
        field: numpy array of rank 4 
        """
        pass