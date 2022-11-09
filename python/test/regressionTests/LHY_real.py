from gp import gp
import numpy as np
from mpi4py import MPI
from matplotlib import pylab as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()

geo=gp.geometry( ( 30, 30, 30 ) , left=[-10,-10,-10] , right=[10,10,10] )
psi=gp.field(geo,nComponents=1,processorGrid=[size,1,1])
psi2=gp.field(geo,nComponents=1,processorGrid=[size,1,1])

psi.load("gs.hdf5")
N=370
#psi.normalize([N])
psi0=psi.array

X,Y,Z = psi.grid()
r=np.sqrt(X**2 + Y**2 + Z**2)
model=gp.LHY(psi)
model.apply(psi,psi2,t=0)


stepper=gp.stepper(timeStep=0.01*geo.spaceStep[0]**2,name="RK4",model=model,imaginary=False)

#stepper.setRenormalization([N])


#plt.plot( r.flatten(), (np.abs(psi.array)**2).flatten() ,"or"  )
#plt.plot( r.flatten(), 1/(0.383729)**2 *0.98* (np.abs(psi2.array)**2).flatten() ,"og"  )


maxTime=100
stepsPerBlock=100
maxDensityOp = gp.maxDensity()
it=0
if comm.Get_rank() == 0:
    print("LHY droplet test run...")

while (stepper.time < maxTime):
    it+=1
    for itt in range(stepsPerBlock):
        stepper.advance(psi,nSteps=1)

    maxDensity=maxDensityOp(psi,root=0)
    
    if comm.Get_rank() == 0:
        print("time: {} , d0: {} , N : {} ".format(stepper.time,maxDensity,psi.norm(0)) )

    psi.save("real_out/{:d}.hdf5".format(it))

plt.plot( r.flatten(), (np.abs(psi.array)**2).flatten() ,"og"  )
#plt.plot( r.flatten(), (np.angle(psi.array) ).flatten() ,"og"  )
#plt.plot( r.flatten(), (np.angle(psi.array) ).flatten() ,"og"  )


#plt.plot( r.flatten(), (np.real(psi.array)**2).flatten() ,"ob"  )
#plt.plot( r.flatten(), (np.imag(psi.array)**2).flatten() ,"or" )

plt.show()


