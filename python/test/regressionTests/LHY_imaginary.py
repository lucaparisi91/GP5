from gp import gp
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()

geo=gp.geometry( ( 30, 30, 30 ) , left=[-10,-10,-10] , right=[10,10,10] )
psi=gp.field(geo,nComponents=1,processorGrid=[size,1,1])
psi2=gp.field(geo,nComponents=1,processorGrid=[size,1,1])

X,Y,Z = psi.grid()
r=np.sqrt(X**2 + Y**2 + Z**2)
psi0=np.exp(-r**2/(2*5**2))

psi.array=psi0

N=370
psi.normalize([N])
psi0=psi.array

model=gp.LHY(psi)

stepper=gp.stepper(timeStep=0.01*geo.spaceStep[0]**2,name="RK4",model=model)

#stepper.setRenormalization([N])

tolerance=1e-9
stepsPerBlock=1000
maxDensityOp = gp.maxDensity()


oldMu=0
mu=1.0e+18
it=0


if comm.Get_rank() == 0:
    print("LHY droplet test run...")

while abs(mu - oldMu) > tolerance: 
    it+=1
    for itt in range(stepsPerBlock):
        psi.normalize([N])
        stepper.advance(psi,nSteps=1)
    
    oldMu=mu
    mu=0.5* np.log(N/psi.norm(0))/(stepper.timeStep)
    psi.normalize([N])
    maxDensity=maxDensityOp(psi,root=0)
    model.apply(psi,psi2 ,t=0 )

    if comm.Get_rank() == 0:
        print("time: {} , mu: {} , d0: {} , conv: {} ".format(stepper.time,mu,maxDensity,abs(mu - oldMu)) )

    psi.save("out/{:d}.hdf5".format(it))



#plt.plot(r.flatten(),np.abs(psi.array).flatten() , "or" )
#plt.show()


