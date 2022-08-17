from platform import processor
from gp import gp
import unittest
import numpy as np
from mpi4py import MPI


class testField(unittest.TestCase):

    def test_geo(self):
        geo=gp.geometry( ( 100, 200, 50 ) )
        assert( geo.shape == [100,200,50]  )
        assert( geo.left == [-1,-1,-1 ]  )
        assert( geo.right == [1,1,1] )
        assert(geo.spaceStep[0] == 2/100)
        assert(geo.spaceStep[1] == 2/200)
        assert(geo.spaceStep[2] == 2/50)
    

    def test_field(self):
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        geo=gp.geometry( ( 100, 200, 50 ) )

        if size == 1:
            psi=gp.field(geo,nComponents=1,processorGrid=[1,1,1])

            assert(psi.localShape ==    [100,200,50] )

            for d in range(3):
                assert( psi.array.shape[d] == psi.localShape[d] )
        
            X,Y,Z=psi.grid()

            self.assertAlmostEqual(np.min(X) , geo.left[0] + geo.spaceStep[0]/2 )
            self.assertAlmostEqual(np.max(X) , geo.right[0] - geo.spaceStep[0]/2 )

            self.assertAlmostEqual( X[1,0,0,0] - X[0,0,0,0] , geo.spaceStep[0] )

            psi.array=(X**2 + Y**2 + Z**2)

            self.assertAlmostEqual( np.sum(psi.array - (X**2 + Y**2 + Z**2 ) ),0 )
        

        if size == 2:
            psi=gp.field(geo,nComponents=1,processorGrid=[2,1,1])

            assert(psi.localShape ==    [50,200,50] )

              
            for d in range(3):
                assert( psi.array.shape[d] == psi.localShape[d] )

            print(psi.offset)


            X,Y,Z=psi.grid()

            self.assertAlmostEqual(np.min(X) , geo.left[0] + geo.spaceStep[0] *(1/2 + psi.offset[0] )  )
            self.assertAlmostEqual(np.max(X) , geo.left[0] + geo.spaceStep[0]*(-1/2 + psi.localShape[0] + psi.offset[0]  ) )

            self.assertAlmostEqual( X[1,0,0,0] - X[0,0,0,0] , geo.spaceStep[0] )

            psi.array=(X**2 + Y**2 + Z**2)

            self.assertAlmostEqual( np.sum(psi.array - (X**2 + Y**2 + Z**2 ) ),0 )


    def test_LHY(self):
        geo=gp.geometry( ( 100, 200, 50 ) )

        comm = MPI.COMM_WORLD
        size = comm.Get_size()


        if size==1:
            psi=gp.field(geo,nComponents=1,processorGrid=[1,1,1])
        else:
            if size==2:
                psi=gp.field(geo,nComponents=1,processorGrid=[2,1,1])
            else:
                return


        model=gp.LHY(psi)

        assert( model.nComponents == 1 )


    def test_LHY_apply(self):
        geo=gp.geometry( ( 100, 100, 100 ),left=[-4,-4,-4] , right=[4,4,4] )

        comm = MPI.COMM_WORLD
        size = comm.Get_size()


        
        psi1=gp.field(geo,nComponents=1,processorGrid=[size,1,1])
        psi2=gp.field(geo,nComponents=1,processorGrid=[size,1,1])

        model=gp.LHY(psi1)

        X,Y,Z =psi1.grid()
        r=np.sqrt(X**2 + Y**2 + Z**2)
       
        assert( model.nComponents == 1 )
        psi1.array=np.exp(-r**2) + 1j*0

        model.apply(psi1,psi2,t=0)

        #plt.plot(r.flatten(),np.real(psi1.array).flatten(),"or")
        #plt.plot(r.flatten(),np.real(psi2.array).flatten(),"og")

        alpha=1
        psi_expected=alpha*(3 - 2*alpha*r**2)*np.real(psi1.array) + ( -3* np.real(psi1.array)**2 + 5/2 * np.real(psi1.array)**3 )*np.real(psi1.array)

        assert( np.max(np.abs(psi_expected - np.real(psi2.array)) ) <1e-4 )
        
        #plt.show()

    def test_renormalize(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()


        geo=gp.geometry( ( 50, 50, 50 ) , left=[-5,-5,-5] , right=[5,5,5] )
        psi=gp.field(geo,nComponents=1,processorGrid=[size,1,1])
        
        X,Y,Z = psi.grid()
        r=np.sqrt(X**2 + Y**2 + Z**2)
        psi0=np.exp(-r**2)

        psi.array=psi0

        psi.normalize(1,component=0)



        self.assertAlmostEqual( psi.norm(0) ,1)

        psi.array=psi0
        psi.normalize([1])
        N = psi.norm()
        assert(len(N) == 1)
        self.assertAlmostEqual(N[0] , 1 )    

    def test_stepper(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()


        geo=gp.geometry( ( 50, 50, 50 ) , left=[-32,-32,-32] , right=[32,32,32] )
        psi=gp.field(geo,nComponents=1,processorGrid=[size,1,1])
        X,Y,Z = psi.grid()
        r=np.sqrt(X**2 + Y**2 + Z**2)
        psi0=np.exp(-r**2/(2*5**2))


        psi.array=psi0

        N=370
        psi.normalize([N])
        psi0=psi.array


        model=gp.LHY(psi)

        stepper=gp.stepper(timeStep=0.1*geo.spaceStep[0]**2,name="RK4",model=model)


        #stepper.setRenormalization([N])


        nBlocks=4
        stepsPerBlock=100
        maxDensityOp = gp.maxDensity()
        if comm.Get_rank() == 0:
            print("LHY droplet test run...")
        for it in (range(nBlocks)):
            for itt in range(stepsPerBlock):
                psi.normalize([N])
                stepper.advance(psi,nSteps=1)
            
            mu=0.5* np.log(N/psi.norm(0))/(stepper.timeStep)
            psi.normalize([N])
            maxDensity=maxDensityOp(psi,root=0)
            
            if comm.Get_rank() == 0:
                print("time: {} , mu: {} , d0: {} ".format(stepper.time,mu,maxDensity) )

            #psi.save("out/{:d}.hdf5".format(it))
        
        #plt.plot(r.flatten(),np.abs(psi.array).flatten() , "or" )

        #plt.show()
    
    
    def test_io(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        
        geo=gp.geometry( ( 50, 50, 50 ) , left=[-32,-32,-32] , right=[32,32,32] )
        psi=gp.field(geo,nComponents=1,processorGrid=[size,1,1])
        X,Y,Z = psi.grid()
        r=np.sqrt(X**2 + Y**2 + Z**2)
        psi0=np.exp(-r**2/(2*5**2))
        psi.array=psi0

        psi.save("test.hdf5")

        psi.array=psi0*0

        psi.load("test.hdf5")

        self.assertAlmostEqual(np.sum(np.abs(psi.array - psi0)) , 0 )

        
        
if __name__ == '__main__':
    unittest.main()