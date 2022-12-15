


#include <petscksp.h>
#include "petscdmda.h"
#include <petscviewerhdf5.h>
#include <iostream>
#include <petscsnes.h>
#include "model.h"
#include "petscsys.h" 

class sphericalIntegrator;

struct petscWave
{
   PetscReal right[DIMENSIONS];
   PetscReal left[DIMENSIONS];
   PetscReal spaceStep[DIMENSIONS];
   Mat J0;
   PetscReal mu;
   Vec diagonalTMP;
   PetscScalar g;
   sphericalIntegrator* integrator;
};
PetscErrorCode fillSpacePositions1D(DM da, Vec X, const PetscReal* left, const PetscReal* right )
{
   PetscScalar * _X;
   DMDALocalInfo info;
   PetscReal spaceStep[1];
   DMDAGetLocalInfo(da, &info);

   spaceStep[0]=(right[0]-left[0] )/info.mx;
   
   PetscCall( DMDAVecGetArray(da, X , &_X ) );

   
   for(int i=info.xs;i<info.xs + info.xm ; i++)
      {

         _X[i] = left[0] + ( i+0.5 )*spaceStep[0];
      }   

   PetscCall(DMDAVecRestoreArray(da,X,&_X) );
   PetscCall(VecAssemblyBegin(X));
   PetscCall(VecAssemblyEnd(X));   

   return 0;

}


class sphericalIntegrator
{
   public:

   sphericalIntegrator( DM da, PetscReal left, PetscReal right) : _da(da) {
      DMCreateGlobalVector(da,&_r2) ;
      DMCreateGlobalVector(da,&_tmp) ;
      fillSpacePositions1D(da,_r2,&left,&right);
      VecPointwiseMult(_r2,_r2,_r2);

      DMDALocalInfo info;
      DMDAGetLocalInfo(da, &info);

      _dx=(right-left )/info.mx;

   }


   auto integrate(Vec psi)
   {
      PetscScalar integ;
      VecPointwiseMult(_tmp,psi,_r2);
      VecSum(_tmp, &integ );
      return integ*4*M_PI*_dx;
   }

   auto getSpaceStep(){return _dx; }
   

   private:

   PetscReal _dx;
   Vec _r2;
   Vec _tmp;
   DM _da;

};


PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void * ctx )
{
   PetscInt left[DIMENSIONS];
   PetscInt right[DIMENSIONS];
   PetscInt shape[DIMENSIONS];

   
   auto Jctx = ( petscWave *) ctx;

   DM da;
   PetscCall(SNESGetDM(snes, &da));

   std::cout << "J: " << Jctx->mu << std::endl;

   MatCopy(Jctx->J0, jac , SAME_NONZERO_PATTERN );   

   VecSet(Jctx->diagonalTMP, -Jctx->mu);

   // add the nonlinear term to the diagonal
   PetscScalar *** _x;
   PetscScalar *** _d;
   auto g = Jctx->g;

   PetscCall(DMDAVecGetArrayRead(da,x, &_x) );
   PetscCall(DMDAVecGetArray(da,Jctx->diagonalTMP, &_d) );
    DMDAGetCorners( da, &(left[0]), &(left[1]), &(left[2]), &(shape[0]), &(shape[1]),  &(shape[2]) );

     for(int i=left[0];i<left[0] + shape[0];i++)
      for(int j=left[1];j<left[1] + shape[1];j++)
         for(int k=left[2];k<left[2] + shape[2];k++)
         {
            _d[k][j][i] = 3*g*_x[k][j][i] * _x[k][j][i]  ;
         }
   

   PetscCall(DMDAVecRestoreArrayRead(da,x, &_x) );
   PetscCall(DMDAVecRestoreArray(da,Jctx->diagonalTMP, &_d) );




   MatDiagonalSet(jac,Jctx->diagonalTMP,ADD_VALUES);


   
   MatCopy(Jctx->J0, B , SAME_NONZERO_PATTERN );   

   return 0;
}

PetscErrorCode FormJacobianSpherical(SNES snes, Vec x, Mat jac, Mat B, void * ctx )
{
   PetscInt left[DIMENSIONS];
   PetscInt right[DIMENSIONS];
   PetscInt shape[DIMENSIONS];
   DMDALocalInfo info;
   DM da;
   SNESGetDM(snes,&da);
   DMDAGetLocalInfo(da, &info);
   
   auto Jctx = ( petscWave *) ctx;

   //std::cout << "J: " << Jctx->mu << std::endl;

   MatCopy(Jctx->J0, jac , SAME_NONZERO_PATTERN );   


   // add the nonlinear term to the diagonal
   PetscScalar * _x;
   PetscScalar * _d;
   auto g = Jctx->g;


   PetscCall(DMDAVecGetArrayRead(da,x, &_x) );
   PetscCall(DMDAVecGetArray(da,Jctx->diagonalTMP, &_d) );

   for(int i=info.xs;i<info.xs + info.xm;i++)
         {
            _d[i] = 3* g*_x[i] * _x[i] -Jctx->mu  ;
         }
   

   PetscCall(DMDAVecRestoreArrayRead(da,x, &_x) );
   PetscCall(DMDAVecRestoreArray(da,Jctx->diagonalTMP, &_d) );


   MatDiagonalSet(jac,Jctx->diagonalTMP,ADD_VALUES);

   
   MatCopy(Jctx->J0, B , SAME_NONZERO_PATTERN );   



   return 0;
}




PetscErrorCode FormFunction( SNES snes,Vec X, Vec Y , void * ctx )
{
   DM da;
   PetscCall(SNESGetDM(snes, &da));
   Vec x;

   petscWave * wave = (petscWave *)(ctx);

   auto g = wave->g;

   PetscCall( DMGetLocalVector(da,&x) );
   PetscCall( DMGlobalToLocalBegin(da, X, INSERT_VALUES, x));
   PetscCall( DMGlobalToLocalEnd(da, X, INSERT_VALUES, x) );

   PetscInt left[3];
   PetscInt shape[3];
   PetscReal spaceStep[3];
   PetscInt globalShape[3];

   PetscScalar *** _x;
   PetscScalar *** _y;

   DMDALocalInfo info;
   {
      DMDAGetLocalInfo(da, &info);
      globalShape[0]=info.mx;
      globalShape[1]=info.my;
      globalShape[2]=info.mz;
   }

   PetscCall(DMDAVecGetArrayRead(da,x, &_x) );
   PetscCall(DMDAVecGetArray(da,Y, &_y) );

   DMDAGetCorners( da, &(left[0]), &(left[1]), &(left[2]), &(shape[0]), &(shape[1]),  &(shape[2]) );

   PetscScalar spaceStepInverse2[DIMENSIONS];

   for(int d=0;d<DIMENSIONS;d++)
   {
      spaceStepInverse2[d]=1/(wave->spaceStep[d] * wave->spaceStep[d] );
   }

   for(int i=left[0];i<left[0] + shape[0];i++)
      for(int j=left[1];j<left[1] + shape[1];j++)
         for(int k=left[2];k<left[2] + shape[2];k++)
         {

            auto x = wave->left[0] + ( i+ 0.5 )*wave->spaceStep[0];
            auto y = wave->left[1] + ( j+0.5 )*wave->spaceStep[1];
            auto z = wave->left[2] + ( k+0.5 )*wave->spaceStep[2];


            _y[k][j][i]=  _x[k][j][i]* ( spaceStepInverse2[0] + spaceStepInverse2[1] + spaceStepInverse2[2] + 0.5*(x*x + y*y + z*z)  ) 
            - _x[k][j][i-1]* 0.5 * spaceStepInverse2[0] - _x[k][j][i+1]* 0.5 * spaceStepInverse2[0] 
            - _x[k][j+1][i]* 0.5 * spaceStepInverse2[1] - _x[k][j-1][i]* 0.5 * spaceStepInverse2[1] 
            - _x[k+1][j][i]* 0.5 * spaceStepInverse2[2] - _x[k-1][j][i]* 0.5 * spaceStepInverse2[2] 
            + g * (_x[k][j][i] * _x[k][j][i] * _x[k][j][i]  )
             ;

         }


   PetscCall(DMDAVecRestoreArrayRead(da,x, &_x) );
   PetscCall(DMRestoreLocalVector(da, &x) );
   
   PetscCall(DMDAVecRestoreArray(da, Y , &_y) );
   PetscScalar mu;
   PetscScalar waveNorm;

   
   VecDot(Y,X,&mu);
   VecNorm(X,NORM_2 , &waveNorm);

   mu=mu/(waveNorm*waveNorm);

   VecAXPY(Y,-mu,X);     
   wave->mu = mu;
   std::cout << mu << std::endl;

   return 0;

}

PetscErrorCode FormFunctionSpherical( SNES snes, Vec X, Vec Y , void * ctx )
{
   DM da;
   PetscCall(SNESGetDM(snes, &da));
   Vec x;

   petscWave * wave = (petscWave *)(ctx);

   auto g = wave->g;

   PetscCall( DMGetLocalVector(da,&x) );
   PetscCall( DMGlobalToLocalBegin(da, X, INSERT_VALUES, x));
   PetscCall( DMGlobalToLocalEnd(da, X, INSERT_VALUES, x) );

   PetscScalar * _x;
   PetscScalar * _y;

   DMDALocalInfo info;

   DMDAGetLocalInfo(da, &info);

   PetscCall(DMDAVecGetArrayRead(da,x, &_x) );
   PetscCall(DMDAVecGetArray(da, Y , &_y) );



   auto spaceStepInverse2=1/(wave->spaceStep[0] * wave->spaceStep[0] );

   _x[-1]=_x[0];
   _x[info.mx]=_x[info.mx-1];

   for(int i=info.xs;i<info.xs + info.xm ;i++ )
         {

            auto x = wave->left[0] + ( i+ 0.5 )*wave->spaceStep[0];

            _y[i]=  _x[i]* ( spaceStepInverse2 + 0.5*( x * x  ) ) 
            - _x[i-1]*0.5*( spaceStepInverse2 - 1./(wave->spaceStep[0] * x) )- _x[i+1]* 0.5 *( spaceStepInverse2  + 1./(wave->spaceStep[0] * x) )
            + g * (_x[i] * _x[i] * _x[i]  );
             ;
         }


   PetscCall(DMDAVecRestoreArrayRead(da,x, &_x) );
   PetscCall(DMRestoreLocalVector(da, &x) );
   
   PetscCall(DMDAVecRestoreArray(da, Y , &_y) );

   /* PetscScalar mu;
   PetscScalar waveNorm;

   VecPointwiseMult(wave->diagonalTMP,Y,X);

   mu=wave->integrator->integrate(wave->diagonalTMP);
   VecPointwiseMult(wave->diagonalTMP,X,X);
   waveNorm=wave->integrator->integrate(wave->diagonalTMP) ;

   mu=mu/waveNorm;
 */
   //wave->mu=mu;
   VecAXPY(Y,-wave->mu ,X);


   //std::cout << "mu: "<< mu << std::endl;
   //std::cout << "norm: " << waveNorm << std::endl;  

   return 0;

}

PetscErrorCode initializeGaussian3D(PetscReal alpha, DM da, Vec X, const PetscReal* left, const PetscReal* right )
{
   PetscScalar *** _X;
   DMDALocalInfo info;
   PetscReal spaceStep[3];
   DMDAGetLocalInfo(da, &info);

   spaceStep[0]=(right[0]-left[0] )/info.mx;
   spaceStep[1]=(right[1]-left[1] )/info.my;
   spaceStep[2]=(right[2]-left[2] )/info.mz;

   PetscCall( DMDAVecGetArray(da, X , &_X ) );

   for(int i=info.xs;i<info.xs + info.xm ; i++)
      for(int j=info.ys;j<info.ys + info.ym ; j++)
         for(int k=info.zs;i<info.zs + info.zm ; k++)
         {
            auto x = left[0] + ( i+0.5 )*spaceStep[0];
            auto y = left[1] + ( j+0.5 )*spaceStep[1];
            auto z = left[2] + ( k+0.5 )*spaceStep[2];

            auto r2 = x*x + y*y + z*z;

            _X[k][j][i]=exp(-r2* alpha );
         }


   PetscCall(DMDAVecRestoreArray(da,X,&_X) );
   PetscCall(VecAssemblyBegin(X));
   PetscCall(VecAssemblyEnd(X));   

   return 0;

}

PetscErrorCode initializeGaussian1D(PetscReal alpha, DM da, Vec X, const PetscReal* left, const PetscReal* right )
{
   PetscScalar * _X;
   DMDALocalInfo info;
   PetscReal spaceStep[1];
   DMDAGetLocalInfo(da, &info);

   spaceStep[0]=( right[0]-left[0] )/info.mx;

   PetscCall( DMDAVecGetArray(da, X , &_X ) );

   for(int i=info.xs;i<info.xs + info.xm ; i++)
         {
            auto x = left[0] + ( i+0.5 )*spaceStep[0];
            _X[i]=exp(-x*x* alpha );
         }


   PetscCall(DMDAVecRestoreArray(da,X,&_X) );
   PetscCall(VecAssemblyBegin(X));
   PetscCall(VecAssemblyEnd(X));   

   return 0;

}



void normalize3D( Vec *X, PetscReal N2, PetscReal* spaceStep)
{
   auto dV = spaceStep[0]*spaceStep[1]*spaceStep[2];

   PetscReal norm;
   VecNorm(*X,NORM_2,&norm);
   norm*=sqrt(dV);

   VecScale(*X, 1/norm );

}

void normalize1D( Vec *X, PetscReal N2, PetscReal* spaceStep, Vec r2)
{
   auto dV = spaceStep[0];
   PetscReal norm;
   VecNorm(*X,NORM_2,&norm);
   norm*=sqrt(dV);
   VecScale(*X, 1/norm );
}



PetscErrorCode  createDM3D(DM * da, PetscInt * shape)
{
   PetscCall( 
      DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, shape[0], shape[1], shape[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1 , 1, NULL, NULL, NULL, da)
   );
   PetscCall(DMSetFromOptions(*da));
   PetscCall(DMSetUp(*da));

   return 0;
}



int main(int argc, char **args)
{
   const int dimensions=1;


   PetscInt    shape[1] { 1000  };
   PetscMPIInt size;
   PetscReal left[1] { 0 };
   PetscReal right[1]{ 60  };
   PetscReal spaceStep[1];

   for(int d=0;d<dimensions;d++)
   {
      spaceStep[d]=(right[d]-left[d])/shape[d];
   }

   PetscCall(PetscInitialize(&argc, &args, (char *)0, NULL));
   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
   PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
   Vec X;
   DM da;
   Mat H;
  
   createDMSpherical(&da,shape);

   Vec density;
   // initialize starting vector
   PetscCall( DMCreateGlobalVector(da,&X) );
   PetscCall( DMCreateGlobalVector(da,&density) );

   initializeGaussian1D(1./(2 * 1), da, X,left,right);
   sphericalIntegrator integ(da,0,right[0] );
   VecPointwiseMult(density,X,X);
   auto norm = integ.integrate(density);
   VecScale(X,sqrt(1./norm));

   //VecScale(X, -1);
   //VecShift(X, 1);

   PetscViewer HDF5viewer;
   PetscViewerHDF5Open(PETSC_COMM_WORLD, "X.hdf5", FILE_MODE_WRITE, &HDF5viewer);

   PetscObjectSetName((PetscObject)X,"init" );
   VecView(X, HDF5viewer);

   PetscViewer ASCIIviewer;
   PetscViewerASCIIOpen(PETSC_COMM_WORLD, "M.txt", &ASCIIviewer);
   PetscViewerPushFormat(ASCIIviewer, PETSC_VIEWER_ASCII_DENSE);

   petscWave wave;
   for(int d=0;d<DIMENSIONS;d++)
   {
      wave.left[d]=left[d];
      wave.right[d]=right[d];
      wave.spaceStep[d]=spaceStep[d];
      wave.mu=200;
      wave.g=100;
      wave.integrator=&integ;
   }


   Vec Y;
   VecDuplicate(X, &(wave.diagonalTMP) );
   VecDuplicate(X, &(Y) );
   VecCopy(X,Y);

   SNES snes;
   PetscCall( SNESCreate( PETSC_COMM_WORLD, &snes));
   PetscCall( SNESSetDM(snes, da));

   std::cout << "set up J" << std::endl;
   Mat J;
   DMSetMatType(da, MATAIJ);
   DMCreateMatrix(da, &J);
   DMCreateMatrix(da, &(wave.J0) );

   setMatrixSpherical( da, J , left, right );
   
   MatCopy(J,wave.J0,DIFFERENT_NONZERO_PATTERN );
   //FormFunctionSpherical( snes, X, Y , &wave );


   //PetscObjectSetName((PetscObject)Y,"evaluation" );
   //VecView(Y, HDF5viewer);


   PetscCall( SNESSetFunction(snes,X,&FormFunctionSpherical, & wave) );
   PetscCall( SNESSetJacobian(snes, J , J ,&FormJacobianSpherical, & wave) );
   PetscCall(SNESSetFromOptions(snes)); 


   Vec deltaX;
   VecDuplicate(X, &(deltaX ) );
   VecCopy(X,Y);
   VecSet( deltaX, 0);

   KSP ksp;
   KSPCreate(PETSC_COMM_WORLD, &ksp);
   KSPSetFromOptions( ksp );
   
   PetscReal N = 1;

   Vec * oldPsi;
   Vec * newPsi;
   
   oldPsi=& Y;
   newPsi= & X;

   PetscReal dt = 0.05 * spaceStep[0]*spaceStep[0] ;
   

   PetscReal error=1e+9;

   while( error > 1e-2 )
   {
      for(int tt=0;tt<1000;tt++)
      {
         std::swap(oldPsi,newPsi);

         FormFunctionSpherical( snes, *oldPsi, *newPsi , &wave );
         FormJacobianSpherical(snes, *oldPsi , J, J, &wave  );
         KSPSetOperators(ksp,J,J);

                  
         FormFunctionSpherical( snes, *oldPsi, deltaX , &wave );
         
         //KSPSolve( ksp, *newPsi , deltaX );
         VecWAXPY(*newPsi,-1.0*dt,deltaX,*oldPsi);

         //auto waveNorm=wave.integrator->integrate(wave.diagonalTMP);
         //VecScale( *newPsi, sqrt(N*1./waveNorm) );

      }

      VecPointwiseMult(wave.diagonalTMP,*newPsi,*newPsi);
      auto waveNorm=wave.integrator->integrate(wave.diagonalTMP);

      FormFunctionSpherical( snes, *newPsi, deltaX , &wave );
      //VecPointwiseMult(wave.diagonalTMP,deltaX,*newPsi);
      //auto error=wave.integrator->integrate(wave.diagonalTMP);
      //error/=waveNorm;

      //VecScale( *newPsi, sqrt(N*1./waveNorm) );
      //VecWAXPY( deltaX,-1,*oldPsi,*newPsi );

      VecNorm(deltaX,NORM_2,&error);
      std::cout << "norm: " << waveNorm << std::endl;
      std::cout << "mu: " << wave.mu << std::endl;
      std::cout << "error: " << error << std::endl;

   } 


   if ( newPsi != &Y)
   {
      VecCopy(*newPsi,Y);
   }

   

   PetscCall(SNESSolve(snes, NULL, Y ) );

   PetscObjectSetName((PetscObject)Y,"solution" );
   VecView(Y, HDF5viewer);


   VecPointwiseMult(wave.diagonalTMP,Y,Y);
   auto waveNorm=wave.integrator->integrate(wave.diagonalTMP);
   std::cout << "N: " << waveNorm << std::endl;

   
   //std::cout << "evaluate" << std::endl;

   //FormFunction(snes,X,Y,&wave);
   
   //std::cout << "evaluate J" << std::endl;


   //FormJacobianSpherical(snes, Y , J, J, &wave  );
   //MatMult( J, X, Y );
   //PetscObjectSetName((PetscObject)Y,"JY" );
   //VecView(Y, HDF5viewer);


   
   std::cout << "save" << std::endl;
   
   //PetscObjectSetName((PetscObject)J,"J" );
   //MatView(J , ASCIIviewer );

   PetscCall(VecDestroy(&X));
   PetscCall(VecDestroy(&Y));
   //PetscCall(MatDestroy(&H));
   PetscViewerDestroy(&HDF5viewer);
   PetscCall(DMDestroy(&da));

   PetscCall(PetscFinalize());


   return 0;

}
