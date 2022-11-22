


#include <petscksp.h>
#include "petscdmda.h"
#include <petscviewerhdf5.h>
#include <iostream>
#include <petscsnes.h>
#include "model.h"


struct petscWave
{
   PetscReal right[DIMENSIONS];
   PetscReal left[DIMENSIONS];
   PetscReal spaceStep[DIMENSIONS];
   Mat J0;
   PetscReal mu;
   Vec diagonalTMP;
   PetscScalar g;
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
            _d[k][j][i] += 3*g*_x[k][j][i] * _x[k][j][i]  ;
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




int main(int argc, char **args)
{

   PetscInt    shape[3] { 20, 20, 20 };
   PetscMPIInt size;
   PetscReal left[3] { -5,-5,-5};
   PetscReal right[3]{ 5, 5, 5 };
   PetscReal spaceStep[3];


   for(int d=0;d<3;d++)
   {
      spaceStep[d]=(right[d]-left[d])/shape[d];
   }

   PetscCall(PetscInitialize(&argc, &args, (char *)0, NULL));
   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
   PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

   Vec  X,x;
   DM da;
   PetscScalar ***_X;
   PetscInt mstart,nstart,pstart,m,n,p;
   Mat H;
   

   //PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

   // initialize discretization
   PetscCall( 
      DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, shape[0], shape[1], shape[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1 , 1, NULL, NULL, NULL, &da)
   );
   //PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 4, 1, 0, 0, &da));


   PetscCall(DMSetFromOptions(da));
   PetscCall(DMSetUp(da));
   DMDASetUniformCoordinates(da, left[0], right[0], left[1], right[1], left[2],right[2] );

   // initialize starting vector
   PetscCall(DMCreateGlobalVector(da,&X));
   PetscCall(DMDAVecGetArray(da,X, &_X) );
   PetscCall( DMDAGetCorners(da, &mstart, &nstart, &pstart, &m, &n, &p) );

   for(int i=mstart;i<mstart + m ; i++)
      for(int j=nstart;j<nstart + n; j++)
         for(int k=pstart;k<pstart + p; k++)
         {
            auto x = left[0] + ( i+0.5 )*spaceStep[0];
            auto y = left[1] + ( j+0.5 )*spaceStep[1];
            auto z = left[2] + ( k+0.5 )*spaceStep[2];

            auto r2 = x*x + y*y + z*z;

            _X[k][j][i]=exp(-r2/(2*2*2) );
         }
   
  


   PetscCall(DMDAVecRestoreArray(da,X,&_X) );
   PetscCall(VecAssemblyBegin(X));
   PetscCall(VecAssemblyEnd(X));   

   auto dV = spaceStep[0]*spaceStep[1]*spaceStep[2];
   PetscReal norm;
   VecNorm(X,NORM_2,&norm);
   norm*=sqrt(dV);


   VecScale(X, 1/norm );


   PetscViewer HDF5viewer;
   PetscViewerHDF5Open(PETSC_COMM_WORLD, "X.hdf5", FILE_MODE_WRITE, &HDF5viewer);
   

   PetscObjectSetName((PetscObject)X,"init" );
   VecView(X, HDF5viewer);


   PetscViewer ASCIIviewer;
   PetscViewerASCIIOpen(PETSC_COMM_WORLD, "M.txt", &ASCIIviewer);
   PetscViewerPushFormat(ASCIIviewer, PETSC_VIEWER_ASCII_DENSE);
   

 /*   DMSetMatType(da, MATAIJ);
   DMCreateMatrix(da, &H);
   std::cout << "build up matrix" << std::endl;
   setMatrix(da,H,left,right); */
   //MatView(H, ASCIIviewer);


   Vec Y;
   VecDuplicate(X,&Y);
   VecSet(Y, 0.);   


   
   petscWave wave;
   for(int d=0;d<DIMENSIONS;d++)
   {
      wave.left[d]=left[d];
      wave.right[d]=right[d];
      wave.spaceStep[d]=spaceStep[d];
      wave.mu=0;
      wave.g=1;
   }


   VecDuplicate(X, &(wave.diagonalTMP) );

   SNES snes;
   PetscCall( SNESCreate( PETSC_COMM_WORLD, &snes));
   PetscCall( SNESSetDM(snes, da));

   std::cout << "set up J" << std::endl;
   Mat J;
   DMSetMatType(da, MATAIJ);
   DMCreateMatrix(da, &J);
   DMCreateMatrix(da, &(wave.J0) );

   setMatrix( da, J , left, right );
   
   MatCopy(J,wave.J0,DIFFERENT_NONZERO_PATTERN);


   PetscCall( SNESSetFunction(snes,Y,&FormFunction, & wave) );
   PetscCall( SNESSetJacobian(snes, J , J ,&FormJacobian, & wave) );
   PetscCall(SNESSetFromOptions(snes));


   PetscCall(SNESSolve(snes, NULL, X));
   
   //std::cout << "evaluate" << std::endl;

   //FormFunction(snes,X,Y,&wave);

   //std::cout << "evaluate J" << std::endl;
   //FormJacobian(snes, X , J, J, &jCtx );




   
   std::cout << "save" << std::endl;
   
   //PetscObjectSetName((PetscObject)J,"J" );
   //MatView(J , ASCIIviewer );


   PetscObjectSetName((PetscObject)X,"solution" );
   VecView( X , HDF5viewer);




   PetscCall(VecDestroy(&X));
   PetscCall(VecDestroy(&Y));
   PetscCall(MatDestroy(&H));
   PetscViewerDestroy(&HDF5viewer);
   PetscCall(DMDestroy(&da));

   PetscCall(PetscFinalize());


   return 0;

}
