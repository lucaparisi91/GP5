

#include <petscksp.h>
#include "petscdmda.h"
#include <petscviewerhdf5.h>
#include <iostream>
#include "model.h"
#include <slepceps.h>
#include <petscdmcomposite.h>


static inline PetscInt DMDALocalIndex3D(DMDALocalInfo *info, PetscInt i, PetscInt j, PetscInt k)
{
   return ((i - info->gzs) * info->gym + (j - info->gys)) * info->gxm + (k - info->gxs);

};


void setMatrixLocal00( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox, Vec diagAddL )
{

   PetscInt left[3];
   PetscInt shape[3];
   MatStencil row,col;
   PetscScalar v;
   PetscReal spaceStep[3];
   PetscInt globalShape[3];
   DMDALocalInfo info;
   PetscScalar *** _diagAddL;

   DMDAVecGetArrayRead(da, diagAddL, & _diagAddL);

   {
      DMDAGetLocalInfo(da, &info);
      globalShape[0]=info.mx;
      globalShape[1]=info.my;
      globalShape[2]=info.mz;
   }
   

   for(int d=0;d< DIMENSIONS ;d++)
      {
         spaceStep[d]=(rightBox[d]-leftBox[d] )/globalShape[d];
      }   


   for(int i=info.xs;i<info.xs + info.xm;i++)
      for(int j=info.ys;j<info.ys + info.ym;j++)
         for(int k=info.zs;k<info.zs + info.xm;k++)
         {

            auto x = leftBox[0] + ( i+0.5 )*spaceStep[0];
            auto y = leftBox[1] + ( j+0.5 )*spaceStep[1];
            auto z = leftBox[2] + ( k+0.5 )*spaceStep[2];

            v=0.5*(x*x + y*y + z*z);

            for(int d=0;d<DIMENSIONS;d++)
            {
               v+=1/(spaceStep[d]*spaceStep[d]);
            }

            v+=_diagAddL[k][j][i];

            PetscInt row [1]  { DMDALocalIndex3D(&info, i + 0, j + 0, k + 0) };
            PetscInt cols [ 7 ] { DMDALocalIndex3D(&info, i + 0, j + 0, k + 0) , DMDALocalIndex3D(&info, i + 1, j + 0, k + 0), DMDALocalIndex3D(&info, i - 1, j + 0, k + 0),DMDALocalIndex3D(&info, i, j + 1, k + 0),DMDALocalIndex3D(&info, i , j - 1, k + 0),DMDALocalIndex3D(&info, i + 0 , j + 0, k + 1),DMDALocalIndex3D(&info, i + 0, j + 0, k - 1) };

            PetscScalar vs[ 7 ] { v,-1/(2*spaceStep[0]*spaceStep[0]) , -1/(2*spaceStep[0]*spaceStep[0]) ,-1/(2*spaceStep[1]*spaceStep[1]),-1/(2*spaceStep[1]*spaceStep[1]),-1/(2*spaceStep[2]*spaceStep[2]),-1/(2*spaceStep[2]*spaceStep[2]) };


            MatSetValuesLocal(H, 1, row, 7, cols  ,vs, INSERT_VALUES);


         }

         DMDAVecRestoreArrayRead(da, diagAddL, &_diagAddL);

         //MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
         //MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

}

void setMatrixLocal01( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox, Vec diagAddL )
{

   PetscInt left[3];
   PetscInt shape[3];
   MatStencil row,col;
   PetscScalar v;
   DMDALocalInfo info;
   PetscScalar *** _diagAddL;

   DMDAVecGetArrayRead(da, diagAddL, & _diagAddL);

   DMDAGetLocalInfo(da, &info);



   for(int i=info.xs;i<info.xs + info.xm;i++)
      for(int j=info.ys;j<info.ys + info.ym;j++)
         for(int k=info.zs;k<info.zs + info.xm;k++)
         {

            
            auto v=_diagAddL[k][j][i];

            PetscInt row [1]  { DMDALocalIndex3D(&info, i + 0, j + 0, k + 0) };
            

            MatSetValuesLocal(H, 1, row, 1, row  , &v, INSERT_VALUES);

         }

         DMDAVecRestoreArrayRead(da, diagAddL, &_diagAddL);
         

}

void setMatrixLocal10( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox, Vec diagAddL )
{

   PetscInt left[3];
   PetscInt shape[3];
   MatStencil row,col;
   PetscScalar v;
   DMDALocalInfo info;
   PetscScalar *** _diagAddL;

   DMDAVecGetArrayRead(da, diagAddL, & _diagAddL);

   DMDAGetLocalInfo(da, &info);



   for(int i=info.xs;i<info.xs + info.xm;i++)
      for(int j=info.ys;j<info.ys + info.ym;j++)
         for(int k=info.zs;k<info.zs + info.xm;k++)
         {

            
            auto v=-_diagAddL[k][j][i];

            PetscInt row [1]  { DMDALocalIndex3D(&info, i + 0, j + 0, k + 0) };
            

            MatSetValuesLocal(H, 1, row, 1, row  , &v, INSERT_VALUES);

         }

         DMDAVecRestoreArrayRead(da, diagAddL, &_diagAddL);
         

}

void setMatrixLocal11( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox, Vec diagAddL )
{

   PetscInt left[3];
   PetscInt shape[3];
   MatStencil row,col;
   PetscScalar v;
   PetscReal spaceStep[3];
   PetscInt globalShape[3];
   DMDALocalInfo info;
   PetscScalar *** _diagAddL;

   DMDAVecGetArrayRead(da, diagAddL, & _diagAddL);

   {
      DMDAGetLocalInfo(da, &info);
      globalShape[0]=info.mx;
      globalShape[1]=info.my;
      globalShape[2]=info.mz;
   }

   for(int d=0;d< DIMENSIONS ;d++)
      {
         spaceStep[d]=(rightBox[d]-leftBox[d] )/globalShape[d];
      }

   for(int i=info.xs;i<info.xs + info.xm;i++)
      for(int j=info.ys;j<info.ys + info.ym;j++)
         for(int k=info.zs;k<info.zs + info.xm;k++)
         {

            auto x = leftBox[0] + ( i+0.5 )*spaceStep[0];
            auto y = leftBox[1] + ( j+0.5 )*spaceStep[1];
            auto z = leftBox[2] + ( k+0.5 )*spaceStep[2];

            v=0.5*(x*x + y*y + z*z);

            for(int d=0;d<DIMENSIONS;d++)
            {
               v+=1/(spaceStep[d]*spaceStep[d]);
            }

            v+=_diagAddL[k][j][i];

            PetscInt row [1]  { DMDALocalIndex3D(&info, i + 0, j + 0, k + 0) };
            PetscInt cols [ 7 ] { DMDALocalIndex3D(&info, i + 0, j + 0, k + 0) , DMDALocalIndex3D(&info, i + 1, j + 0, k + 0), DMDALocalIndex3D(&info, i - 1, j + 0, k + 0),DMDALocalIndex3D(&info, i, j + 1, k + 0),DMDALocalIndex3D(&info, i , j - 1, k + 0),DMDALocalIndex3D(&info, i + 0 , j + 0, k + 1),DMDALocalIndex3D(&info, i + 0, j + 0, k - 1) };

            PetscScalar vs[ 7 ] { -v,1/(2*spaceStep[0]*spaceStep[0]) , 1/(2*spaceStep[0]*spaceStep[0]) ,1/(2*spaceStep[1]*spaceStep[1]),1/(2*spaceStep[1]*spaceStep[1]),1/(2*spaceStep[2]*spaceStep[2]),1/(2*spaceStep[2]*spaceStep[2]) };


            MatSetValuesLocal(H, 1, row, 7, cols  ,vs, INSERT_VALUES);


         }

         DMDAVecRestoreArrayRead(da, diagAddL, &_diagAddL);

         //MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
         //MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

}




int main( int argc,char** args)
{
   PetscInt    shape[3] { 100, 100 , 100 };
   PetscMPIInt size;
   PetscReal left[3] { -5,-5,-5};
   PetscReal right[3]{ 5, 5, 5 };
   PetscReal spaceStep[3];
   PetscReal g=100;
   PetscReal mu=4.37768;


   EPS eps;
   Vec psi0;
   Vec psi0L;
   Vec H00D,H00DL,H01D,H01DL;



   for(int d=0;d<3;d++)
   {
      spaceStep[d]=(right[d]-left[d])/shape[d];
   }


   PetscCall(SlepcInitialize(&argc,&args,(char*)0,NULL));

  /*  DM da;

   PetscCall( 
      DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, shape[0], shape[1], shape[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1 , 1, NULL, NULL, NULL, &da)
   );
   PetscCall(DMSetUp(da)); */


   DM da,da_u,da_v;

   PetscCall( 
      DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, shape[0], shape[1], shape[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1 , 1, NULL, NULL, NULL, &da_u)
   );
   PetscCall(DMSetUp(da_u));

   PetscCall( 
      DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, shape[0], shape[1], shape[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1 , 1, NULL, NULL, NULL, &da_v)
   );
   PetscCall(DMSetUp(da_v));

   DMCompositeCreate(PETSC_COMM_WORLD, &da);
   DMSetOptionsPrefix(da, "pack_");
   DMCompositeAddDM(da, da_u);
   DMCompositeAddDM(da, da_v);
   DMDASetFieldName(da_u, 0, "u");
   DMDASetFieldName(da_v, 0, "v");
   DMSetFromOptions(da);



   /*  Loads the ground state solution */
 
   PetscCall(DMCreateGlobalVector(da_u,&psi0));
   PetscCall(DMCreateGlobalVector(da_u,&H00D));
   PetscCall(DMCreateGlobalVector(da_u,&H01D));


   PetscViewer HDF5viewer;
   PetscViewerHDF5Open(PETSC_COMM_WORLD, "X.hdf5", FILE_MODE_READ, &HDF5viewer);
   PetscObjectSetName((PetscObject)psi0,"solution" );
   VecLoad(psi0, HDF5viewer);
   PetscObjectSetName((PetscObject)psi0,"psi0" );

   /* Creates the diagonal vector for H00 */

   PetscCall( VecPointwiseMult(H00D, psi0, psi0) );
   VecScale(H00D, 2*g );
   VecShift( H00D , -mu );
   PetscCall( VecPointwiseMult(H01D, psi0, psi0) );
   VecScale(H01D, g );


   Mat H;
   Mat Bs[4];

   DMSetMatType(da, MATAIJ);
   DMSetMatType(da_u, MATAIJ);
   DMSetMatType(da_v, MATAIJ);
   
   DMCreateMatrix(da, &H );
   MatMPIAIJSetPreallocation(H, 8, NULL, 1 , NULL);
   MatSeqAIJSetPreallocation(H, 8, NULL );



   //MatSetOption(H, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);

   IS *is;

   DMCompositeGetLocalISs(da, &is);
   
   PetscCall( DMGetLocalVector(da_u,&H00DL) );
   PetscCall( DMGlobalToLocalBegin(da_u, H00D, INSERT_VALUES, H00DL));
   PetscCall( DMGlobalToLocalEnd(da_u, H00D, INSERT_VALUES, H00DL) );

   PetscCall( DMGetLocalVector(da_u,&H01DL) );
   PetscCall( DMGlobalToLocalBegin(da_u, H01D, INSERT_VALUES, H01DL));
   PetscCall( DMGlobalToLocalEnd(da_u, H01D, INSERT_VALUES, H01DL) );


   std::cout << "Build 00" << std::endl;

   MatGetLocalSubMatrix(H, is[0], is[0], &(Bs[0] ) );
   setMatrixLocal00(da_u,Bs[0],left,right,H00DL);
   MatRestoreLocalSubMatrix(H, is[0], is[0], &(Bs[0]));
   std::cout << "Build 11" << std::endl;
   MatGetLocalSubMatrix(H, is[1], is[1], &(Bs[3] ) );
   setMatrixLocal11(da_u,Bs[3],left,right,H00DL);
   MatRestoreLocalSubMatrix(H, is[1], is[1], &(Bs[3]));
   
   std::cout << "Build 01" << std::endl;
   MatGetLocalSubMatrix(H, is[0], is[1], &(Bs[1] ) );
   setMatrixLocal01(da_u,Bs[1],left,right,H01DL);
   MatRestoreLocalSubMatrix(H, is[0], is[1], &(Bs[1]));


   std::cout << "Build 10" << std::endl;
   MatGetLocalSubMatrix(H, is[1], is[0], &(Bs[2] ) );
   setMatrixLocal10(da_u,Bs[2],left,right,H01DL);
   MatRestoreLocalSubMatrix(H, is[1], is[0], &(Bs[2])); 
   

   PetscCall( DMRestoreLocalVector(da,&H00DL) );
   PetscCall( DMRestoreLocalVector(da,&H01DL) );
   
   std::cout << "Assembly" << std::endl;
   MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);


  /*  DMSetMatType(da, MATAIJ);
   DMCreateMatrix(da, &(Bs[0]) );
   DMCreateMatrix(da, &(Bs[3]) );
   DMCreateMatrix(da, &(Bs[2]) );
   DMCreateMatrix(da, &(Bs[1]) );

   setMatrix(da,Bs[0],left,right);
   MatScale(Bs[0], 1 );
 */
   //PetscCall(MatDiagonalSet(Bs[0], H00D ,  ADD_VALUES ) );

   //PetscCall(MatDiagonalSet(Bs[1], H01D ,  INSERT_VALUES ) );
   
   //PetscCall( MatDiagonalSet(Bs[2], H01D ,  INSERT_VALUES ) );
   //PetscCall( MatScale(Bs[2], -1 ) );   

   //setMatrix(da,Bs[3],left,right);
   //PetscCall(MatDiagonalSet(Bs[3], H00D ,  ADD_VALUES ) );
   //MatScale(Bs[3], -1 );

   //Bs[1]=NULL;
   //Bs[2]=NULL;


   //MatCreateNest(PETSC_COMM_WORLD, 2 , NULL, 2, NULL , Bs, &H);

   PetscViewer ASCIIviewer;
   PetscViewerASCIIOpen(PETSC_COMM_WORLD, "M.txt", &ASCIIviewer);
   PetscViewerPushFormat(ASCIIviewer, PETSC_VIEWER_ASCII_DENSE);
   //MatView(H, ASCIIviewer);
   PetscViewerDestroy(&ASCIIviewer);


   
   std::cout << "evp init" << std::endl;

   PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
   PetscCall(EPSSetOperators(eps,H,NULL));
   PetscCall(EPSSetProblemType(eps,EPS_NHEP));
   PetscCall(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE));
   PetscCall(EPSSetFromOptions(eps));
   PetscCall(EPSSolve(eps));

    PetscInt nconv;
    PetscCall(EPSGetConverged(eps,&nconv));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %" PetscInt_FMT "\n\n",nconv));


    for(int i=0;i<nconv;i++)
    {
        PetscScalar eigr, eigi;
        EPSGetEigenvalue(eps,i, & eigr, & eigi);
        std::cout << "Eig "<< i << ":" << eigr << " " << eigi << std::endl;
    }

   PetscCall(EPSDestroy(&eps));

    
    //PetscCall(MatDestroy(&H));
    //PetscCall(DMDestroy(&da));
   PetscCall(SlepcFinalize() ); 

   

};