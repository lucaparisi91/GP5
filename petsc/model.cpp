#include "model.h"


void setMatrix( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox )
{


   PetscInt left[3];
   PetscInt shape[3];
   MatStencil row,col;
   PetscScalar v;
   PetscReal spaceStep[3];
   PetscInt globalShape[3];
   DMDALocalInfo info;
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

   DMDAGetCorners( da, &(left[0]), &(left[1]), &(left[2]), &(shape[0]), &(shape[1]),  &(shape[2]) );

   for(int i=left[0];i<left[0] + shape[0];i++)
      for(int j=left[1];j<left[1] + shape[1];j++)
         for(int k=left[2];k<left[2] + shape[2];k++)
         {
            row.i=i;
            row.j=j;
            row.k=k;

            auto x = leftBox[0] + ( i+0.5 )*spaceStep[0];
            auto y = leftBox[1] + ( j+0.5 )*spaceStep[1];
            auto z = leftBox[2] + ( k+0.5 )*spaceStep[2];


            v=0.5*(x*x + y*y + z*z);


            for(int d=0;d<DIMENSIONS;d++)
            {
               v+=1/(spaceStep[d]*spaceStep[d]);
            }

            MatSetValuesStencil(H, 1, &row, 1, &row  ,&v, INSERT_VALUES);
            
            v=-1/(2*spaceStep[0]*spaceStep[0]);   
            col.i=i-1;
         
            col.j=j;
            col.k=k;
            MatSetValuesStencil(H, 1, &row, 1, &col  ,&v, INSERT_VALUES);
            col.i=i+1;
            MatSetValuesStencil(H, 1, &row, 1, &col  ,&v, INSERT_VALUES);

            v=-1/(2*spaceStep[1]*spaceStep[1]);   
            col.i=i;
            col.j=j-1;
            col.k=k;
            MatSetValuesStencil(H, 1, &row, 1, &col  ,&v, INSERT_VALUES);
            col.j=j+1;
            MatSetValuesStencil(H, 1, &row, 1, &col  ,&v, INSERT_VALUES);

            v=-1/(2*spaceStep[2]*spaceStep[2]);   
            col.i=i;
            col.j=j;
            col.k=k-1;
            MatSetValuesStencil(H, 1, &row, 1, &col  ,&v, INSERT_VALUES);
            col.k=k+1;
            MatSetValuesStencil(H, 1, &row, 1, &col  ,&v, INSERT_VALUES);
 
         }



         MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
         MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

}
