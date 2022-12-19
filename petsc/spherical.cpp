#include "spherical.h"

void setKineticMatrixSpherical( DM da, Mat H, const PetscReal* leftBox,const PetscReal * rightBox )
{


   PetscInt left[3];
   PetscInt shape[3];
   MatStencil row,col;
   PetscScalar v;
   PetscReal spaceStep[3];
   PetscInt globalShape[3];
   DMDALocalInfo info;
   DMDAGetLocalInfo(da, &info);


   for(int d=0;d< 1 ;d++)
      {
         spaceStep[d]=(rightBox[d]-leftBox[d] )/info.mx;
      }   
   
   for( int i=info.xs;i<info.xs + info.xm  ;i++ )
         {
            if (i==0)
            {
               auto x = leftBox[0] + 0.5*spaceStep[0];
               v=0.5*x*x + 1/(spaceStep[0]*spaceStep[0]) -1/(2*spaceStep[0]*spaceStep[0]) + 0.5/(x*spaceStep[0]);



               MatSetValues(H, 1, &i , 1, &i  ,&v, INSERT_VALUES);
               

               v=-1/(2*spaceStep[0]*spaceStep[0]) - 0.5/(x*spaceStep[0]);   

               auto col = i + 1;

               MatSetValues(H, 1, &i, 1, &col  ,&v, INSERT_VALUES);

            }
            else if (i==info.xm - 1)
            {
               auto x = leftBox[0] + ( i+0.5 )*spaceStep[0];

               v=0.5*x*x + 1/(spaceStep[0]*spaceStep[0]) -1/(2*spaceStep[0]*spaceStep[0]) - 0.5/(x*spaceStep[0]);   

               MatSetValues(H, 1, &i , 1, &i  ,&v, INSERT_VALUES);
               

               v=-1/(2*spaceStep[0]*spaceStep[0]) + 0.5/(x*spaceStep[0]);
               auto col = i - 1;
               MatSetValues(H, 1, &i, 1, &col  ,&v, INSERT_VALUES);
               
            }
            else
            {

               auto x = leftBox[0] + ( i+0.5 )*spaceStep[0];

               v=0.5*(x*x ) + 1/(spaceStep[0]*spaceStep[0]);


               MatSetValues(H, 1, &i , 1, &i  ,&v, INSERT_VALUES);
               

               v=-1/(2*spaceStep[0]*spaceStep[0]) + 0.5/(x*spaceStep[0]);


               auto col = i - 1;

               MatSetValues(H, 1, &i, 1, &col  ,&v, INSERT_VALUES);
               col=i+1;
               v=-1/(2*spaceStep[0]*spaceStep[0]) - 0.5/(x*spaceStep[0]);   

               MatSetValues(H, 1, &i, 1, &col  ,&v, INSERT_VALUES);
            }

         }

         MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
         MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

}


void gpSpherical::evaluate( Vec X, Vec Y )
{
   Vec x;
   auto & geo = getGeometry();


   DMGetLocalVector(geo.getDM(),&x) ;
   DMGlobalToLocalBegin(geo.getDM(), X, INSERT_VALUES, x);
   DMGlobalToLocalEnd(geo.getDM(), X, INSERT_VALUES, x) ;

   PetscScalar * _x;
   PetscScalar * _y;

   DMDALocalInfo info;

   DMDAGetLocalInfo(geo.getDM(), &info);

   DMDAVecGetArrayRead(geo.getDM(),x, &_x) ;
   DMDAVecGetArray(geo.getDM(), Y , &_y) ;

   auto spaceStepInverse2=1/(geo.getSpaceStep()[0] * geo.getSpaceStep()[0] );

   _x[-1]=_x[0];
   _x[info.mx]=_x[info.mx-1];

   for(int i=info.xs;i<info.xs + info.xm ;i++ )
         {

            auto x = geo.getLeft()[0] + ( i+ 0.5 )*geo.getSpaceStep()[0];

            _y[i]=  _x[i]* ( spaceStepInverse2 + 0.5*( x * x  ) ) 
            - _x[i-1]*0.5*( spaceStepInverse2 - 1./(geo.getSpaceStep()[0] * x) )- _x[i+1]* 0.5 *( spaceStepInverse2  + 1./(geo.getSpaceStep()[0] * x) )
            + g * (_x[i] * _x[i] * _x[i]  );
             ;
         }


   DMDAVecRestoreArrayRead(geo.getDM(),x, &_x) ;
   DMRestoreLocalVector(geo.getDM(), &x) ;
   
   DMDAVecRestoreArray(geo.getDM(), Y , &_y) ;

   VecAXPY(Y,-mu ,X);

}




PetscErrorCode gpSpherical::evaluateJacobian( Vec X , Mat jac)
{

   DMDALocalInfo info;
   auto & da = _geo->getDM();
   Vec x;

   DMDAGetLocalInfo(da, &info);

   
   //std::cout << "J: " << Jctx->mu << std::endl;

   MatCopy( _J0, jac , SAME_NONZERO_PATTERN );   


   // add the nonlinear term to the diagonal
   PetscScalar * _x;
   PetscScalar * _d;
   

   PetscCall(DMDAVecGetArrayRead(da,x, &_x) );
   PetscCall(DMDAVecGetArray(da,diagonalTMP, &_d) );

   for(int i=info.xs;i<info.xs + info.xm;i++)
         {
            _d[i] = 3* g*_x[i] * _x[i] -mu  ;
         }
   

   PetscCall(DMDAVecRestoreArrayRead(da,x, &_x) );
   PetscCall(DMDAVecRestoreArray(da,diagonalTMP, &_d) );


   MatDiagonalSet(jac,diagonalTMP,ADD_VALUES);

   return 0;
}


PetscErrorCode gpSpherical::initialize()
{
   auto & geo = getGeometry();

   DMSetMatType(geo.getDM(), MATAIJ);
   DMCreateMatrix(geo.getDM(), &_J0);

   setKineticMatrixSpherical( geo.getDM(), _J0 , geo.getLeft(), geo.getRight() );

   PetscCall( DMCreateGlobalVector(geo.getDM(),&diagonalTMP) );



   return 0;
}

geometry::geometry( real_t R, PetscInt shape) 
{

  DMDACreate1d(PETSC_COMM_WORLD , DM_BOUNDARY_GHOSTED, shape, 1, 1, NULL, & (_da) ) ;
  DMSetFromOptions(_da);
  DMSetUp(_da);
  _left[0]=0;
  _right[0]=R;
  _spaceStep[0]=R/shape;
  _integrator=std::make_shared<sphericalIntegrator>(_da,0,R);

}



sphericalIntegrator::sphericalIntegrator( DM da, PetscReal left, PetscReal right) : _da(da) {
      DMCreateGlobalVector(da,&_r2) ;
      DMCreateGlobalVector(da,&_tmp) ;
      fillSpacePositions1D(da,_r2,&left,&right);
      VecPointwiseMult(_r2,_r2,_r2);

      DMDALocalInfo info;
      DMDAGetLocalInfo(da, &info);

      _dx=(right-left )/info.mx;

   }

real_t sphericalIntegrator::integrate(Vec psi)
   {
      PetscScalar integ;
      VecPointwiseMult(_tmp,psi,_r2);
      VecSum(_tmp, &integ );
      return integ*4*M_PI*_dx;
   }

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