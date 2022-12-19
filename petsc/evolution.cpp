   
#include "evolution.h"
#include <iostream>


euleroOptimization::euleroOptimization(real_t dt,std::shared_ptr<model_t> model) : _dt(dt),_model(model)
{
    _maxError=1e-1;
    _stepsPerBlock=1000;
}

PetscErrorCode euleroOptimization::optimize(Vec psi)
{
    
    real_t error=1e+9;
    Vec psi2;
    Vec density;
    Vec Hpsi;
    VecDuplicate(psi, &psi2 );
    VecDuplicate(psi, &density );
    VecDuplicate(psi, &Hpsi );
    

    Vec * oldPsi;
    Vec * newPsi;
    oldPsi=&psi2;
    newPsi=&psi;



    while( error > _maxError )
    {
        for(int tt=0;tt<_stepsPerBlock;tt++)
        {
            std::swap(oldPsi,newPsi);

            _model->evaluate(*oldPsi,Hpsi);

            VecWAXPY(*newPsi,-1.0*_dt,Hpsi,*oldPsi);
        }

       
        VecPointwiseMult(density,*newPsi,*newPsi);

        auto N=_model->getGeometry().getIntegrator().integrate(density);

        _model->evaluate(*oldPsi,Hpsi);


        VecNorm(Hpsi,NORM_2,&error);

        std::cout << "N: " << N << std::endl;
        std::cout << "error: " << error << std::endl;

    }

    if ( newPsi != &psi)
    {
        VecCopy(*newPsi,psi);
    }

    PetscCall(VecDestroy(&psi2));
    PetscCall(VecDestroy(&density));
    PetscCall(VecDestroy(&Hpsi));




    return 0;

   };
