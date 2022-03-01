#include "stepper.h"
#include <iostream>


namespace gp
{

    void euleroStepper::advance(  tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {

        getFunctional()->apply(fieldDataOld,fieldDataNew,time);

        fieldDataNew= fieldDataOld - timeStep()*fieldDataNew;
        int nComponents=fieldDataOld.dimensions()[DIMENSIONS];
        
        for(int c=0;c<nComponents;c++)
        {
            normalize( normalizations()[c],fieldDataNew,c,getDiscretization() );
        };

    }
    

    void RK4Stepper::advance(  tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {    
        getFunctional()->apply(fieldDataOld,k1,time);
        fieldDataNew= fieldDataOld - timeStep()*0.5*k1;
        
        getFunctional()->apply(fieldDataNew,k2,time + 0.5*std::abs(timeStep()) );

        fieldDataNew= fieldDataOld - timeStep()*0.5*k2;

        getFunctional()->apply( fieldDataNew,k3, time + 0.5*std::abs(timeStep()) );

        fieldDataNew= fieldDataOld - timeStep()*k3;
        getFunctional()->apply(fieldDataNew,k4,time + std::abs(timeStep()) );

        fieldDataNew=fieldDataOld - (1./6) * timeStep()*(k1 + 2 * k2 + 2*k3 + k4);

        int nComponents=fieldDataOld.dimensions()[DIMENSIONS];

        for(int c=0;c<nComponents;c++)
        {
            normalize( normalizations()[c],fieldDataNew,c,getDiscretization() );
        };  

    }

    void RK4Stepper::init()
    {
        auto localShape = getDiscretization()->getLocalMesh()->shape();

        k1= tensor_t( EXPAND_D(localShape) , nComponents()  );
        k2= tensor_t( EXPAND_D(localShape) , nComponents()  );
        k3= tensor_t( EXPAND_D(localShape) , nComponents()  );
        k4= tensor_t( EXPAND_D(localShape) , nComponents()  );

    }

}