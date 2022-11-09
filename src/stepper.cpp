#include "stepper.h"
#include <iostream>
#include "timers.h"
namespace gp
{

    void stepper::normalize(tensor_t & field)
    {

        START_TIMER("normalize");
        int nComponents=field.dimensions()[DIMENSIONS];

        if ( reNormalize() )
        {    
            for(int c=0;c<nComponents;c++)
            {
                //std::cout << normalizations()[c] << std::endl;
                gp::normalize( normalizations()[c],field,c,getDiscretization() );
            };
        }
        STOP_TIMER("normalize");   
    }        

    stepper::stepper() :
    _reNormalize(false)
    {
        _constraint=std::make_shared<nullConstraint>();
        
    }
    void euleroStepper::advance(  tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        START_TIMER("step");

        getFunctional()->apply(fieldDataOld,fieldDataNew,time);
        fieldDataNew= fieldDataOld - timeStep()*fieldDataNew;
        int nComponents=fieldDataOld.dimensions()[DIMENSIONS];

        STOP_TIMER("step");    

        getConstraint()->apply(fieldDataNew);

        normalize(fieldDataNew);

    }

    void RK4Stepper::advance(  tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        START_TIMER("step");    
        getFunctional()->apply(fieldDataOld,k1,time);
        fieldDataNew= fieldDataOld - timeStep()*0.5*k1;
        normalize(fieldDataNew);

        getFunctional()->apply(fieldDataNew,k2,time + 0.5*std::abs(timeStep()) );

        fieldDataNew= fieldDataOld - timeStep()*0.5*k2;
        normalize(fieldDataNew);

        getFunctional()->apply( fieldDataNew,k3, time + 0.5*std::abs(timeStep()) );

        fieldDataNew= fieldDataOld - timeStep()*k3;
        normalize(fieldDataNew);
        getFunctional()->apply(fieldDataNew,k4,time + std::abs(timeStep()) );

        fieldDataNew=fieldDataOld - (1./6) * timeStep()*(k1 + 2 * k2 + 2*k3 + k4);
        getConstraint()->apply(fieldDataNew);

        normalize(fieldDataNew);

        STOP_TIMER("step");       

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