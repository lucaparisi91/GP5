#include "../src/traits.h"
#include "spherical.h"


class euleroOptimization
{
    public:
    
    using model_t = gpSpherical;

    PetscErrorCode optimize( Vec psi ); // fills X with the evolved state

    euleroOptimization(real_t dt,std::shared_ptr<model_t> model);

    void setStepsPerBlock(int n) {_stepsPerBlock=n; }
    void setMaxError( real_t error) {_maxError=error; }

    private:

    real_t _dt;
    real_t _maxError;
    std::shared_ptr<model_t> _model;
    int _stepsPerBlock;

};