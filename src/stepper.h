#include "traits.h"
#include "functional.h"


class stepper
{
    public:

    virtual void apply( tensor_t & fieldOld, tensor_t & fieldNew, Real time)=0;

    void setTimeStep(Real timeStep){_timeStep=timeStep;}

    void setDiscretization(std::shared_ptr<discretization> discr)
    {
        _discr=discr;
    }

    virtual void init()

    virtual void setNormalizations( const std::vector<Real> & normalizations_)
    {
        _normalizations=normalizations_;
    }

    virtual  const std::vector<Real> & normalizations() {return _normalizations;}


    Real timeStep() const  { return _timeStep;}

    auto  & functional() { return *_func;}
    
    void setFunctional(std::shared_ptr<functional_t> func) {_func=func;}
    virtual void setNComponents(int n) {_nComponents=n;}
    auto nComponents(){return _nComponents;}

    auto getDiscreatization() {return _discr;}

    private:

    Real _timeStep;
    std::shared_ptr<functional_t> _func;
    int _nComponents=0;
    std::vector<Real> _normalizations;
    std::shared_ptr<discretization> _discr;

};


class euleroStepper : public stepper
{
    public:

    euleroStepper(){};

    virtual void advance( tensor_t & fieldOld, tensor_t & fieldNew, Real time) ;

    private:

};


class RK4Stepper : public stepper
{
    public:

    RK4Stepper(){}

    virtual void advance( tensor_t & fieldOld, tensor_t & fieldNew, Real time);

    virtual void init() override;

    private:

    tensor_t k1;
    tensor_t k2;
    tensor_t k3;
    tensor_t k4;
};