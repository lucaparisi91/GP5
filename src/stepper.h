#include "traits.h"
#include "functional.h"
#include "geometry.h"
#include "constraint.h"


namespace gp
{
    class stepper
    {

        public:

        stepper();

        using functional_t = functional;

        virtual void advance( tensor_t & fieldOld, tensor_t & fieldNew, real_t time)=0;

        void setTimeStep(complex_t timeStep){_timeStep=timeStep;}

        void setDiscretization(std::shared_ptr<discretization> discr)
        {
            _discr=discr;
        }

        virtual void init() {};

        virtual void setNormalizations( const std::vector<real_t> & normalizations_)
        {
            _normalizations=normalizations_;
        }

        virtual  const std::vector<real_t> & normalizations() {return _normalizations;}

        complex_t timeStep() const  { return _timeStep;}

        

        auto getFunctional() { return _func;}
        
        void setFunctional(std::shared_ptr<functional_t> func) {_func=func;}
        virtual void setNComponents(int n) {_nComponents=n;}
        auto nComponents(){return _nComponents;}

        auto getDiscretization() {return _discr;}

        void enableReNormalization(bool enable ) {_reNormalize=enable;}
        bool reNormalize() const  {return _reNormalize;}

        auto getConstraint() const {return _constraint; }
        
        void  setConstraint(std::shared_ptr<constraint> constr) { _constraint=constr; }
        

        private:

        complex_t _timeStep;

        std::shared_ptr<functional_t> _func;
        int _nComponents=0;
        std::vector<real_t> _normalizations;
        std::shared_ptr<discretization> _discr;
        bool _reNormalize;

        std::shared_ptr<constraint> _constraint;
    };

    class euleroStepper : public stepper
    {
        public:
        euleroStepper(){};

        virtual void advance( tensor_t & fieldOld, tensor_t & fieldNew, real_t time) ;


        private:

    };


    class RK4Stepper : public stepper
    {
        public:

        RK4Stepper(){}

        virtual void advance( tensor_t & fieldOld, tensor_t & fieldNew, real_t time);

        virtual void init() override;

        private:

        tensor_t k1;
        tensor_t k2;
        tensor_t k3;
        tensor_t k4;
    };

}