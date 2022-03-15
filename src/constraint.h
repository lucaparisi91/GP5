#include "traits.h"
#include "geometry.h"


namespace gp
{

    class constraint
    {
        public:
        virtual void apply(tensor_t & field) const =0;
    };


    class nullConstraint : public constraint
    {
        public:

        virtual void apply(tensor_t & field) const {};

    };

    class phaseConstraint : public constraint
    {
        public:
        phaseConstraint( const tensor_t & phase);
        void apply(tensor_t & phi) const;

        private:
        
        tensor_t _phase_exp;
        tensor_t _phase;

    };


    class constraintConstructor
    {
        public:

        void setDiscretization( std::shared_ptr<discretization> discr )
        { _discr=discr; }

        std::shared_ptr<constraint> create( YAML::Node config);

        void setNComponents( int nComponents ){_nComponents=nComponents;}
        private:

        std::shared_ptr<discretization> _discr;
        int _nComponents;        
    };


}