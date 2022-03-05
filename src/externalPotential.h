#include "traits.h"
#include "geometry.h"
#include <memory>

namespace gp
{
    class externalPotential
    {
        public:
        virtual std::shared_ptr<tensor_t> create( std::shared_ptr<discretization> discr, int nComponents ) const = 0 ;  
    };

    class harmonicPotential : public externalPotential
    {
        public:
        harmonicPotential(const std::vector<realDVec_t> & omegas);

        virtual std::shared_ptr<tensor_t> create( std::shared_ptr<discretization> discr, int nComponents ) const ;


        private:

        std::vector<realDVec_t> _omegas;

    };

    class vortexPotential : public externalPotential
    {
        public:
        vortexPotential(){}
        virtual std::shared_ptr<tensor_t> create( std::shared_ptr<discretization> discr, int nComponents ) const ;  
    };

    class sumPotential : public externalPotential
    {
        public:
        
        sumPotential(std::vector<std::shared_ptr<externalPotential> > pots) : _pots(pots) {};

        virtual std::shared_ptr<tensor_t> create( std::shared_ptr<discretization> discr, int nComponents ) const ;

        private:

        std::vector<std::shared_ptr<externalPotential> > _pots;
    };


    class potentialFromFile : public externalPotential
    {
        public:
        
        potentialFromFile(std::string filename) : _filename(filename) { }

        virtual std::shared_ptr<tensor_t> create( std::shared_ptr<discretization> discr, int nComponents ) const ;


        
        private:
        std::string _filename;

    };

    class externalPotentialConstructor
    {
        public:
        std::shared_ptr<externalPotential> create( const config_t & settings );

    };

}

