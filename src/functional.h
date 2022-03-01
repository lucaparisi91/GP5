#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H


#include "operators.h"

namespace gp
{
    class functional
    {
    public:

        functional() : _nComponents(1) {}

        virtual void apply( tensor_t & fieldOld, tensor_t & fieldNew, real_t time )=0;

        void setLaplacianOperator(std::shared_ptr<gp::operators::laplacian> op) {_lap=op;}

        auto getLaplacianOperator() {return _lap;}


        virtual std::shared_ptr<discretization> getDiscretization() {return _discr;}

        virtual void setDiscretization(std::shared_ptr<discretization> discr) {_discr=discr;}

        auto getNComponents() const {return _nComponents;}
        auto setNComponents(int n) {_nComponents=n;}
        

        virtual void init() { }
        private:

        std::shared_ptr<gp::operators::laplacian> _lap;
        std::shared_ptr<discretization> _discr;
        int _nComponents;

    };

    class gpFunctional : public functional
    {
        public:
        
        gpFunctional() ;
        
        virtual void setOmegas( const std::vector<realDVec_t> & omegas){_omegas=omegas; _setTrappingPotential=true; }


        virtual void setCouplings( const  Eigen::Tensor<real_t,2> & couplings) {_couplings=couplings;_setCouplings=true; }

        virtual void init() override;

        virtual void apply( tensor_t & source, tensor_t & destination, real_t time ) override;

        void setMasses(const std::vector<real_t> & masses){_masses=masses;}

        private:

        void addPotential( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time );
        void addSingleComponentMeanFieldInteraction( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time );
        void addTwoComponentMeanFieldInteraction( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time );

        std::shared_ptr<tensor_t> _V;
        std::vector<realDVec_t> _omegas;

        Eigen::Tensor<real_t,2> _couplings;
        std::vector<real_t> _masses;

        std::vector<real_t> _inverseMasses;

        bool _setTrappingPotential;
        bool _setCouplings;

    };

}


#endif