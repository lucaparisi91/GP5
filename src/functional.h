#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H


#include "operators.h"

namespace gp
{
    class functional
    {
    public:

        functional() : _nComponents(1),_setExternalPotential(false) {}

        virtual void apply( tensor_t & fieldOld, tensor_t & fieldNew, real_t time )=0;

        void setLaplacianOperator(std::shared_ptr<gp::operators::laplacian> op) {_lap=op;}

        auto getLaplacianOperator() {return _lap;}


        virtual std::shared_ptr<discretization> getDiscretization() {return _discr;}

        virtual void setDiscretization(std::shared_ptr<discretization> discr) {_discr=discr;}

        auto getNComponents() const {return _nComponents;}
        void setNComponents(int n) {_nComponents=n;}


        void setExternalPotential( std::shared_ptr<tensor_t> V) {_V=V;_setExternalPotential=true; }    

        virtual void init() { }
        private:

        std::shared_ptr<gp::operators::laplacian> _lap;
        std::shared_ptr<discretization> _discr;
        int _nComponents;
        
        bool _setExternalPotential;

        std::shared_ptr<tensor_t> _V;

        protected:

        void addPotential( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time );


    };

    class gpFunctional : public functional
    {
        public:
        
        gpFunctional() ;

        virtual void setCouplings( const  Eigen::Tensor<real_t,2> & couplings) {_couplings=couplings;_setCouplings=true; }

        virtual void init() override;

        virtual void apply( tensor_t & source, tensor_t & destination, real_t time ) override;

        void setMasses(const std::vector<real_t> & masses){_masses=masses;}


        
        
        private:

        
        void addSingleComponentMeanFieldInteraction( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time );
        void addTwoComponentMeanFieldInteraction( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time );



        std::vector<realDVec_t> _omegas;

        Eigen::Tensor<real_t,2> _couplings;
        std::vector<real_t> _masses;

        std::vector<real_t> _inverseMasses;

        bool _setCouplings;

    };


    class LHYDropletFunctional : public functional
    {
        public:
        LHYDropletFunctional() : functional() {}

        virtual void apply( tensor_t & source, tensor_t & destination, real_t time ) override;


    };

    class LHYDropletUnlockedFunctional : public functional
    {
        public:
        LHYDropletUnlockedFunctional(real_t alpha, real_t beta,real_t eta) : functional(),_alpha(alpha),_beta(beta),_eta(eta) {}

        virtual void apply( tensor_t & source, tensor_t & destination, real_t time ) override;

        private:

        real_t _alpha;
        real_t _beta;
        real_t _eta;

    };



class functionalConstructor
{
    public:
    std::shared_ptr<functional> create(const config_t & settings);

    void setLaplacianOperator(std::shared_ptr<gp::operators::laplacian> op) {_laplacian=op;}


    void setNComponents(int n) {_nComponents=n;}

    virtual void setDiscretization(std::shared_ptr<discretization> discr) {_discr=discr;}


    private:

    std::shared_ptr<discretization> _discr;
    int _nComponents;
    std::shared_ptr<gp::operators::laplacian> _laplacian;


};

}

#endif