
#ifndef FOURIER_TRANSFORM_H
#define FOURIER_TRANSFORM_H


#include "traits.h"
#include "geometry.h"
#include "p3dfft.h"


namespace gp
{
    enum FFT_DIRECTION { FORWARD=0, BACKWARD=1 };

    template<class T1,class T2>
    class fourierTransform
    {
        public:

        using discretization_t = discretization;
        using mesh_t = mesh;
        using domain_t = domain;

        virtual void apply(  tensor_t & source , tensor_t &  destination, FFT_DIRECTION dir )=0;

        virtual std::shared_ptr<discretization_t> getDiscretizationRealSpace()=0;
        virtual std::shared_ptr<discretization_t> getDiscretizationFourierSpace()=0;
    };


    template<class T1,class T2>
    class p3dfftFourierTransform : public fourierTransform<T1,T2>
    {
        public:
        using discretization_t = discretization;
        using mesh_t = mesh;
        using domain_t = domain;

        
        p3dfftFourierTransform( std::shared_ptr<domain_t> & globalDomain, std::shared_ptr<mesh_t> & globalMesh, const intDVec_t & processors , MPI_Comm comm  );
        
        virtual void apply(  tensor_t & source , tensor_t &  destination, FFT_DIRECTION dir ) override;


        ~p3dfftFourierTransform();


        virtual std::shared_ptr< discretization_t > getDiscretizationRealSpace() override {return _discr;}
        virtual std::shared_ptr< discretization_t > getDiscretizationFourierSpace() override {return _discr2;}

        private:

        std::shared_ptr< discretization_t > _discr;
        std::shared_ptr< discretization_t > _discr2;


        MPI_Comm _comm;
        int trans_f;
        int trans_b;
        Grid* Xpencil;
        Grid* Zpencil;
        int Pgrid;


    };


    template<class T1,class T2>
    class fftwFourierTransform  : public fourierTransform<T1,T2>
    {
        public:

        using discretization_t = discretization;
        using mesh_t = mesh;
        using domain_t = domain;

        
        fftwFourierTransform( std::shared_ptr<discretization_t> discr, int nComponents  );

        virtual void apply(  tensor_t & source , tensor_t &  destination, FFT_DIRECTION dir );

        ~fftwFourierTransform();

        virtual std::shared_ptr< discretization_t > getDiscretizationRealSpace() override {return _discr;}
        virtual std::shared_ptr< discretization_t > getDiscretizationFourierSpace() override {return _discr2;}


        private:

        void init(tensor_t & field);

        std::shared_ptr<discretization_t> _discr;
        std::shared_ptr<discretization_t> _discr2;
        fftw_plan planForward;
        fftw_plan planBackward;

    };



    template<class T1,class T2>
    class fourierTransformCreator
    {
        public:

        fourierTransformCreator();

        std::shared_ptr<fourierTransform<T1,T2> > create();

        void setDomain(std::shared_ptr<domain> newDomain ) {_domain=newDomain;}

        void setGlobalMesh(std::shared_ptr<mesh> newMesh ) {_globalMesh=newMesh;}


        void setNComponents(int newNComponents ) {_nComponents=newNComponents;}

        void setProcessorGrid(const intDVec_t & grid ) {_processorGrid=grid;}

        void setCommunicator( const MPI_Comm & comm_) {_comm=comm_;}

        private:

        std::shared_ptr<domain> _domain;
        std::shared_ptr<mesh> _globalMesh;
        intDVec_t _processorGrid;
        int _nComponents;

        MPI_Comm _comm;

        
    };

};

#endif