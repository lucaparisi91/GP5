
#include "traits.h"
#include "geometry.h"
#include "p3dfft.h"


namespace gp
{
    enum FFT_DIRECTION { FORWARD=0, BACKWARD=1 };


    template<class T1,class T2>
    class p3dfftFourierTransform
    {
        public:

        using discretization_t = discretization;
        using mesh_t = mesh;
        using domain_t = domain;
        


        
        p3dfftFourierTransform( std::shared_ptr<domain_t> & globalDomain, std::shared_ptr<mesh_t> & globalMesh, const intDVec_t & processors , MPI_Comm comm  );
        
        virtual void apply(  tensor_t & source , tensor_t &  destination, FFT_DIRECTION dir );


        ~p3dfftFourierTransform();

        auto getDiscretizationRealSpace() {return _discr;}
        auto getDiscretizationFourierSpace() {return _discr2;}


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
    class fftwFourierTransform
    {
        public:

        using discretization_t = discretization;
        using mesh_t = mesh;
        using domain_t = domain;
        
        fftwFourierTransform( std::shared_ptr<discretization_t> discr, int nComponents  );

        virtual void apply(  tensor_t & source , tensor_t &  destination, FFT_DIRECTION dir );

        ~fftwFourierTransform();


        auto getDiscretizationRealSpace() {return _discr;}
        auto getDiscretizationFourierSpace() {return _discr2;}


        private:

        void init(tensor_t & field);

        std::shared_ptr<discretization_t> _discr;
        std::shared_ptr<discretization_t> _discr2;
        fftw_plan planForward;
        fftw_plan planBackward;

    };

};