#include "fourierTransform.h"

namespace gp
{

    template<class T1, class T2>
    p3dfftFourierTransform<T1,T2>::p3dfftFourierTransform( std::shared_ptr<domain_t> & globalDomain, std::shared_ptr<mesh_t> & globalMesh, const intDVec_t & processors , MPI_Comm comm, int nComponents  )
        {
        this->setNComponents(nComponents);

        _discr=std::make_shared<discretization_t>();
        _discr->setDomain(globalDomain);
        _discr->setGlobalMesh(globalMesh);
        _discr->setCommunicator(comm);

        const auto & globalShape = globalMesh->shape() ;

        _discr2=std::make_shared<discretization_t>();
        
        _discr2->setGlobalMesh(globalMesh);
        _discr2->setDomain(globalDomain);
        _discr2->setCommunicator(comm);
        
        int type_ids1[3] = { p3dfft::CFFT_FORWARD_D,p3dfft::CFFT_FORWARD_D,p3dfft::CFFT_FORWARD_D};
        int type_ids2[3] = {p3dfft::CFFT_BACKWARD_D,p3dfft::CFFT_BACKWARD_D,p3dfft::CFFT_BACKWARD_D};

        int pdims[] = { processors[0] , processors[1] , processors[2] };
        int mem_order1[] = {0,1,2};
        int mem_order2[] = {0,1,2};
        

        int dmap1[] = {0,1,2};
        int dmap2[] = {0,1,2};
        

        int gdims[] = { globalShape[0] , globalShape[1] , globalShape[2] };


       

        auto type_forward = p3dfft_init_3Dtype(type_ids1);

        auto type_backward = p3dfft_init_3Dtype(type_ids2);
        
         Pgrid = p3dfft_init_proc_grid(pdims,comm);


        Xpencil = p3dfft_init_data_grid(gdims,-1,Pgrid,dmap1,mem_order1);
        
        
        Zpencil = p3dfft_init_data_grid(gdims,-1,Pgrid,dmap2,mem_order2);
        
        

        trans_f = p3dfft_plan_3Dtrans(Xpencil,Zpencil,type_forward);
        trans_b = p3dfft_plan_3Dtrans(Zpencil,Xpencil,type_backward);

        intDVec_t offset,localShape,offset2,localShape2;

        for( int i=0;i<DIMENSIONS;i++) {
            offset[ mem_order1[i] ] = Xpencil->GlobStart[i];
            localShape[ mem_order1[i] ] = Xpencil->Ldims[i];
            offset2[ mem_order2[i] ] = Zpencil->GlobStart[i];
            localShape2[ mem_order2[i] ] = Zpencil->Ldims[i];

            //std::cout << i << " "<<localShape2[i] << std::endl;

        }


        auto localMesh=std::make_shared<mesh_t>( localShape);
        auto localMesh2=std::make_shared<mesh_t>( localShape2);

        localMesh->setGlobalOffset(offset);
        localMesh2->setGlobalOffset(offset2);

        _discr->setLocalMesh(localMesh);
        _discr2->setLocalMesh(localMesh2);
        }

    template<class T1,class T2>
    p3dfftFourierTransform<T1,T2>::~p3dfftFourierTransform()
    {
        p3dfft_free_data_grid(Xpencil);
        p3dfft_free_data_grid(Zpencil);
        p3dfft_free_proc_grid(Pgrid);

    }

    template<class T1,class T2>
    void p3dfftFourierTransform<T1,T2>::apply(tensor_t & field,tensor_t & field2,FFT_DIRECTION direction)
    {
        auto fftSize=getDiscretizationFourierSpace()->getLocalMesh()->size();
        auto realSize=getDiscretizationRealSpace()->getLocalMesh()->size();

        if (direction == FFT_DIRECTION::FORWARD)
         {
             for(int iComp=0;iComp<this->getNComponents();iComp++)
            {
                p3dfft_exec_3Dtrans_double(trans_f,( (real_t * )field.data() ) + iComp*realSize*2 ,( (real_t *)field2.data() ) + iComp*fftSize*2,0);
            }
             
         }
        else if (direction == FFT_DIRECTION::BACKWARD)
        {
             for(int iComp=0;iComp<this->getNComponents();iComp++)
            {
                p3dfft_exec_3Dtrans_double(trans_b,( (real_t * )field.data() ) + iComp*realSize*2 ,( (real_t *)field2.data() ) + iComp*fftSize*2,0);
            }
           
        }
        else
        {
            throw std::runtime_error("Unkown FFT direction");
        }        

    }



template<class T1,class T2>
fftwFourierTransform<T1,T2>::fftwFourierTransform( std::shared_ptr<discretization_t> discr , int nComponents ) :
_discr(discr),
_discr2(discr)
{
    this->setNComponents(nComponents);

    const auto & shape=_discr->getLocalMesh()->shape();

    tensor_t initField( shape[0] , shape[1], shape[2] ,nComponents  );

    initField.setConstant(0);

    init(initField);
    
}

fftw_plan createC2CFFTWPlan( tensor_t & spatialData , tensor_t & fourierData , FFT_DIRECTION direction  )
{
    const auto & dimensions = spatialData.dimensions();
    
    int nComponents=1;
    intDVec_t NT { dimensions[2],dimensions[1],dimensions[0] };

    int dir;

    if (direction == FFT_DIRECTION::FORWARD)
    {
        dir=FFTW_FORWARD;
    }
    else
    {
        dir=FFTW_BACKWARD;
    }
    
    auto dis = NT[0]*NT[1]*NT[2];

    auto fftPlan= fftw_plan_many_dft(DIMENSIONS, NT.data(), nComponents ,
    (fftw_complex*)spatialData
    .data(), NT.data(),
    1 , dis ,
    (fftw_complex*)fourierData.data(), NT.data(),
    1, dis ,dir , FFTW_MEASURE );

    return fftPlan;
}

template<class T1, class T2>
void fftwFourierTransform<T1,T2>::init( tensor_t & field  )
{
    planForward= createC2CFFTWPlan(field, field, FFT_DIRECTION::FORWARD);
    planBackward= createC2CFFTWPlan(field,field, FFT_DIRECTION::BACKWARD);  
};

template<class T1,class T2>
void fftwFourierTransform<T1,T2>::apply(   tensor_t & source , tensor_t &  destination, FFT_DIRECTION dir )
{
    auto sizeReal = getDiscretizationRealSpace()->getLocalMesh()->size();
    auto sizeFFT = getDiscretizationFourierSpace()->getLocalMesh()->size();

    if (dir==FFT_DIRECTION::FORWARD)
    {
        for(int iComp=0;iComp<this->getNComponents();iComp++)
        {
            fftw_execute_dft(planForward,((fftw_complex*)source.data() ) + iComp*sizeReal,((fftw_complex*)destination.data() ) + iComp*sizeFFT );
        }
        
    }
    else if (dir==FFT_DIRECTION::BACKWARD)
    {
         for(int iComp=0;iComp<this->getNComponents();iComp++)
        {
            fftw_execute_dft(planBackward,((fftw_complex*)source.data() ) + iComp*sizeFFT,((fftw_complex*)destination.data() ) + iComp*sizeReal );
        }
    }
    else
    {
        throw std::runtime_error("Unkown direction in fourier transform");
    }

};


template<class T1,class T2>
fftwFourierTransform<T1,T2>::~fftwFourierTransform()
    {
        fftw_destroy_plan(planForward);
        fftw_destroy_plan(planBackward);
    }




template<class T1,class T2>
std::shared_ptr<fourierTransform<T1,T2> > fourierTransformCreator<T1,T2>::create()
    {
        int numProcs,rank;

        MPI_Comm_size (_comm, &numProcs);
        MPI_Comm_rank (_comm, &rank);

        if (numProcs == 1)
        {
            auto discrReal = createUniformDiscretization( _domain, _globalMesh ,_processorGrid,_comm);
            auto fftOp=std::make_shared<gp::fftwFourierTransform<T1,T2> >(discrReal, _nComponents);
            return fftOp;
        }
        else
        {
            auto fftOp=std::make_shared<gp::p3dfftFourierTransform<T1,T2>  >( _domain,_globalMesh,_processorGrid,_comm,_nComponents);

            return fftOp;
        }


    };

template<class T1,class T2>
fourierTransformCreator<T1,T2>::fourierTransformCreator()
{
    _domain = NULL;
    _globalMesh = NULL;
    _processorGrid = {-1,-1,-1};

    _comm = MPI_COMM_WORLD;
}

template class p3dfftFourierTransform<complex_t,complex_t> ;
template class fftwFourierTransform<complex_t,complex_t> ;
template class fourierTransformCreator<complex_t,complex_t>;

};
