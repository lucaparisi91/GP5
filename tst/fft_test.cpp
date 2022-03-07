#include "gtest/gtest.h"
#include "mpi.h"
#include "../src/geometry.h"
#include "../src/io.h"
#include "../src/tools.h"
#include "../src/fourierTransform.h"
#include "../src/operators.h"


TEST(fft,forward_backward)
{
    realDVec_t left { -0.5,-0.5,-0.5};
    realDVec_t right { 0.5,0.5,0.5};
    intDVec_t N { 50, 170 , 150};

    p3dfft::setup();


    auto domain = std::make_shared<gp::domain>( left,right );
    auto globalMesh = std::make_shared<gp::mesh>( N);
    MPI_Comm comm(MPI_COMM_WORLD);

    int numProcs,rank;

    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    gp::fourierTransformCreator<complex_t,complex_t> fftC;
    fftC.setNComponents(1);
    fftC.setCommunicator(MPI_COMM_WORLD);
    fftC.setDomain(domain);
    fftC.setGlobalMesh(globalMesh);
    fftC.setProcessorGrid(intDVec_t{1,1,numProcs});


    auto fftOp = fftC.create();

    auto discr = fftOp->getDiscretizationRealSpace();
    auto discr2 = fftOp->getDiscretizationFourierSpace();

    const auto & shape= discr->getLocalMesh()->shape();
    const auto & offset= discr->getLocalMesh()->getGlobalOffset();

    const auto & shape2= discr2->getLocalMesh()->shape();
    const auto & offset2= discr2->getLocalMesh()->getGlobalOffset();


    realDVec_t deltax;
    for(int d=0;d<DIMENSIONS;d++)
    {
        deltax[d]=discr->getSpaceStep(d);
    }


    tensor_t field(shape[0],shape[1],shape[2],1);
    tensor_t field2(shape2[0],shape2[1],shape2[2],1);
    tensor_t field3(shape[0],shape[1],shape[2],1);


    field.setConstant(0);
    field2.setConstant(1);
    
    gp::initGaussian( {1,0.5,0.2} , discr,field,0);
    
    fftOp->apply(field, field2,gp::FFT_DIRECTION::FORWARD);

    fftOp->apply(field2, field3,gp::FFT_DIRECTION::BACKWARD);

    field3=field3* (1./complex_t(N[0] * N[1] * N[2],0) );

    for(int i=0;i<shape[0];i++)
        for(int j=0;j<shape[1];j++)
            for(int k=0;k<shape[2];k++)
    {
        ASSERT_NEAR( field(i,j,k,0).real(),field3(i,j,k,0).real(),TOL);
        ASSERT_NEAR( field(i,j,k,0).imag(),field3(i,j,k,0).imag(),TOL);
        
    }

    fftOp=NULL;

    p3dfft::cleanup();

}


TEST(fft, derivative)
{
    realDVec_t left {-0.5,-0.5,-0.5};
    realDVec_t right {0.5,0.5,0.5};
    intDVec_t N { 300, 250, 77};
    
    p3dfft::setup();

    auto domain = std::make_shared<gp::domain>( left,right );
    auto globalMesh = std::make_shared<gp::mesh>( N);
    MPI_Comm comm(MPI_COMM_WORLD);

    int numProcs,rank;

    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    gp::fourierTransformCreator<complex_t,complex_t> fftC;
    fftC.setNComponents(1);
    fftC.setCommunicator(MPI_COMM_WORLD);
    fftC.setDomain(domain);
    fftC.setGlobalMesh(globalMesh);
    fftC.setProcessorGrid(intDVec_t{1,1,numProcs});
    fftC.setOrdering({2,0,1});

    auto fftOp = fftC.create();

    auto laplacian = std::make_shared<gp::operators::laplacian>(fftOp);
    
    auto discrReal = fftOp->getDiscretizationRealSpace();
    
    
    const auto & shapeReal= discrReal->getLocalMesh()->shape();


    tensor_t field(shapeReal[0],shapeReal[1],shapeReal[2],1);
    tensor_t fieldL(shapeReal[0],shapeReal[1],shapeReal[2],1);

    field.setConstant(0);
    fieldL.setConstant(0);
    real_t sigma=0.05;
    gp::initGaussian( sigma , discrReal,field,0);

    laplacian->apply(field, fieldL);

    save(field,"gaussian.hdf5",*discrReal);
    save(fieldL,"gaussianL.hdf5",*discrReal);



    auto X=positions(discrReal,0,1);
    auto Y=positions(discrReal,1,1);
    auto Z=positions(discrReal,2,1);

    auto R2 = X*X + Y*Y + Z*Z;
    complex_t alpha=1/(2*sigma*sigma) + 1j*0;

    auto L2 = (-alpha*R2 ).exp() * (-R2*alpha*complex_t(2,0) + complex_t(3,0)) * complex_t(-2,0)*alpha;
    
    Eigen::Tensor<double,0> max = (L2 - fieldL).abs().maximum(); 
    ASSERT_LE( max()  , TOL);

   
    laplacian = NULL;
    fftOp=NULL;

    p3dfft::cleanup();

}


TEST(fft, derivative_multipleComponents)
{
    realDVec_t left {-1,-0.5,-0.5};
    realDVec_t right {1,0.5,0.5};
    intDVec_t N { 200, 100, 100};
    int nComponents = 2;

    p3dfft::setup();

    auto domain = std::make_shared<gp::domain>( left,right );
    auto globalMesh = std::make_shared<gp::mesh>( N);
    MPI_Comm comm(MPI_COMM_WORLD);

    int numProcs,rank;

    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    gp::fourierTransformCreator<complex_t,complex_t> fftC;
    fftC.setNComponents(nComponents);
    fftC.setCommunicator(MPI_COMM_WORLD);
    fftC.setDomain(domain);
    fftC.setGlobalMesh(globalMesh);
    fftC.setProcessorGrid(intDVec_t{1,1,numProcs});
    

    auto fftOp = fftC.create();

    auto laplacian = std::make_shared<gp::operators::laplacian>(fftOp);

    auto discrReal = fftOp->getDiscretizationRealSpace();
    
    
    const auto & shapeReal= discrReal->getLocalMesh()->shape();

    tensor_t field(shapeReal[0],shapeReal[1],shapeReal[2],nComponents);
    tensor_t fieldL(shapeReal[0],shapeReal[1],shapeReal[2],nComponents);

    field.setConstant(0);
    fieldL.setConstant(0);
    real_t sigma1=0.05;
    real_t sigma2=0.02;

    gp::initGaussian( sigma1 , discrReal,field,0);
    gp::initGaussian( sigma2 , discrReal,field,1);



    laplacian->apply(field, fieldL);


    save(field,"gaussian.hdf5",*discrReal);
    save(fieldL,"gaussianL.hdf5",*discrReal);

    laplacian = NULL;
    fftOp=NULL;

    p3dfft::cleanup();


}
