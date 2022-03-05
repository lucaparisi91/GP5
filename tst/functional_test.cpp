#include "gtest/gtest.h"
#include "mpi.h"
#include "../src/geometry.h"
#include "../src/io.h"
#include "../src/tools.h"
#include "../src/functional.h"
#include "../src/externalPotential.h"

TEST(functional, gpFunctional)
{
    realDVec_t left {-1,-0.5,-0.5};
    realDVec_t right {1,0.5,0.5};
    intDVec_t N { 200, 100, 100};
    int nComponents=2;


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
    fftC.setProcessorGrid(intDVec_t{numProcs,1,1});


    auto fftOp = fftC.create();

    
    auto discr = fftOp->getDiscretizationRealSpace();


    Eigen::Tensor<real_t , 2> couplings(nComponents,nComponents);

    couplings.setConstant(0);
    
    auto laplacian = std::make_shared<gp::operators::laplacian>(fftOp);

    auto func=std::make_shared<gp::gpFunctional>();
    gp::harmonicPotential pot({  {1,1,1}, {1,1,1}});

    auto V = pot.create(discr,nComponents);
    
    func->setExternalPotential(V);
    func->setCouplings(  couplings  );
    func->setMasses({ 1, 1});
    func->setLaplacianOperator(laplacian);
    func->setDiscretization( discr );
    func->setNComponents( nComponents );
    func->init();
    
    const auto & shape= discr->getLocalMesh()->shape();


    tensor_t field(shape[0],shape[1],shape[2],nComponents);
    tensor_t field2(shape[0],shape[1],shape[2],nComponents);

    field.setConstant(0);
    field2.setConstant(0);
    real_t sigma1=0.05;
    real_t sigma2=0.02;

    gp::initGaussian( sigma1 , discr,field,0);
    gp::initGaussian( sigma2 , discr,field,1);

    func->apply(field,field2,0);
    
    
    save(field,"gaussian.hdf5",*discr);
    save(field2,"func.hdf5",*discr);

    func = NULL;
    fftOp=NULL;
    laplacian = NULL;



    p3dfft::cleanup();

}

