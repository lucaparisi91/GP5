#include "gtest/gtest.h"
#include "mpi.h"

#include "../src/geometry.h"
#include "../src/io.h"
#include "../src/tools.h"
#include "../src/fourierTransform.h"



TEST(fft,fftw)
{
    realDVec_t left {-0.5,-0.5,-0.5};
    realDVec_t right {0.5,0.5,0.5};
    intDVec_t N { 100, 100, 100};

    auto domain = std::make_shared<gp::domain>( left,right );
    auto globalMesh = std::make_shared<gp::mesh>( N);
    MPI_Comm comm(MPI_COMM_WORLD);

    int numProcs,rank;

    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    auto discr = createUniformDiscretization( domain, globalMesh ,intDVec_t{numProcs,1,1}, comm);

    const auto & shape= discr->getLocalMesh()->shape();
    const auto & offset= discr->getLocalMesh()->getGlobalOffset();

    realDVec_t deltax;
    for(int d=0;d<DIMENSIONS;d++)
    {
        deltax[d]=discr->getSpaceStep(d);
    }



    tensor_t field(shape[0],shape[1],shape[2],1);
    tensor_t field2(shape[0],shape[1],shape[2],1);
    tensor_t field3(shape[0],shape[1],shape[2],1);
    

    field.setConstant(0);
    field2.setConstant(1);

    gp::initGaussian( 1 , discr,field,0);

    field3=field;

    if (numProcs == 1)
    {
        gp::fftwFourierTransform<complex_t,complex_t> fftOp(discr,1);

        fftOp.apply(field, field2,gp::FFT_DIRECTION::FORWARD);

        fftOp.apply(field2, field3,gp::FFT_DIRECTION::BACKWARD);

        field3=field3* (1./complex_t(N[0] * N[1] * N[2],0) );

        for(int i=0;i<shape[0];i++)
            for(int j=0;j<shape[1];j++)
                for(int k=0;k<shape[2];k++)
        {
            ASSERT_NEAR( field(i,j,k,0).real(),field3(i,j,k,0).real(),TOL);
        }

    }

}



TEST(fft,fftw_derivative)
{
    realDVec_t left {-0.5,-0.5,-0.5};
    realDVec_t right {0.5,0.5,0.5};
    intDVec_t N { 100, 100, 100};

    auto domain = std::make_shared<gp::domain>( left,right );
    auto globalMesh = std::make_shared<gp::mesh>( N);
    MPI_Comm comm(MPI_COMM_WORLD);

    int numProcs,rank;

    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    auto discr = createUniformDiscretization( domain, globalMesh ,intDVec_t{numProcs,1,1}, comm);

    const auto & shape= discr->getLocalMesh()->shape();
    const auto & offset= discr->getLocalMesh()->getGlobalOffset();

    tensor_t field(shape[0],shape[1],shape[2],1);
    tensor_t fieldK(shape[0],shape[1],shape[2],1);
    tensor_t fieldL(shape[0],shape[1],shape[2],1);

    field.setConstant(0);
    fieldK.setConstant(0);
    fieldL.setConstant(0);

    auto KX = momentums(discr,0,1);
    auto KY = momentums(discr,1,1);
    auto KZ = momentums(discr,2,1);


    tensor_t K2 = KX*KX + KY*KY + KZ*KZ;

    gp::initGaussian( 0.1 , discr,field,0);

    if (numProcs == 1)
    {
        gp::fftwFourierTransform<complex_t,complex_t> fftOp(discr,1);

        fftOp.apply(field, fieldK,gp::FFT_DIRECTION::FORWARD);
        fieldK=-fieldK*K2;

        fftOp.apply(fieldK, fieldL, gp::FFT_DIRECTION::BACKWARD);

        fieldL=fieldL* (1./complex_t(N[0] * N[1] * N[2],0) );

    }

    save(field,"gaussian.hdf5",*discr);
    save(fieldL,"gaussianL.hdf5",*discr);
    

}