#include "gtest/gtest.h"
#include "mpi.h"

#include "../src/geometry.h"
#include "../src/io.h"
#include "../src/tools.h"

TEST(io,io)
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
    
    gp::initGaussian( 0.1 , discr,field,0);


    save(field,"field.hdf5",*discr);

    auto field2 = load("field.hdf5",*discr,1);


    for(int i=0;i<shape[0];i++)
        for(int j=0;j<shape[1];j++)
            for(int k=0;k<shape[2];k++)
    {
        ASSERT_NEAR( field(i,j,k,0).real(),field2(i,j,k,0).real(),TOL);
    }


}

