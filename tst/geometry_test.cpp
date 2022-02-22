#include "gtest/gtest.h"
#include "mpi.h"

#include "../src/geometry.h"

TEST( geometry, mesh )
{
    std::array<int,DIMENSIONS> shape { 100,150,180};
    gp::mesh mesh ( shape);
    
    ASSERT_EQ( mesh.size() , shape[0]*shape[1]*shape[2] );
    ASSERT_EQ( mesh.shape()[0] , shape[0] );
    ASSERT_EQ( mesh.shape()[1] , shape[1] );
    ASSERT_EQ( mesh.shape()[2] , shape[2] );

    ASSERT_EQ( mesh.getGlobalOffset()[0] , 0);
    ASSERT_EQ( mesh.getGlobalOffset()[1] , 0);
    ASSERT_EQ( mesh.getGlobalOffset()[2] , 0);
    
}

TEST( geometry, domain )
{
    gp::domain domain({ -0.5,-0.5,-0.5},{0.5,0.5,0.5} );

    ASSERT_EQ( domain.getLBox()[0], 1 );
    ASSERT_EQ( domain.getLBox()[1], 1 );
    ASSERT_EQ( domain.getLBox()[2], 1 );

}
TEST( geometry, discretization )
{
    realDVec_t left {-0.5,-0.5,-0.5};
    realDVec_t right {0.5,0.5,0.5};
    intDVec_t N {100,100,100};

    
    auto domain = std::make_shared<gp::domain>( left,right );

    auto globalMesh = std::make_shared<gp::mesh>( N);

    gp::discretization discretization {};

    int num_procs,rank;
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    discretization.setGlobalMesh(globalMesh);
    discretization.setDomain(domain);
    discretization.setCommunicator(MPI_COMM_WORLD);
    ASSERT_EQ( discretization.getSpaceStep(0) , 0.01 );

    if (num_procs == 1)
    {
        discretization.setLocalMesh(globalMesh);
        ASSERT_EQ(discretization.getLocalMesh()->shape()[0],100);
    }

    if (num_procs == 2)
    {
        auto localMesh = std::make_shared<gp::mesh>(intDVec_t{50,100,100});

        if (rank == 0)
        {
            localMesh->setGlobalOffset( {0,0,0});
            discretization.setLocalMesh(localMesh);
            ASSERT_EQ(discretization.getLocalMesh()->getGlobalOffset()[0],0);
            
        }
        else if (rank == 1)
        {

            localMesh->setGlobalOffset( {50,0,0});
            discretization.setLocalMesh(localMesh);
            ASSERT_EQ(discretization.getLocalMesh()->getGlobalOffset()[0],50);

        }

        ASSERT_EQ(discretization.getLocalMesh()->shape()[0],50);

        discretization.setLocalMesh(localMesh);
    }
    
}

TEST( geometry, createDiscretizationUniform )
{

    realDVec_t left {-0.5,-0.5,-0.5};
    realDVec_t right {0.5,0.5,0.5};
    intDVec_t N { 100, 100, 100};

    auto domain = std::make_shared<gp::domain>( left,right );
    auto globalMesh = std::make_shared<gp::mesh>( N);
    MPI_Comm comm(MPI_COMM_WORLD);



    
    int numProcs, rank;
    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    if (numProcs  == 1)
    {
        auto discr = createUniformDiscretization( domain, globalMesh ,intDVec_t{1,1,1}, comm);
        
        for (int d=0;d<DIMENSIONS;d++)
        {
            ASSERT_EQ(N[d],discr->getLocalMesh()->shape()[d]);
            ASSERT_EQ(N[d],discr->getGlobalMesh()->shape()[d]);
            
        }
        
    }

    if (numProcs  == 2)
    {
        auto discr = createUniformDiscretization( domain, globalMesh ,intDVec_t{2,1,1}, comm);
        

        for (int d=1;d<DIMENSIONS;d++)
        {
            ASSERT_EQ(N[d],discr->getLocalMesh()->shape()[d]);
            ASSERT_EQ(N[d],discr->getGlobalMesh()->shape()[d]);  
            ASSERT_EQ(0,discr->getLocalMesh()->getGlobalOffset()[d]);
             
        }

        if (rank == 1)
        {
            ASSERT_EQ(N[0]/2,discr->getLocalMesh()->getGlobalOffset()[0]);
        }
        if (rank == 0)
        {
            ASSERT_EQ(0,discr->getLocalMesh()->getGlobalOffset()[0]);
        }
        

        ASSERT_EQ(N[0]/2,discr->getLocalMesh()->shape()[0]);
        ASSERT_EQ(N[0],discr->getGlobalMesh()->shape()[0]);
        
    }

    if (numProcs  ==  3)
    {
        auto discr = createUniformDiscretization( domain, globalMesh ,intDVec_t{3,1,1}, comm);
        
        for (int d=1;d<DIMENSIONS;d++)
        {
            ASSERT_EQ(N[d],discr->getLocalMesh()->shape()[d]);
            ASSERT_EQ(N[d],discr->getGlobalMesh()->shape()[d]);
            
        }

        int coords[DIMENSIONS];

        auto status =  MPI_Cart_coords( discr->getCommunicator(), rank, DIMENSIONS, coords);

        ASSERT_EQ( coords[1],0);
        ASSERT_EQ( coords[2],0);
        
        if ( coords[0]==0)
        {
            ASSERT_EQ( 0, discr->getLocalMesh()->getGlobalOffset()[0]);

        }
        if ( coords[0]==1)
        {
            ASSERT_EQ( 34, discr->getLocalMesh()->getGlobalOffset()[0]);

        }
        if ( coords[0]==2)
        {
            ASSERT_EQ( 67, discr->getLocalMesh()->getGlobalOffset()[0]);
        }


        ASSERT_EQ(N[0],discr->getGlobalMesh()->shape()[0]);
        
    }
        
    

}




