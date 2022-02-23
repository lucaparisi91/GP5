#include "geometry.h"
#include <iostream>
namespace gp{
    mesh::mesh( const intDVec_t & N) :
    _N(N)
    {
        _size=1;
        for(int d=0;d<DIMENSIONS;d++)
        {
            
            _size*=_N[d];

            _nGhosts[d]=0;

            _offset[d]=0;
        }

    }



    domain::domain() :
    _left{0,0,0}, _right{0,0,0},_lBox{0,0,0}
    {

    }

    domain::domain( realDVec_t left,realDVec_t right) :
    _left(left),_right(right)
    {
        updateLBox();
    }

    void domain::updateLBox()
    {
        for(int d=0;d<DIMENSIONS;d++)
        {
            _lBox[d]=_right[d] - _left[d];
        }
    }

    discretization::discretization()
    {

    }

    std::shared_ptr<discretization> createUniformDiscretization( std::shared_ptr<domain> globalDomain, std::shared_ptr<mesh> & globalMesh, intDVec_t processorGrid, MPI_Comm & comm)
    {

        auto discr=std::make_shared<discretization>();
        discr->setDomain(globalDomain);
        discr->setGlobalMesh(globalMesh);

        int num_procs,rank;
        MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
        int pgrid[] = { processorGrid[0], processorGrid[1],processorGrid[2] };
        int periods[] = { 1, 1, 1};
        int coords [ ] = {-1, -1 ,-1};


        MPI_Comm newComm;

        auto status = MPI_Cart_create( comm, DIMENSIONS, pgrid, periods,false, &newComm);

        status =  MPI_Cart_coords( newComm, rank, DIMENSIONS, coords);

        intDVec_t localShape{0,0,0};
        intDVec_t offset{0,0,0};
        
        for(int d=0;d<DIMENSIONS;d++)
        {
            localShape[d]=globalMesh->shape()[d] / processorGrid[d];
            offset[d]=localShape[d] * coords[d];

            int remainderCells=globalMesh->shape()[d] % processorGrid[d];

            if (rank < remainderCells)
            {
                localShape[d]+=1;
                offset[d]+=rank;
            }
            else
            {
                offset[d]+=remainderCells;
            }
        }

    
        auto localMesh = std::make_shared<gp::mesh>( localShape );
        localMesh->setGlobalOffset(offset);

        discr->setLocalMesh(localMesh);
        discr->setCommunicator(newComm);

        return discr;
    }

    
tensor_t momentums(std::shared_ptr<discretization> discr,int d, int nComponents)
{
    auto delta = discr->getFourierStep( d );
    const auto & globalShape = discr->getGlobalMesh()->shape();
    const auto & shape = discr->getLocalMesh()->shape();
    const auto & offset = discr->getLocalMesh()->getGlobalOffset();

    tensor_t K( shape[0],shape[1],shape[2],nComponents);

    for(int i=0;i<shape[0];i++)
        for(int j=0;j<shape[1];j++)
            for(int k=0;k<shape[2];k++)
                {
                    intDVec_t index{i,j,k};
                    
                    int iG = index[d] + offset[d];
                    for(int iC=0;iC<nComponents;iC++)
                    {
                        if ( globalShape[d] % 2 == 0)
                        {
                            if (iG <globalShape[d]/2)
                            {
                                K(i,j,k,iC)=delta * iG;
                            }
                            else
                            {
                                K(i,j,k,iC)=(-globalShape[d] + iG)* delta;
                            }
                        }
                        else
                        {
                            if (iG <=(globalShape[d]-1)/2)
                            {
                                K(i,j,k,iC)=delta * iG;
                            }
                            else
                            {
                                K(i,j,k,iC)=(-globalShape[d] + iG)* delta;
                            }
                        }
                    }
                }
    return K;
}












tensor_t positions(std::shared_ptr<discretization> discr,int d, int nComponents)
{
    auto delta = discr->getSpaceStep( d );
    const auto & left = discr->getDomain()->getLeft();
    
    const auto & globalShape = discr->getGlobalMesh()->shape();
    const auto & shape = discr->getLocalMesh()->shape();
    const auto & offset = discr->getLocalMesh()->getGlobalOffset();

    tensor_t X( shape[0],shape[1],shape[2],nComponents);

    for(int i=0;i<shape[0];i++)
        for(int j=0;j<shape[1];j++)
            for(int k=0;k<shape[2];k++)
                {
                    intDVec_t index{i,j,k};
                    int iG = index[d] + offset[d];
                    for(int iC=0;iC<nComponents;iC++)
                    {  
                        X(i,j,k,iC)=left[d] + delta *( iG + 0.5);   
                    }
                }
    return X;
}













};