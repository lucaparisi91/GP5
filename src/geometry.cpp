#include "geometry.h"
#include <iostream>
#include "tools.h"

namespace gp{

    mesh::mesh( const intDVec_t & N) :
    _N(N)
    {
        for(int d=0;d<DIMENSIONS;d++)
        {
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

    std::shared_ptr<domain> domain::permute( const  intDVec_t & order) const 
    {
        realDVec_t left2,right2;
        for(int i=0;i<DIMENSIONS;i++)
        {
            left2[  order[i] ] = _left[ i ];
            right2[ order[i] ] = _right[ i ];
        }

        return std::make_shared<domain>(left2,right2);
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

real_t norm( const tensor_t & field, int c, std::shared_ptr<discretization> & discr )
{
    const auto & dimensions= field.dimensions();
    real_t sum=0;
    for(int i=0;i<dimensions[0];i++)
        for(int j=0;j<dimensions[1];j++)
            for(int k=0;k<dimensions[2];k++)
                {
                    sum+=field(i,j,k,c).real()*field(i,j,k,c).real() ;
                    sum+=field(i,j,k,c).imag()*field(i,j,k,c).imag() ;
                }
    
    real_t sumReduce=0;
    
    MPI_Allreduce(&sum, &sumReduce, 1,  toMPIDataType<real_t>::type, MPI_SUM, discr->getCommunicator() );    

    real_t dV=1;
    for(int d=0;d<DIMENSIONS;d++)
    {
        dV*=discr->getSpaceStep(d);
    }

    return sumReduce*dV;

}

void normalize( real_t N, tensor_t & field, int c, std::shared_ptr<discretization> discr)
{
    auto normInverseSqrt =sqrt( N  * 1./norm(field,c,discr) );
    const auto & dimensions=field.dimensions();

    for(int k=0;k<dimensions[2];k++)
         for(int j=0;j<dimensions[1];j++)
            for(int i=0;i<dimensions[0];i++)
                {
                    field(i,j,k,c)*=value_t(normInverseSqrt,0);
                }
                
}

void initGaussian( real_t sigma, std::shared_ptr<discretization> discr, tensor_t & tensor, int comp)
{
    return initGaussian( {sigma,sigma,sigma},discr,tensor,comp);
}

void initGaussian( const realDVec_t & sigma, std::shared_ptr<discretization> discr, tensor_t & field, int comp)
{
    const auto & shape= discr->getLocalMesh()->shape();
    const auto & offset= discr->getLocalMesh()->getGlobalOffset();

    const auto & left = discr->getDomain()->getLeft();
    const auto & right = discr->getDomain()->getRight();

    realDVec_t deltax;
    for(int d=0;d<DIMENSIONS;d++)
    {
        deltax[d]=discr->getSpaceStep(d);
    }
    
    realDVec_t alpha={1./(2*sigma[0]*sigma[0])   ,1./(2*sigma[1]*sigma[1]), 1./(2*sigma[2]*sigma[2])   };

    for(int i=0;i<shape[0];i++)
        for(int j=0;j<shape[1];j++)
            for(int k=0;k<shape[2];k++)
            {
                real_t x = left[0] + (i + offset[0] +0.5)*deltax[0];
                real_t y = left[1] + (j + offset[1] +0.5)*deltax[1];
                real_t z = left[2] + (k + offset[2] +0.5)*deltax[2];

                
                real_t tmp = alpha[0]*x*x + alpha[1]*y*y + alpha[2]*z*z;

                field(i,j,k,comp)=complex_t(exp(-tmp),0) ;
            }
              
}

std::shared_ptr<mesh> mesh::permute( const intDVec_t & ordering ) const
{
    intDVec_t N2, offset2;
    for(int d=0;d<DIMENSIONS;d++)
    {
        offset2[ordering[d] ]=_offset[d];
        N2[ordering[d] ]=_N[d ];
    }

    auto mesh2 = std::make_shared<mesh>( N2 );
    mesh2->setGlobalOffset(offset2);

    return mesh2;
}





};