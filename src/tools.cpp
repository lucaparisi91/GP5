#include "tools.h"
#include <iostream>
namespace gp
{

void initGaussian(real_t sigma, std::shared_ptr<discretization> discr, tensor_t & field, int comp)
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
    
    real_t alpha=1./(2*sigma*sigma);
    for(int i=0;i<shape[0];i++)
        for(int j=0;j<shape[1];j++)
            for(int k=0;k<shape[2];k++)
            {
                real_t x = left[0] + (i + offset[0] +0.5)*deltax[0];
                real_t y = left[1] + (j + offset[1] +0.5)*deltax[1];
                real_t z = left[2] + (k + offset[2] +0.5)*deltax[2];

                real_t r2 = x*x + y*y + z*z;
                field(i,j,k,comp)=complex_t(exp(-alpha*r2),0) ;
            }
              
}


}