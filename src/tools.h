#ifndef TOOLS_H
#define TOOLS_H

#include "geometry.h"

namespace gp
{
void initGaussian(real_t sigma, std::shared_ptr<discretization> discr, tensor_t & tensor, int comp);

}

#endif
