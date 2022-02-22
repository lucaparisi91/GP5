#ifndef IO_H
#define IO_H

#include "traits.h"
#include "geometry.h"

namespace gp
{

    void save(const tensor_t & field , const std::string & filename, const discretization & discr );
    
    tensor_t load( const std::string & filename, const discretization & discr, int nComponents);


    
};


#endif