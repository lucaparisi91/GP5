#include "geometry.h"

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