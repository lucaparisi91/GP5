#include "tools.h"
#include <iostream>
#include <complex>

template<>
MPI_Datatype toMPIDataType<double>::type = MPI_DOUBLE;

template<>
MPI_Datatype toMPIDataType<std::complex<double> >::type = MPI_DOUBLE_COMPLEX ;

namespace gp
{    

}