#ifndef TRAITS_H
#define TRAITS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <array>
#include <complex>
#define DIMENSIONS 3
#include <mpi.h>

using real_t = double;

using realDVec_t = std::array<real_t,DIMENSIONS>;
using intDVec_t = std::array<int,DIMENSIONS>;
using sizeDVec_t = std::array<size_t,DIMENSIONS>;


using complex_t = std::complex<real_t>;
using tensor_t = Eigen::Tensor<complex_t,DIMENSIONS + 1 > ;


struct geometry;

using geometry_t = geometry;

#if DIMENSIONS==1
    #define TRUNCATE_D(a,b,c) a
    #define EXPAND_D(a ) a[0]
    
#endif

#if DIMENSIONS==2
    #define TRUNCATE_D(a,b,c) a,b
    #define EXPAND_D(a ) a[0],a[1]
    
#endif

#if DIMENSIONS==3
    #define TRUNCATE_D(a,b,c) a,b,c
    #define EXPAND_D(a ) a[0],a[1],a[2]

    
#endif


#endif


