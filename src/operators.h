#ifndef OPERATORS_H
#define OPERATORS_H

#include "geometry.h"
#include "fourierTransform.h"

namespace gp
{
    namespace operators
    {
        class laplacian
        {
            public:
            laplacian(std::shared_ptr< fourierTransform<complex_t,complex_t> > fft);

            void apply( tensor_t & source, tensor_t & destination);           

            private:

            std::shared_ptr< fourierTransform<complex_t,complex_t> > _fft;

            std::shared_ptr<tensor_t> K2,fieldK;



        };

    };
};

#endif