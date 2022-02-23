#include "operators.h"

namespace gp{
    namespace operators
    {
        laplacian::laplacian(std::shared_ptr< fourierTransform<complex_t,complex_t> > fft) :
        _fft(fft)
        {
            auto discrReal = _fft->getDiscretizationRealSpace();
            auto discrFourier = _fft->getDiscretizationFourierSpace();
            const auto & shapeFourier=discrFourier->getLocalMesh()->shape();
            int nComponents = 1;
            K2=std::make_shared<tensor_t>(shapeFourier[0],shapeFourier[1],shapeFourier[2],nComponents);

            auto KX = momentums(discrFourier,0,nComponents);
            auto KY = momentums(discrFourier,1,nComponents);
            auto KZ = momentums(discrFourier,2,nComponents);
            *K2 = KX*KX + KY*KY + KZ*KZ;

            fieldK=std::make_shared<tensor_t>(shapeFourier[0],shapeFourier[1],shapeFourier[2],nComponents);

            
        }


        void laplacian::apply( tensor_t & source, tensor_t & destination)
        {
            const auto & globalShape= _fft->getDiscretizationRealSpace()->getGlobalMesh()->shape();
            _fft->apply(source, *fieldK,gp::FFT_DIRECTION::FORWARD);
            *fieldK=-(*fieldK)*(*K2);
            _fft->apply(*fieldK, destination, gp::FFT_DIRECTION::BACKWARD);

            destination=destination* (1./complex_t(globalShape[0] * globalShape[1] * globalShape[2],0) );
        }
    };
};