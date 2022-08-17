#include "pyTools.h"

std::vector<real_t> toVector(const py::array_t<double> & x)
{
    
    py::buffer_info info = x.request();
    if (info.shape.size() != 1)
    {
        throw std::runtime_error("Numpy array should have rank 1");
    };
    auto xRaw = x.unchecked<1>();
    
    std::vector<real_t> xCpp(info.shape[0]);

    for(int i=0;i<info.shape[0];i++)
        {
            xCpp[i]=xRaw(i);
        }
    return xCpp;
}


