#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <mpi.h>
#include "traits.h"
#include <pybind11/stl.h>

#include "geometry.h"
#include "fourierTransform.h"
#include "pyTools.h"
namespace py = pybind11;
#include "functional.h"
#include "stepper.h"
#include "io.h"
#include "cluster.h"
namespace pyInterface
{
    class geometry
    {
        public:

        geometry( std::array<real_t,DIMENSIONS> left,std::array<real_t,DIMENSIONS> right , std::array<int,DIMENSIONS> shape  ) : 
        _left(left),_right(right),_shape(shape)
    {
        
    }

    const auto & getLeft() const {return _left;}
    const auto & getRight() const {return _right;}
    const auto & getShape() const {return _shape;}


    private:

    std::array<real_t,DIMENSIONS> _left;
    std::array<real_t,DIMENSIONS> _right;
    std::array<int,DIMENSIONS> _shape;

};

class field 
{
    public:
    using value_t =std::complex<real_t> ;

    field(const pyInterface::geometry& geo , int nComponents , std::array<int,DIMENSIONS> processorGrid,
    std::array<int,DIMENSIONS> fftOrdering
    )
    {
        p3dfft::setup();

        auto fftC = std::make_shared< gp::fourierTransformCreator<complex_t,complex_t> >();

         _domain=std::make_shared<gp::domain>(geo.getLeft(),geo.getRight() );

        auto globalMesh = std::make_shared<gp::mesh>(geo.getShape());


        fftC->setNComponents(nComponents);
        fftC->setCommunicator(MPI_COMM_WORLD);
        fftC->setDomain(_domain);
        fftC->setGlobalMesh(  globalMesh  );
        fftC->setProcessorGrid( processorGrid);
        fftC->setOrdering(fftOrdering);


        _fftOp = fftC->create();
        auto localShape= _fftOp->getDiscretizationRealSpace()->getLocalMesh()->shape();
        _state=std::make_shared<Eigen::Tensor<value_t,4> >();

        _state->resize( localShape[0],localShape[1],localShape[2] ,nComponents);

        _state->setConstant(0);

        _nComponents=nComponents;

    }

    auto getDiscretization()
    {
        return _fftOp->getDiscretizationRealSpace();
    };

    auto nComponents() {return _nComponents;}

    auto getLocalShape()
    {
        return getDiscretization()->getLocalMesh()->shape();
    }

    auto getOffset()
    {
        return getDiscretization()->getLocalMesh()->getGlobalOffset();
    }


    auto toPyArray() {return toArray(*_state);  };

    void  setTensor(py::array_t<value_t >  arr )
    {
        *_state=toTensor<value_t ,4>(arr);
    }


    void save( const std::string & filename )
    {
        gp::save( *_state, filename , *getDiscretization()  );
    }

    void load( const std::string & filename )
    {
        *_state=gp::load( filename , *getDiscretization() , nComponents()  );
    }


    void setState(  std::shared_ptr< Eigen::Tensor<value_t,4> > state2)
    {
        _state=state2;
    }


    auto getNorm( int c)
    {
        
        return gp::norm(*_state,c,getDiscretization());

    }


    void normalize( int component, real_t N )
    {
        gp::normalize( N , *_state,component ,getDiscretization() );      
    }


    auto getState() {return _state;}

    auto getFFTOp() {return _fftOp; }



    private:
    
    std::shared_ptr<gp::domain> _domain;
    std::shared_ptr<gp::fourierTransform<value_t,value_t> > _fftOp;
    std::shared_ptr< Eigen::Tensor<value_t,4> > _state ;
    int _nComponents;

};

class model
{
    public:

    void initFunctional(std::shared_ptr<gp::functional>  func, field & psi)
    {
        auto fftOp=psi.getFFTOp();
        auto laplacian = std::make_shared<gp::operators::laplacian>( fftOp );
        func->setLaplacianOperator(laplacian);
        func->setDiscretization( psi.getDiscretization() );
        func->setNComponents(psi.nComponents() );
    }


    virtual std::shared_ptr<gp::functional> getFunctional() =0 ;

    void apply(field & source, field & dest,real_t time)
    {
        getFunctional()->apply(*source.getState(),*dest.getState() ,time);

    }
};





class LHY : public model
{
    public:
    LHY( field & psi )
    {
        
        _func=std::make_shared<gp::LHYDropletFunctional>();
        initFunctional(_func,psi);


    }

    auto nComponents() {return _func->getNComponents(); }
    virtual std::shared_ptr<gp::functional> getFunctional() {return _func; }




    private:

    std::shared_ptr<gp::LHYDropletFunctional> _func;
};


class timeStepper
{
    public:

    timeStepper( real_t timeStep, std::string stepperName, bool isImaginary  )
    {
        if (stepperName == "eulero")
        {
            _stepper=std::make_shared<gp::euleroStepper>();
        }
        else if (stepperName == "RK4")
        {
            _stepper = std::make_shared<gp::RK4Stepper>();
        }

        if (isImaginary)
        {
            _stepper->setTimeStep( complex_t(timeStep,0));
        }
        else
        {
            _stepper->setTimeStep( complex_t ( 0,timeStep ));
        }
        _timeStep=timeStep;
        t=0;
        _newState=std::make_shared<Eigen::Tensor<std::complex<real_t>,4 > >();
        _oldState=std::make_shared<Eigen::Tensor<std::complex<real_t>,4 > >();


    }


    void setFunctional( LHY & model)
    {
        auto func = model.getFunctional();
        _stepper->setFunctional(func );
        _stepper->setDiscretization( func->getDiscretization() );
        _stepper->setNComponents( func->getNComponents() );
        _stepper->init();
    }

    void setRenormalization( const std::vector<real_t> & normalizations)
    {
        _stepper->setNormalizations( normalizations);
        _stepper->enableReNormalization(true);
    }


    void unsetRenormalization()
    {
        _stepper->enableReNormalization(false);
    }

    void advance(field & field, int nSteps)
    {   
        _oldState= field.getState();

        _newState->resize(_oldState->dimensions() );
        for( size_t i=0;i<nSteps;i++)
        {
            _stepper->advance( *_oldState,*_newState,t);
            std::swap(_oldState,_newState);
            t+=_timeStep;
        }
        field.setState(_oldState);

        
    }



    auto getTime() {return t;}

    
    private:

    std::shared_ptr<gp::stepper> _stepper;
    std::shared_ptr<gp::functional> _func;

    std::shared_ptr<Eigen::Tensor<std::complex<real_t>,4 > > _oldState;
    std::shared_ptr<Eigen::Tensor<std::complex<real_t>,4 > > _newState;
    real_t _timeStep;
    real_t t;


};

class decomposition
{
    public:

    
    auto decompose(field & psi, real_t densityCutOff)
    {
        auto state = psi.getState();

        auto & dimensions = state->dimensions();

        _decomposition=std::make_shared< Eigen::Tensor<int,DIMENSIONS+1> >();
        _decomposition->resize(dimensions);

        gp::decompose::decompose(*state,densityCutOff,*_decomposition);

        return toArray(*_decomposition);

    }

private:

    std::shared_ptr<Eigen::Tensor<int,DIMENSIONS+1> > _decomposition;

};






}






PYBIND11_MODULE(gpCpp, m) {
     py::class_<pyInterface::geometry>(m, "geometry")
     .def(py::init< std::array<real_t,DIMENSIONS> , std::array<real_t,DIMENSIONS> , std::array<int,DIMENSIONS> >() )
     .def("getLeft",&pyInterface::geometry::getLeft)
      .def("getRight",&pyInterface::geometry::getRight)
     .def("getShape",&pyInterface::geometry::getShape)
     ;
      py::class_<pyInterface::field>(m, "field")
     .def(py::init< const pyInterface::geometry& , int , std::array<int,DIMENSIONS> ,
    std::array<int,DIMENSIONS>  >() )
    .def("getLocalShape",&pyInterface::field::getLocalShape )
    .def("getTensor",&pyInterface::field::toPyArray )
     .def("setTensor",&pyInterface::field::setTensor )
     .def("getOffset",&pyInterface::field::getOffset )
    .def("getNorm",&pyInterface::field::getNorm )
    .def("normalize",&pyInterface::field::normalize )
    .def("save",&pyInterface::field::save )
    .def("load",&pyInterface::field::load )

     ;
    py::class_<pyInterface::model>(m, "model")
    .def("apply",&pyInterface::model::apply);


    py::class_<pyInterface::decomposition>(m, "decomposition")
    .def("decompose",&pyInterface::decomposition::decompose);


    py::class_<pyInterface::LHY,pyInterface::model>(m, "LHY")
    .def(py::init< pyInterface::field &  >() )
    .def("nComponents",&pyInterface::LHY::nComponents )
    ;
    py::class_<pyInterface::timeStepper>(m, "timeStepper")
    .def(py::init< real_t,std::string,bool  >() )
    .def("setRenormalization",&pyInterface::timeStepper::setRenormalization )
    .def("unsetRenormalization",&pyInterface::timeStepper::unsetRenormalization )

    .def("setFunctional",&pyInterface::timeStepper::setFunctional )
    .def("advance",&pyInterface::timeStepper::advance )
    .def("getTime",&pyInterface::timeStepper::getTime )
    ;
    

  ;           
}