#include "traits.h"
#include <mpi.h>
#include <p3dfft.h>
#include "geometry.h"
#include "fourierTransform.h"
#include "operators.h"
#include "functional.h"
#include "io.h"
#include "stepper.h"
#include <filesystem>
#include "externalPotential.h"


int main(int argc,char** argv)
{
    MPI_Init(& argc,&argv);
    p3dfft::setup();

    if (argc !=  2 )
    {
        throw std::runtime_error(" The program must have one argument only : the yaml input file");
    }

    std::string confFileName(argv[1]);

    YAML::Node config = YAML::LoadFile(confFileName);

    // ######## Set up the geometry #####################
    
    auto globalShape= config["mesh"]["shape"].as<intDVec_t>();
    auto globalMesh = std::make_shared<gp::mesh>(globalShape);

    auto left = config["domain"]["left"].as<realDVec_t>();
    auto right = config["domain"]["right"].as<realDVec_t>();

    auto domain=std::make_shared<gp::domain>(left,right);

    int nComponents=config["wavefunction"]["nComponents"].as<int>();

    auto processorGrid = config["parallel"]["processorGrid"].as<intDVec_t>();

    //######### Builds FFT operator ################
    std::cerr << "Initializing FFT operator ..." <<std::endl;

    gp::fourierTransformCreator<complex_t,complex_t> fftC;
    fftC.setNComponents(nComponents);
    fftC.setCommunicator(MPI_COMM_WORLD);
    fftC.setDomain(domain);
    fftC.setGlobalMesh(globalMesh);
    fftC.setProcessorGrid( processorGrid);
    auto fftOp = fftC.create();
    
    // ############### Builds the functionals
    std::cerr << "Initializing functional ..." <<std::endl;

    auto laplacian = std::make_shared<gp::operators::laplacian>(fftOp);

    auto func=std::make_shared<gp::gpFunctional>();
    auto funcConfigs = config["functional"];

    if (funcConfigs["name"].as<std::string>() != "gpFunctional" )
    {
        throw std::runtime_error("Unkown functional");
    }

    std::shared_ptr<gp::externalPotential> pot=NULL;


    for ( auto it = funcConfigs.begin() ; it != funcConfigs.end() ; it++)
    {
        if ( it->first.as<std::string>() == "externalPotential" )
        {
            gp::externalPotentialConstructor vConstr;

            pot=vConstr.create( it->second);
                        
        } 
        else if ( it->first.as<std::string>() == "coupling" )
        {
            auto _couplings = it->second.as<std::vector<std::vector<real_t> > >();

            Eigen::Tensor<real_t , 2> couplings(nComponents,nComponents); 
            couplings.setConstant(0);


            if (_couplings.size() != nComponents)
            {
                throw std::runtime_error("Incompatible number of components");
            }

            for(int j=0;j<nComponents;j++)
            {
                if (_couplings[j].size() != nComponents)
                    {
                    throw std::runtime_error("Incompatible number of components");
                }

                for(int i=0;i<nComponents;i++)
                {
                    couplings(i,j)= _couplings[j][i];
                }
            }

            func->setCouplings(couplings);
        } 
        else if ( it->first.as<std::string>() == "masses" )
        {
            auto masses = it->second.as<std::vector<real_t> >();
            func->setMasses(masses);
        }

        
        

    }


    auto discr = fftOp->getDiscretizationRealSpace();
    func->setNComponents(nComponents);
    func->setDiscretization(discr);
    func->setLaplacianOperator(laplacian);

    if ( pot != NULL )
    {
        auto V = pot->create(discr,nComponents);
        func->setExternalPotential(V);

    }
    
    func->init();

    auto localShape = discr->getLocalMesh()->shape();

//  ####### initialize fields 
    std::cerr << "Initializing fields ..." <<std::endl;

    tensor_t oldField(localShape[0],localShape[1],localShape[2],nComponents);
    tensor_t newField(localShape[0],localShape[1],localShape[2],nComponents);

    oldField.setConstant(0);
    newField.setConstant(0);

    auto initialConditionFileName = config["initialCondition"]["file"].as<std::string>();

    auto normalizations = config["wavefunction"]["normalization"].as<std::vector<real_t> >();


    oldField = load( initialConditionFileName , *discr, nComponents);


// ############ initialize stepper #####################

    std::cerr << "Initializing stepper ..." <<std::endl;
    auto stepperConfig = config["evolution"];

    auto timeStep = stepperConfig["timeStep"].as<real_t>();
    auto stepperName = stepperConfig["stepper"].as<std::string>();
    auto isImaginary = stepperConfig["imaginaryTime"].as<bool>();    

    
    std::shared_ptr<gp::stepper> stepper;

    if (stepperName == "eulero")
    {
        stepper=std::make_shared<gp::euleroStepper>();
    }
    else if (stepperName == "RK4")
    {
        stepper = std::make_shared<gp::RK4Stepper>();
    }

    if (isImaginary)
    {
        stepper->setTimeStep( complex_t(timeStep,0));
    }
    else
    {
        stepper->setTimeStep( complex_t ( 0,-timeStep ));
    }


    stepper->setFunctional(func);
    stepper->setNComponents(nComponents);
    stepper->setDiscretization(discr);
    stepper->setNormalizations( normalizations);
    stepper->init();
    

    auto maxTime = stepperConfig["maxTime"].as<real_t>();
    
    auto outDir = config["output"]["folder"].as<std::string>();
    auto nIterations = config["output"]["nIterations"].as<size_t>();


    if ( not  std::filesystem::exists(outDir) )
    {
        std::filesystem::create_directories(outDir);
    }

// load time stepping rules
    real_t t=0;
    size_t k=0;
    while ( t <= maxTime)
    {
        std::cout << "Time: " << t << std::endl;
        gp::save( oldField, outDir + "/out_" + std::to_string(k) + ".hdf5" , *discr  );
        for( size_t  i=0;i<nIterations;i++)
        {
            stepper->advance(oldField,newField,t);
            std::swap(newField,oldField);
            t+=timeStep;
        }
        k++;
    }

    
    stepper=NULL;
    fftOp=NULL;
    func=NULL;
    laplacian=NULL;
    
    p3dfft::cleanup();

    MPI_Finalize();
    

}








