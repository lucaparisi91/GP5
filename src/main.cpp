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
#include "timers.h"

int main(int argc,char** argv)
{
    MPI_Init(& argc,&argv);
    p3dfft::setup();

    if (argc !=  2 )
    {
        throw std::runtime_error(" The program must have one argument only : the yaml input file");
    }

    std::string confFileName(argv[1]);
    int numProcs,rank;

    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    YAML::Node config = YAML::LoadFile(confFileName);
    // ########## Set up the geometry ####################
    auto globalShape= config["mesh"]["shape"].as<intDVec_t>();
    auto globalMesh = std::make_shared<gp::mesh>(globalShape);

    auto left = config["domain"]["left"].as<realDVec_t>();
    auto right = config["domain"]["right"].as<realDVec_t>();

    auto domain=std::make_shared<gp::domain>(left,right);

    int nComponents=config["wavefunction"]["nComponents"].as<int>();

    auto processorGrid = config["parallel"]["processorGrid"].as<intDVec_t>();

    intDVec_t ordering = config["parallel"]["fftOrdering"].as<intDVec_t>();

    //######### Builds FFT operator ################
    if (rank == 0)
    {
        std::cerr << "Initializing FFT operator ..." <<std::endl;
    }


    auto fftC = std::make_shared< gp::fourierTransformCreator<complex_t,complex_t> >();

    fftC->setNComponents(nComponents);
    fftC->setCommunicator(MPI_COMM_WORLD);
    fftC->setDomain(domain);
    fftC->setGlobalMesh(globalMesh);
    fftC->setProcessorGrid( processorGrid);
    fftC->setOrdering(ordering);


    auto fftOp = fftC->create();
    
    // ############### Builds the functionals
    if (rank == 0)
    {

        std::cerr << "Initializing functional ..." <<std::endl;
    }

    
    auto laplacian = std::make_shared<gp::operators::laplacian>(fftOp);

    auto discr = fftOp->getDiscretizationRealSpace();

    auto funcC=std::make_shared<gp::functionalConstructor>();

    funcC->setLaplacianOperator(laplacian);
    funcC->setDiscretization(discr);
    funcC->setNComponents(nComponents);

    auto func = funcC->create(config["functional"]);

    auto localShape = discr->getLocalMesh()->shape();


//  ####### initialize fields 
    if (rank == 0)
    {

        std::cerr << "Initializing fields ..." <<std::endl;
    }

    tensor_t oldField(localShape[0],localShape[1],localShape[2],nComponents);
    tensor_t newField(localShape[0],localShape[1],localShape[2],nComponents);

    oldField.setConstant(0);
    newField.setConstant(0);

    auto initialConditionFileName = config["initialCondition"]["file"].as<std::string>();

    auto normalizations = config["wavefunction"]["normalization"].as<std::vector<real_t> >();


    oldField = load( initialConditionFileName , *discr, nComponents);

// ############ initialize stepper #####################

    if (rank == 0)
    {

        std::cerr << "Initializing stepper ..." <<std::endl;
    }
    auto stepperConfig = config["evolution"];

    auto timeStep = stepperConfig["timeStep"].as<real_t>();
    auto stepperName = stepperConfig["stepper"].as<std::string>();
    auto isImaginary = stepperConfig["imaginaryTime"].as<bool>();    
    bool reNormalize = stepperConfig["reNormalize"].as<bool>();
    
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
        stepper->setTimeStep( complex_t ( 0,timeStep ));
    }

    stepper->setFunctional(func);
    stepper->setNComponents(nComponents);
    stepper->setDiscretization(discr);
    stepper->setNormalizations( normalizations);
    stepper->enableReNormalization(reNormalize);

    for(auto it=stepperConfig.begin();it!=stepperConfig.end();it++)
    {
        if  ( it->first.as<std::string>() == "constraint")
        {
            gp::constraintConstructor constrC;
            constrC.setDiscretization( discr);
            constrC.setNComponents( nComponents);

            auto constraint = constrC.create(it->second);

            stepper->setConstraint(constraint);
            
        }
    }
    stepper->init();

    auto maxTime = stepperConfig["maxTime"].as<real_t>();
    
    auto outDir = config["output"]["folder"].as<std::string>();
    auto nIterations = config["output"]["nIterations"].as<size_t>();

    if ( not  std::filesystem::exists(outDir) )
    {
        std::filesystem::create_directories(outDir);
    }


    if (rank == 0)
    {
        std::cerr << "START" <<std::endl;
    }
    
    START_TIMER("total");



    real_t t=0;
    int k=0;
    while ( t <= maxTime)
    {
        if (rank == 0)
        {
            std::cout << "Time: " << t << std::endl;
        }

        START_TIMER("io");
        gp::save( oldField, outDir + "/out_" + std::to_string(k) + ".hdf5" , *discr  );
        if (rank == 0)
        {
            std::cout << "saved" << std::endl;
        }
        STOP_TIMER("io");


       
        for( int  i=0;i<nIterations;i++)
        {
            stepper->advance(oldField,newField,t);
            std::swap(newField,oldField);
            t+=timeStep;
        }
        k++;

        
        
    }
    STOP_TIMER("total");


    /* int cRank = 0;
    while ( cRank < numProcs) {
        if (cRank == rank) {
            std::cerr << "Rank: " << std::to_string(rank) << std::endl; 
            std::cerr << timers::getInstance().report() << std::endl;        
            std::cerr << std::flush;
        }
        cRank ++;
        MPI_Barrier (MPI_COMM_WORLD);
    }
 */

    if (rank == 0 )
    {
        std::cerr << "Rank: " << std::to_string(rank) << std::endl; 
        std::cerr << timers::getInstance().report() << std::endl;        
        std::cerr << std::flush;
    }

    stepper=NULL;
    fftOp=NULL;
    func=NULL;
    laplacian=NULL;
    discr=NULL;
    fftC = NULL;
    funcC=NULL;
    
    p3dfft::cleanup();

    MPI_Finalize();
    

}








