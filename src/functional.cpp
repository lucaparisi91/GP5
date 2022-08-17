#include "functional.h"
#include "externalPotential.h"

namespace gp{

    gpFunctional::gpFunctional() :
    _setCouplings(false),
    functional::functional()
    {

    }

    void gpFunctional::apply( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        getLaplacianOperator()->apply(fieldDataOld,fieldDataNew);

        const auto & dimensions = fieldDataNew.dimensions();

        for (int c=0;c<dimensions[3];c++) 
        for(int k=0;k<dimensions[2];k++)
            for(int j=0;j<dimensions[1];j++) 
                for(int i=0;i<dimensions[0];i++)
                { 
                fieldDataNew(i,j,k,c)=-0.5*_inverseMasses[c]*fieldDataNew(i,j,k,c); 
                }
            
            addPotential(fieldDataOld,fieldDataNew,time);
            

            if (_setCouplings)
            {
            if ( getNComponents() == 1)
            {
                addSingleComponentMeanFieldInteraction(fieldDataOld,fieldDataNew,time);
            }
            else if ( getNComponents() == 2)
            {
                addTwoComponentMeanFieldInteraction(fieldDataOld,fieldDataNew,time);
            }

            }

    }


    void functional::addPotential( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        if (_setExternalPotential)
        {
            fieldDataNew+=(*_V)*fieldDataOld;
        }
    
    }
    void gpFunctional::addSingleComponentMeanFieldInteraction( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        int c=0;
        const auto & dimensions = fieldDataNew.dimensions();
    
        real_t g = _couplings(c,c);
        for(int k=0;k<dimensions[2];k++)
            for(int j=0;j<dimensions[1];j++) 
            for(int i=0;i<dimensions[0];i++)
            { 
                auto density = fieldDataOld(i,j,k,c).real() *fieldDataOld(i,j,k,c).real() + fieldDataOld(i,j,k,c).imag() * fieldDataOld(i,j,k,c).imag()  ;

                fieldDataNew(i,j,k,c)+=g*density*fieldDataOld(i,j,k,c)   ;
            }  
    }


    void gpFunctional::addTwoComponentMeanFieldInteraction( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
    const auto & dimensions = fieldDataNew.dimensions();
    
        real_t g11 = _couplings(0,0);
        real_t g22 = _couplings(1,1);
        real_t g12 = _couplings(0,1);
        
        for(int k=0;k<dimensions[2];k++)
            for(int j=0;j<dimensions[1];j++) 
                for(int i=0;i<dimensions[0];i++)
                { 
                    auto density0 = std::norm(fieldDataOld(i,j,k,0) );
                    auto density1 = std::norm(fieldDataOld(i,j,k,1) );

                    fieldDataNew(i,j,k,0)+=(g11*density0 + g12*density1 )*fieldDataOld(i,j,k,0)   ;
                    fieldDataNew(i,j,k,1)+=(g22*density1 + g12*density0 )*fieldDataOld(i,j,k,1)   ;
                }
    }


    void gpFunctional::init()
    {
        _masses.resize(getNComponents(),1);
        _inverseMasses.resize(getNComponents(),1);

        for(int i=0;i<getNComponents();i++)
        {
            _inverseMasses[i]=1./_masses[i];
        };

    }


void LHYDropletFunctional::apply( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        getLaplacianOperator()->apply(fieldDataOld,fieldDataNew);

        
        const auto & dimensions = fieldDataNew.dimensions();

        for (int c=0;c<dimensions[3];c++) 
            for(int k=0;k<dimensions[2];k++)
                for(int j=0;j<dimensions[1];j++) 
                    for(int i=0;i<dimensions[0];i++)
                    { 
                        auto density= std::norm(fieldDataOld(i,j,k,c));
                        fieldDataNew(i,j,k,c)=-0.5*fieldDataNew(i,j,k,c) + ( -3 * density + 5/2. * std::pow(density,3/2.))*fieldDataOld(i,j,k,c); 
                    }    
        addPotential(fieldDataOld,fieldDataNew,time);

    }


void LHYDropletUnlockedFunctional::apply( tensor_t & fieldDataOld, tensor_t & fieldDataNew, real_t time )
    {
        getLaplacianOperator()->apply(fieldDataOld,fieldDataNew);

        const auto & dimensions = fieldDataNew.dimensions();

        for(int k=0;k<dimensions[2];k++)
            for(int j=0;j<dimensions[1];j++) 
                for(int i=0;i<dimensions[0];i++)
                    { 
                        auto density1= std::norm(fieldDataOld(i,j,k,0));
                        auto density2= std::norm(fieldDataOld(i,j,k,1));

                        fieldDataNew(i,j,k,0)=-0.5*fieldDataNew(i,j,k,0) + ( 
                            density1 + _eta*density2 + _alpha*std::pow(density1 + _beta*density2,3./2)
                        )*fieldDataOld(i,j,k,0); 

                        fieldDataNew(i,j,k,1)=-0.5*fieldDataNew(i,j,k,1) + ( 
                            _eta*_beta*density1 + _beta*density2 + _alpha*_beta*_beta*std::pow(density1 + _beta*density2,3./2)
                        )*fieldDataOld(i,j,k,1); 
                    }
        
        addPotential(fieldDataOld,fieldDataNew,time);
        

    }



std::shared_ptr<functional> functionalConstructor::create(const config_t & settings)
{       
    std::shared_ptr<gp::functional> func=NULL;
    std::shared_ptr<externalPotential> pot=NULL;

    for ( auto itF = settings.begin() ; itF != settings.end() ; itF++)
    {
        auto funcConfigs = itF->second;

        if (funcConfigs["name"].as<std::string>() == "gpFunctional" )
        {
            auto gpFunc = std::make_shared<gp::gpFunctional>();

            for ( auto it = funcConfigs.begin() ; it != funcConfigs.end() ; it++)
            {
                if ( it->first.as<std::string>() == "coupling" )
                {
                    auto _couplings = it->second.as<std::vector<std::vector<real_t> > >();

                    Eigen::Tensor<real_t , 2> couplings(_nComponents,_nComponents); 
                    couplings.setConstant(0);

                    if (_couplings.size() != _nComponents)
                    {
                        throw std::runtime_error("Incompatible number of components");
                    }

                    for(int j=0;j<_nComponents;j++)
                    {
                        if (_couplings[j].size() != _nComponents)
                            {
                            throw std::runtime_error("Incompatible number of components");
                        }

                        for(int i=0;i<_nComponents;i++)
                        {
                            couplings(i,j)= _couplings[j][i];
                        }
                    }

                    gpFunc->setCouplings(couplings);
                } 
                else if ( it->first.as<std::string>() == "masses" )
                {
                    auto masses = it->second.as<std::vector<real_t> >();
                    gpFunc->setMasses(masses);
                } 

            
    
            }     
            func=gpFunc;

           
        }
        else if (funcConfigs["name"].as<std::string>() == "LHYDroplet" )
        {
            func=std::make_shared<LHYDropletFunctional>();
        }
        else if ( funcConfigs["name"].as<std::string>() == "UnlockedLHYDroplet" )
        {
            auto alpha = funcConfigs["alpha"].as<real_t>();
            auto beta = funcConfigs["beta"].as<real_t>();
            auto eta = funcConfigs["eta"].as<real_t>();
            
            func=std::make_shared<LHYDropletUnlockedFunctional >(alpha, beta, eta);
        }
        else if ( funcConfigs["name"].as<std::string>() == "externalPotential" )
        {

        }
        else
        {
            throw std::runtime_error(" Unkown functional");
        }

    }

    
    for ( auto itF = settings.begin() ; itF != settings.end() ; itF++)
    {
        auto funcConfigs = itF->second;
        
        if (funcConfigs["name"].as<std::string>() == "externalPotential" )
        {
            gp::externalPotentialConstructor vConstr;

            pot=vConstr.create( funcConfigs["V"] );   

            if ( pot != NULL )
            {
                auto V = pot->create(_discr,_nComponents);
                func->setExternalPotential(V);
            }

        }
    }

    func->setNComponents(_nComponents);
    func->setDiscretization(_discr);
    func->setLaplacianOperator(_laplacian);
    func->init();

    return func;
}


}