#include "externalPotential.h"
#include <iostream>
#include "io.h"
namespace gp {

    harmonicPotential::harmonicPotential(const std::vector<realDVec_t> & omegas) : _omegas(omegas)
    {

    }

    std::shared_ptr<tensor_t> harmonicPotential::create( std::shared_ptr<discretization> discr, int nComponents ) const
    {
            auto X= positions( discr,0, nComponents );
            auto Y= positions( discr,1, nComponents );
            auto Z= positions( discr,2, nComponents );

            auto _V=std::make_shared<tensor_t>(X.dimensions() );
            _V->setConstant(0);

            const auto & dimensions = X.dimensions();
            for (int c=0;c<dimensions[3];c++) 
                for(int k=0;k<dimensions[2];k++)
                    for(int j=0;j<dimensions[1];j++) 
                        for(int i=0;i<dimensions[0];i++)
                        { 
                            (*_V)(i,j,k,c)=(_omegas[c][0]*_omegas[c][0]*X(i,j,k,c)*X(i,j,k,c) + _omegas[c][1]*_omegas[c][1]*Y(i,j,k,c)*Y(i,j,k,c) + _omegas[c][2]*_omegas[c][2]*Z(i,j,k,c)*Z(i,j,k,c) )*0.5;
                        }
        

            return _V;

    }

     std::shared_ptr<tensor_t> vortexPotential::create( std::shared_ptr<discretization> discr, int nComponents ) const
    {
            auto X= positions( discr,0, nComponents );
            auto Y= positions( discr,1, nComponents );

            auto _V=std::make_shared<tensor_t>(X.dimensions() );
            _V->setConstant(0);

            const auto & dimensions = X.dimensions();
            for (int c=0;c<dimensions[3];c++) 
                for(int k=0;k<dimensions[2];k++)
                    for(int j=0;j<dimensions[1];j++) 
                        for(int i=0;i<dimensions[0];i++)
                        { 
                            auto r2 = X(i,j,k,c) * X(i,j,k,c) + Y(i,j,k,c)*Y(i,j,k,c);
                            (*_V)(i,j,k,c)=0.5/r2;
                        }
        

            return _V;

    }


    std::shared_ptr<tensor_t> sumPotential::create( std::shared_ptr<discretization> discr, int nComponents ) const
    {
        auto shape = discr->getLocalMesh()->shape();

        auto V=std::make_shared<tensor_t>(EXPAND_D(shape),nComponents);
         
        V->setConstant( complex_t{0,0} );
        
        for (auto pot : _pots)
        {
            auto tmp=pot->create(discr,nComponents);
            *V+=*tmp;
        } 

        return V;

    }

     std::shared_ptr<tensor_t> potentialFromFile::create( std::shared_ptr<discretization> discr, int nComponents ) const
    {
        auto _V= load( _filename , *discr, nComponents);
        auto V = std::make_shared<tensor_t>(_V.dimensions() );
        *V=_V;
        return V ;
    }



    std::shared_ptr<externalPotential> externalPotentialConstructor::create( const config_t & settings ) 
    {
        std::vector<std::shared_ptr<externalPotential> > pots;

        for ( auto it = settings.begin() ; it!=settings.end() ; it++)
        {
            auto settingsPot=it->second;

            if (settingsPot["name"].as<std::string>() == "harmonic" )
            {
                    auto omegas = settingsPot["omegas"].as<std::vector<realDVec_t> >();

                pots.push_back(std::make_shared<harmonicPotential>(omegas) );
            }
            else if (settingsPot["name"].as<std::string>() == "vortex" )
            {
                  

                    pots.push_back(std::make_shared<vortexPotential>( ) );
            }
            else if (settingsPot["name"].as<std::string>() == "potentialFromFile" )
            {
                  auto filename = settingsPot["filename"].as<std::string >();
                pots.push_back(std::make_shared<potentialFromFile>( filename) );
            }
            else 
            {
                throw std::runtime_error("Unkown potential");

            }
                 

        }
        
        return std::make_shared<sumPotential>(pots);
                   
    }


}