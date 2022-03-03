#include "functional.h"


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

}