#include "constraint.h"
#include "io.h"

namespace gp {

    void phaseConstraint::apply(tensor_t & field) const 
    {
        field=(field*field.conjugate()).sqrt()*_phase_exp;
    }

    
    phaseConstraint::phaseConstraint(const tensor_t & phase) :
    _phase(phase),
    _phase_exp(phase)
    {
        _phase_exp=( _phase * complex_t(0,1)).exp();
    }

    std::shared_ptr<constraint> constraintConstructor::create( YAML::Node config)
    {
        std::shared_ptr<constraint> constr= std::make_shared<nullConstraint>();

          for ( auto it = config.begin() ; it != config.end() ; it++)
            {
                auto constrConfig = it->second;
                auto name = constrConfig["name"].as<std::string>();

                if (name == "phaseConstraint")
                {
                    auto file = constrConfig["file"].as<std::string>();
                    auto phase= load( file,*_discr,_nComponents);
                    return std::make_shared<phaseConstraint>(phase);
                }
            }
            

        return constr;
    }
}