#ifndef SPHERICAL_H
#define SPHERICAL_H

#include "../src/traits.h"
#include "petscdmda.h"


void setKineticMatrixSpherical( DM da, Mat H, const  PetscReal* leftBox,const PetscReal * rightBox );


PetscErrorCode fillSpacePositions1D(DM da, Vec X, const PetscReal* left, const PetscReal* right );


class sphericalIntegrator
{
   public:

   sphericalIntegrator( DM da, PetscReal left, PetscReal right) ;


   real_t integrate(Vec psi);

   auto getSpaceStep(){return _dx; }
   

   private:

   PetscReal _dx;
   Vec _r2;
   Vec _tmp;
   DM _da;

};





class geometry
{
  public:

  geometry( real_t R, PetscInt shape);

  const auto &  getLeft() const {return _left;}
  const auto & getRight() const {return _right;}
  const auto &  getSpaceStep() const {return _spaceStep; }  

  const auto &  getIntegrator() const {return *_integrator; }  

  auto &  getIntegrator() {return *_integrator; }  

  
  const auto & getDM(){return _da;}
  
  private:
  real_t _left[1];
  real_t _right[1];
  real_t _spaceStep[1];
  DM _da;
  std::shared_ptr<sphericalIntegrator>  _integrator;
   

};




class gpSpherical
{
    public:

    gpSpherical(real_t g_,real_t mu_,std::shared_ptr<geometry> geo) : _geo(geo),mu(mu_),g(g_)
    { initialize(); }

    
    void evaluate( Vec X, Vec Y ); // Y = f(X)

    PetscErrorCode evaluateJacobian(Vec X ,Mat J);

    auto & getGeometry() {return *_geo;}



    private:

    PetscErrorCode initialize();


    real_t  g;
    real_t mu; 
    std::shared_ptr<geometry> _geo;

    Mat _J0;
    Vec diagonalTMP;

};


void getNorm( geometry geo, Vec psi);

#endif