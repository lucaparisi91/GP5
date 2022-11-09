#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "traits.h"
#include <memory>

namespace gp{

class mesh
{
    public:

    mesh() : mesh({0,0,0}) {}
    mesh(const intDVec_t & N   );


    size_t size() const {return _N[0]*_N[1]*_N[2];}
    const auto & shape() const {return _N;}

    const auto & getNGhosts() const {return _nGhosts;}

    auto getNGhosts(int d) const {return _nGhosts[d];}


    const auto & getGlobalOffset() const  { return _offset; }

    void  setGlobalOffset( const intDVec_t & index )  { _offset=index; }

    auto getGlobalOffset(int d ) const  { return _offset[d]; }

    std::shared_ptr<mesh> permute( const intDVec_t & ordering)  const;

    private:

    intDVec_t _N;
    intDVec_t _nGhosts;
    intDVec_t _offset;

};

class domain
{
    public:
    domain();

    domain( realDVec_t left,realDVec_t right);

    const auto & getLBox() const  {return _lBox;}
    const auto & getLeft() const {return _left;}
    const auto & getRight() const {return _right;}

    void setLeft(const realDVec_t & left) { _left=left;updateLBox();}
    void setRight(const realDVec_t & right){_right=right;updateLBox();};
    std::shared_ptr<domain> permute( const intDVec_t &  order) const ;

    private:


    void updateLBox();

    realDVec_t _lBox;
    realDVec_t _left;
    realDVec_t _right;

};

class discretization
{
    public:

    using mesh_t = mesh;
    using domain_t = domain;

    discretization();

    const auto &  getLocalMesh() const {return _localMesh;}
    const auto &  getGlobalMesh() const {return _globalMesh;}
    const auto &  getDomain() const {return _domain;}

    void setLocalMesh( std::shared_ptr<mesh_t>  newMesh)  {_localMesh=newMesh;};
    void setGlobalMesh( std::shared_ptr<mesh_t> newMesh) {_globalMesh=newMesh;}
    void setDomain( std::shared_ptr<domain_t>  newDomain) {_domain=newDomain;};

    const MPI_Comm & getCommunicator() const { return _comm; }

    void setCommunicator(const MPI_Comm & comm ) { _comm=comm; }
    
    real_t getSpaceStep( int d) const { return _domain->getLBox()[d] /_globalMesh->shape()[d]; } ;

    real_t getFourierStep(int d) const {return 2*M_PI/_domain->getLBox()[d]; }

    private:

    std::shared_ptr<mesh_t> _globalMesh;
    std::shared_ptr<mesh_t> _localMesh;
    MPI_Comm _comm;
    intDVec_t processorGrid;
    std::shared_ptr<domain> _domain;

};


tensor_t momentums(std::shared_ptr<discretization> discr,int d, int nComponents);

tensor_t positions(std::shared_ptr<discretization> discr,int d, int nComponents);




std::shared_ptr<discretization> createUniformDiscretization( std::shared_ptr<domain> globalDomain, std::shared_ptr<mesh> & globalMesh, intDVec_t processorGrid, MPI_Comm & comm);

void initGaussian( const realDVec_t & sigma, std::shared_ptr<discretization> discr, tensor_t & tensor, int comp);

void initGaussian( real_t sigma, std::shared_ptr<discretization> discr, tensor_t & tensor, int comp);

void applyVortexPhase( std::shared_ptr<discretization> discr, tensor_t & field, int comp);


void normalize( real_t N, tensor_t & field, int c, std::shared_ptr<discretization> discr);
real_t norm( const tensor_t & field, int c, std::shared_ptr<discretization> discr );

void normalize( const std::vector<int> & N, tensor_t & field, std::shared_ptr<discretization> discr);


};



#endif
