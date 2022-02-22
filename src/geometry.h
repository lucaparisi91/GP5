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

    const auto & size() const {return _size;}
    const auto & shape() const {return _N;}

    const auto & getNGhosts() const {return _nGhosts;}

    auto getNGhosts(int d) const {return _nGhosts[d];}


    const auto & getGlobalOffset() const  { return _offset; }

    void  setGlobalOffset( const intDVec_t & index )  { _offset=index; }

    auto getGlobalOffset(int d ) const  { return _offset[d]; }

    private:

    intDVec_t _N;
    size_t _size;
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
    void setDomain( std::shared_ptr<domain_t> & newDomain) {_domain=newDomain;};

    const MPI_Comm & getCommunicator() const { return _comm; }

    void setCommunicator(const MPI_Comm & comm ) { _comm=comm; }

    real_t getSpaceStep( int d) const { return _domain->getLBox()[d] /_globalMesh->shape()[d];} ;

    real_t getFourierStep(int d) const {return 2*M_PI/_domain->getLBox()[d]; }


    private:

    std::shared_ptr<mesh_t> _globalMesh;
    std::shared_ptr<mesh_t> _localMesh;
    MPI_Comm _comm;
    intDVec_t processorGrid;
    std::shared_ptr<domain> _domain;

};


tensor_t momentums(std::shared_ptr<discretization> discr,int d, int nComponents);

std::shared_ptr<discretization> createUniformDiscretization( std::shared_ptr<domain> globalDomain, std::shared_ptr<mesh> & globalMesh, intDVec_t processorGrid, MPI_Comm & comm);

};



#endif