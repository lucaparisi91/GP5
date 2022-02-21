#include "traits.h"


class mesh
{
    public:


    mesh(const intDVec_t & N   );

    const auto & size() const {return _size;}
    const auto & shape() const {return _N;}

    const auto & nGhosts() const {return _nGhosts;}

    auto nGhosts(int d) const {return _nGhosts[d];}


    const auto & globalOffset() const  { return _offset; }

    void  setOffset( const intDVec_t & index )  { _offset=index; }

    
    auto getOffset(int d ) const  { return _offset[d]; }



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

    void setLeft(const realDVec_t & left);
    void setRight(const realDVec_t & right);


    private:

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

    void setLocalMesh( const mesh_t & localMesh)  ;
    void setGlobalMesh( const mesh_t & localMesh)  ;
    void setDomain( const domain_t & domain);

    const MPI_Comm & getCommunicator() const { return _comm; }

    void setCommunicator(const MPI_Comm & comm ) { _comm=comm; }

    real_t spacestep( int d) const ;

    private:

    mesh _globalMesh;
    mesh _localMesh;
    MPI_Comm _comm;
    intDVec_t processorGrid;
    domain _domain;

};








