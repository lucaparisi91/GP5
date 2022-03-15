#include "traits.h"
#include <mpi.h>
#include <p3dfft.h>


int main(int argc,char** argv)
{
    std::array<real_t,3> lBox = { 1, 1, 1 } ;
    std::array<int,3> N = { 100 , 100, 100 };
    std::array<real_t ,3 > deltax = { lBox[0] * 1./N[0] , lBox[1] * 1./N[1], lBox[2] * 1./N[2] };

    std::array<int,3> shapeLocal;
    std::array<int,3> offsetGlobal;
    std::array<int,3> shapeLocal2;
    std::array<int,3> offsetGlobal2;

    MPI_Init( & argc, &argv);

    p3dfft::setup();


    int type_ids1[3] = {p3dfft::CFFT_FORWARD_D,p3dfft::CFFT_FORWARD_D,p3dfft::CFFT_FORWARD_D};
    int type_ids2[3] = {p3dfft::CFFT_BACKWARD_D,p3dfft::CFFT_BACKWARD_D,p3dfft::CFFT_BACKWARD_D};
    
    int pdims[] = { 1 , 1 , 2 };
    int mem_order1[] = {0,1,2};
    int mem_order2[] = {1,2,0};

    int dmap1[] = {0,1,2};
    int dmap2[] = {1,2,0};
    

  
    int gdims[] = { N[0] , N[1] , N[2] };

    int Pgrid = p3dfft_init_proc_grid(pdims,MPI_COMM_WORLD);

     // Initialize the initial grid 
                     // this is an X pencil, since Px =1
    
    auto type_forward = p3dfft_init_3Dtype(type_ids1);
    auto type_backward = p3dfft_init_3Dtype(type_ids2);

    auto Xpencil = p3dfft_init_data_grid(gdims,-1,Pgrid,dmap1,mem_order1);

  
    auto Zpencil = p3dfft_init_data_grid(gdims,-1,Pgrid,dmap2,mem_order2);

    auto trans_f = p3dfft_plan_3Dtrans(Xpencil,Zpencil,type_forward);

    
    auto trans_b = p3dfft_plan_3Dtrans(Zpencil,Xpencil,type_backward);


    for( int i=0;i<3;i++) {
        offsetGlobal[mem_order1[i] ] = Xpencil->GlobStart[i];
        shapeLocal[mem_order1[i]] = Xpencil->Ldims[i];
        offsetGlobal2[mem_order2[i] ] = Zpencil->GlobStart[i];
        shapeLocal2[mem_order2[i]] = Zpencil->Ldims[i];
    }

    real_t alpha=1.0;
    tensor_t field(  shapeLocal[0],shapeLocal[1],shapeLocal[2],1);
    tensor_t fieldFourier(  shapeLocal2[0],shapeLocal2[1],shapeLocal2[2],1);

    for(int i=0;i<shapeLocal[0];i++)
        for(int j=0;j<shapeLocal[1];j++)
            for(int k=0;k<shapeLocal[2];k++)
            {
                real_t x = -lBox[0]/2. + (i + offsetGlobal[0] +  0.5) * deltax[0];
                real_t y = -lBox[1]/2. + (j + offsetGlobal[1] + 0.5) * deltax[1];
                real_t z = -lBox[2]/2. + (k + offsetGlobal[2] + 0.5 ) * deltax[2];

                real_t r = std::sqrt( x*x + y*y + z*z);
                field(i,j,k,0)=exp(-alpha * x*x);
            }
    
    p3dfft_exec_3Dtrans_double(trans_f,(real_t * )field.data(),(real_t *)fieldFourier.data(),0);

    p3dfft_exec_3Dtrans_double(trans_b,(real_t * )fieldFourier.data(),(real_t *)field.data(),0);


    p3dfft_free_data_grid(Xpencil);
    p3dfft_free_data_grid(Zpencil);
    p3dfft_free_proc_grid(Pgrid);
    p3dfft_cleanup();



    MPI_Finalize();

}