#include "io.h"
#include <hdf5.h>
#include <iostream>



template<class T> 
struct toHDF5Type
{

};


template<>
struct  toHDF5Type<double>
{
    static hid_t type;
};

hid_t toHDF5Type<double>::type = H5T_NATIVE_DOUBLE ;


template<>
struct  toHDF5Type<bool>
{
    static hid_t type;
};

hid_t toHDF5Type<bool>::type = H5T_NATIVE_HBOOL ;


template<>
struct  toHDF5Type< std::complex<double> >
{
  
    static hid_t type;
};

namespace __hdf5Tools
{
     hsize_t dims[]= {2};
}


 hid_t toHDF5Type<std::complex<double> >::type= H5Tarray_create(H5T_NATIVE_DOUBLE, 1, __hdf5Tools::dims );







namespace gp{


void save(const tensor_t & field , const std::string & filename, const discretization & discr )
{


    const auto & shape= discr.getGlobalMesh()->shape();
    const auto & localShape=discr.getLocalMesh()->shape();
    const auto & offset = discr.getLocalMesh()->getGlobalOffset();


    std::array<long,DIMENSIONS + 1> globalShape = { shape[0] , shape[1] , shape[2] , field.dimensions()[3]  };

    herr_t status;

    hid_t plistFile_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plistFile_id, discr.getCommunicator(), MPI_INFO_NULL);

    hid_t file = H5Fcreate (filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,plistFile_id);

    hsize_t hGlobalShape[DIMENSIONS + 1];
    hsize_t hMemoryShape[DIMENSIONS + 1];

    for (int d=0;d<DIMENSIONS+1;d++)
    {
        hGlobalShape[DIMENSIONS  - d]=globalShape[d];
    }
    for (int d=0;d<DIMENSIONS;d++)
    {
        hMemoryShape[DIMENSIONS - d]=localShape[d];
    }
    hMemoryShape[0]=hGlobalShape[0];

    hid_t fileDataSpaceId= H5Screate_simple(DIMENSIONS + 1, hGlobalShape, NULL);


    hid_t memoryDataSpaceId = H5Screate_simple(DIMENSIONS + 1, hMemoryShape, NULL);




     hsize_t blocks[DIMENSIONS + 1 ] , counts[DIMENSIONS + 1] , strides[DIMENSIONS + 1] , starts[DIMENSIONS + 1];

    // select in local memory
    for (int d=0;d<DIMENSIONS;d++)
    {
        blocks[DIMENSIONS  - d ]=localShape[d];
        starts[DIMENSIONS - d ]=0;
        counts[DIMENSIONS - d ] = 1;
        strides[DIMENSIONS - d ] = 1;
    }

    blocks[0]=globalShape[DIMENSIONS ];
    starts[0]=0;
    counts[0]=1;
    strides[0]=1;



    status = H5Sselect_hyperslab (memoryDataSpaceId, H5S_SELECT_SET, starts, strides, counts, blocks);


    // select in the data file
    for (int d=0;d<DIMENSIONS;d++)
    {
        blocks[DIMENSIONS  - d ]=localShape[d];
        starts[DIMENSIONS - d ]=offset[d];
        counts[DIMENSIONS - d ] = 1;
        strides[DIMENSIONS - d ] = 1;
    }

    blocks[0]=globalShape[DIMENSIONS ];
    starts[0]=0;
    counts[0]=1;
    strides[0]=1;

    status = H5Sselect_hyperslab (fileDataSpaceId, H5S_SELECT_SET, starts, strides, counts, blocks);

   


    hid_t dataSetId = H5Dcreate2(file, "field", toHDF5Type< value_t>::type,fileDataSpaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    H5Dwrite (dataSetId, toHDF5Type< value_t>::type , memoryDataSpaceId, fileDataSpaceId, plist_id , field.data() );

    H5Pclose(plist_id);


    status=H5Sclose(memoryDataSpaceId);
    status=H5Sclose(fileDataSpaceId);
    status=H5Dclose(dataSetId);

    H5Pclose(plistFile_id);
    status = H5Fclose (file);

}

tensor_t load( const std::string & filename, const discretization & discr, int nComponents)
{

    const auto & globalShape = discr.getGlobalMesh()->shape();


    herr_t status;
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
     H5Pset_fapl_mpio(plist_id, discr.getCommunicator(), MPI_INFO_NULL);

    auto localShape = discr.getLocalMesh()->shape();
    auto offset = discr.getLocalMesh()->getGlobalOffset();


    hid_t file = H5Fopen (filename.c_str(), H5F_ACC_RDONLY, plist_id);

    H5Pclose(plist_id);


    hsize_t hGlobalShape[DIMENSIONS+1];
    hsize_t hMemoryShape[DIMENSIONS + 1];

    for (int d=0;d<DIMENSIONS;d++)
    {
        hGlobalShape[DIMENSIONS - d ]=globalShape[d];
    }

    hGlobalShape[0]=nComponents;


    for (int d=0;d<DIMENSIONS;d++)
    {
        hMemoryShape[DIMENSIONS  - d]=localShape[d];
    }
    hMemoryShape[0]=nComponents;


    hid_t fileDataSpaceId= H5Screate_simple (DIMENSIONS + 1, hGlobalShape, NULL);

    hid_t memoryDataSpaceId = H5Screate_simple (DIMENSIONS + 1, hMemoryShape, NULL);

    /* 
    Select the valid data from the local cube
     */

    hsize_t blocks[DIMENSIONS + 1] , counts[DIMENSIONS + 1] , strides[ DIMENSIONS + 1 ] , starts[DIMENSIONS + 1];

    for (int d=0;d<DIMENSIONS;d++)
    {
        blocks[DIMENSIONS - d ]=localShape[d];
        starts[DIMENSIONS - d ]=0;
        counts[DIMENSIONS - d ] = 1;
        strides[ DIMENSIONS - d ] = 1;
    }

    blocks[0 ]=nComponents;
    starts[0]=0;
    counts[0] = 1;
    strides[ 0] = 1;

    status = H5Sselect_hyperslab (memoryDataSpaceId, H5S_SELECT_SET, starts, strides, counts, blocks);

    // select the hyper slab in the datafile

    for (int d=0;d<DIMENSIONS;d++)
    {
        blocks[DIMENSIONS - d  ]=localShape[d];
        starts[DIMENSIONS - d ]=0 + offset[d] ;
        counts[ DIMENSIONS -d ] = 1;
        strides[ DIMENSIONS -d ] = 1;
    }

    blocks[0 ]=nComponents;
    starts[0]=0;
    counts[0] = 1;
    strides[ 0] = 1;

    status = H5Sselect_hyperslab (fileDataSpaceId, H5S_SELECT_SET, starts, strides, counts, blocks);


    hid_t dataSetId = H5Dopen (file, "field", H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    tensor_t field(EXPAND_D(localShape),nComponents);
    field.setConstant(0);


    
    H5Dread (dataSetId, toHDF5Type< value_t >::type, memoryDataSpaceId, fileDataSpaceId, plist_id, field.data() );

    H5Pclose(plist_id);

    status=H5Sclose(memoryDataSpaceId);
    status=H5Sclose(fileDataSpaceId);
    status=H5Dclose(dataSetId);
    status = H5Fclose (file);

    return field;       
}

};

