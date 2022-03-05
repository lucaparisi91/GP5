#include "gtest/gtest.h"
#include "mpi.h"


int main(int argc, char **argv) {
   MPI_Init(&argc,&argv);
  ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();
  //if (pTools::rank() != 0) {
  //  delete listeners.Release(listeners.default_result_printer());
  //}
    ::testing::InitGoogleTest(&argc, argv);
    
  int ret= RUN_ALL_TESTS();

  MPI_Finalize();


}
