#define DIMENSIONS 3

#include "petscdmda.h"


void setMatrix( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox );

void setMatrixSpherical( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox );