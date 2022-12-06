
#ifndef MODEL_H
#define MODEL_H

#define DIMENSIONS 3

#include "petscdmda.h"


void setMatrix( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox );

void setMatrixSpherical( DM da, Mat H, PetscReal* leftBox,PetscReal * rightBox );

PetscErrorCode  createDMSpherical(DM * da, PetscInt * shape);

#endif