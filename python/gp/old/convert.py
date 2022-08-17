import argparse
from itertools import count
import os
from tkinter import N
import field
import re

from functools import reduce
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert .hdf5 files to netCDF files for visualization')
    parser.add_argument('files',nargs="+", type=str )
    parser.add_argument('--out', type=str )

    args = parser.parse_args()
    for file in args.files:

        psi=field.load(file)
        outputFile=re.sub("\.hdf5$",".nc", file )

        if args.out is not None:
            outputFile=os.path.join(args.out,os.path.basename(outputFile))

        field.saveNetCDF(psi,outputFile)





