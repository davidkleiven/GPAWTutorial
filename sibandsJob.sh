#!/bin/bash

#PBS -N SIbands
#PBS -l walltime=00:02:00
#PBS -l select=1:ncpus=32
#PBS -A nn9497k

module load python
export GPAW_FFTWSO=''
export LD_LIBRARY_PATH="/usr/lib64":"/home/ntnu/davidkl/.local/lib":${LD_LIBRARY_PATH}
export GPAW_SETUP_PATH="/home/ntnu/davidkl/GPAW/gpawData/gpaw-setups-0.9.20000"
export PATH=${PATH}:"/home/ntnu/davidkl/.local/bin"

FOLDER="/home/ntnu/davidkl/GPAWTutorial"

cd ${FOLDER}

mpirun -np 4 gpaw-python ${FOLDER}/siBands.py
