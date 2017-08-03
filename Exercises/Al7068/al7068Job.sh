#!/bin/bash

#PBS -N AL7068
#PBS -l walltime=04:00:00
#PBS -l select=2:ncpus=32:mpiprocs=16
#PBS -A nn9497k

module load python
export GPAW_FFTWSO=''
export LD_LIBRARY_PATH="/usr/lib64":"/home/ntnu/davidkl/.local/lib":${LD_LIBRARY_PATH}
export GPAW_SETUP_PATH="/home/ntnu/davidkl/GPAW/gpawData/gpaw-setups-0.9.20000"
export PATH=${PATH}:"/home/ntnu/davidkl/.local/bin"

FOLDER="/home/ntnu/davidkl/GPAWTutorial/Exercises/Al7068"

cd ${FOLDER}

mpirun -np 32 gpaw-python ${FOLDER}/energyAlloy.py
