#!/bin/bash

#PBS -N SIbands
#PBS -l walltime=00:02:00
#PBS -l select=1:ncpus=1
#PBS -A nn9497k

module load python
export GPAW_FFTWSO=''
export LD_LIBRARY_PATH="/usr/lib64":"/home/ntnu/davidkl/.local/lib":${LD_LIBRARY_PATH}
export GPAW_SETUP_PATH="/home/ntnu/davidkl/GPAW/gpawData/gpaw-setups-0.9.20000"

FOLDER="/home/ntnu/davidkl/GPAWTutorial"

cd ${FOLDER}

python ${FOLDER}/siBands.py
