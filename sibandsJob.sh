#!/bin/bash

#PBS -N SIbands
#PBS -l walltime=00:02:00
#PBS -l select=1:ncpus=1
#PBS -A NN9497K

module load python
export GPAW_FFTWSO=''

FOLDER="/home/ntnu/davidkl/GPAWTutorials"

cd ${FOLDER}

python ${FOLDER}/sibands.py
