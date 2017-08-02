#!/bin/bash

#PBS -N SIbands
#PBS -l walltime=00:00:15
#PBS -l select=1:ncpus=1
#PBS -A nn9497k

module load python

FOLDER="/home/ntnu/davidkl/GPAWTutorial/Exercises"

cd ${FOLDER}

python ${FOLDER}/eigenvalueTest.py
