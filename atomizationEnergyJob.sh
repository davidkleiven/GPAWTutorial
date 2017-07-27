#!/bin/bash

#PBS -l walltime=00:02:00

module load python

FOLDER="/home/ntnu/davidkl/GPAWTutorials"

cd ${FOLDER}

python ${FOLDER}/atomizationEnergy.py
