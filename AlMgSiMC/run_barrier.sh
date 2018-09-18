for (( i=0; i<1000000; i++))
do
  mpirun.mpich -np 4 python3 free_energy_barrier.py
done
