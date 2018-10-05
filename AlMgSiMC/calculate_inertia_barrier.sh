
export OMP_NUM_THREADS=2
for ((i=0;i<10;i++))
do
  # Run calculation
  nice -19 /lib64/openmpi/bin/mpirun -np 4 python3 inertia_barrier.py run $i

  # Fit bias potential
  python3 inertia_barrier.py update_bias $i
done
