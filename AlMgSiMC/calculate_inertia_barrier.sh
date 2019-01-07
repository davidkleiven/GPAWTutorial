
export OMP_NUM_THREADS=1
for ((i=45;i<85;i++))
do
  # Run calculation
  nice -19 mpirun -np 4 python3 inertia_barrier.py run $i

  # Fit bias potential
  python3 inertia_barrier.py update_bias $i
done
