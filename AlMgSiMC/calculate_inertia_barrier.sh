
for ((i=0;i<100;i++))
do
  # Run calculation
  mpirun -np 4 python3 inertia_barrier.py run $i

  # Fit bias potential
  python3 inertia_barrier.py update_bias $i
done
