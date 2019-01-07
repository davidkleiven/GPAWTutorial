START=$1
END=$2

export OMP_NUM_THREADS=1

for ((size=$START;size<$END;size++))
do
    echo $size
    nice -19 python3 relax_cluster.py --size=$size
done