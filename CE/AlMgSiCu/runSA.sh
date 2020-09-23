START=$1
END=$2

export PYTHONPATH=~/Documents/clease/

for (( i=$START; i<$END; i++ ))
do
    echo "Running job $i"
    nice -19 python3 sa.py $i
done