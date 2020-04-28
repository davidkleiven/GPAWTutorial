FOLDER="/work/sophus/nestedlasso"
for i in {0..99}
do
    echo "Running file $i of 100"
    fname="$FOLDER/dataset$i.csv"
    out="$FOLDER/nestedlasso$i.json"
    /home/gudrun/davidkl/Documents/goselect/main nestedlasso -csvfile=$fname -out=$out -lambMin=1e-10 -keep=0.9
done