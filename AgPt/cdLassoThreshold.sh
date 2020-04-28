FOLDER="/work/sophus/nestedlasso"
for i in {0..99}
do
    echo "Running file $i of 100"
    fname="$FOLDER/dataset$i.csv"
    out="$FOLDER/thresholdLasso$i.json"
    /home/gudrun/davidkl/Documents/goselect/main lasso -csvfile=$fname -out=$out -lambMin=1e-5 -lambMax=1.0 -type=cd -cov=threshold
done