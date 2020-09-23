STEP=5
FILE="conc_file.csv"
NUM_JOBS=0

echo "Al,Mg,Si,Cu" > $FILE

for (( nsi=0; nsi<50; nsi += $STEP))
do
    for (( ncu=0; ncu<100; ncu += $STEP ))
    do
        for (( nmg=0;nmg<100;nmg += $STEP ))
        do
            if (($nsi+$nmg+$ncu > 100))
            then
                continue
            fi
            nal=$((100-$nsi-$ncu-$nmg))

            echo $nal,$nmg,$nsi,$ncu >> $FILE
            NUM_JOBS=$(($NUM_JOBS+1))
        done
    done
done

echo "Total number of jobs: ${NUM_JOBS}"