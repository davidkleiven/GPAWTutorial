STEP=5
FILE="data/sa_concs.csv"
NUM_JOBS=0

echo "Al,Mg,Si,Cu" > $FILE

for (( nsi=0; nsi<51; nsi += $STEP))
do
    for (( ncu=0; ncu<95; ncu += $STEP ))
    do
        for (( nmg=0;nmg<95;nmg += $STEP ))
        do
            if (($nsi+$nmg+$ncu > 100))
            then
                continue
            fi
            nal=$((100-$nsi-$ncu-$nmg))
	    if (($nal == 100))
            then
                continue
	    fi

            echo $nal,$nmg,$nsi,$ncu >> $FILE
            NUM_JOBS=$(($NUM_JOBS+1))
        done
    done
done

echo "Total number of jobs: ${NUM_JOBS}"
