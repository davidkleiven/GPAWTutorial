SCRIPT=/home/davidkleiven/Documents/atatUtils/scripts/phase_fractions.py
DB=/home/davidkleiven/Documents/GPAWTutorial/CE/AlMgSiCu/almgsicu.db

for x1 in {0..100..5}
do
    for x2 in {0..100..5}
    do
        for x3 in {0..100..5}
        do
            x4=$((100 - $x1 - $x2 - $x3))
            if [ $x4 -lt 0 ]
            then
                continue
            fi
            c1=$(awk "BEGIN {print $x1/100}")
            c2=$(awk "BEGIN {print $x2/100}")
            c3=$(awk "BEGIN {print $x3/100}")
            c4=$(awk "BEGIN {print $x4/100}")
            echo "Conc: $c1 $c2 $c3 $c4"
            python3 $SCRIPT $DB --conc Al=$c1,Mg=$c2,Si=$c3,Cu=$c4 >> stability.txt
        done
    done
done