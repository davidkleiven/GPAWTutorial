while read p; do
	echo "Group $p"
	python3 group2server.py $p
done <converged.txt
