FOLDER=$1

for fname in ${FOLDER}/chgl*.grid
do
    mmsp2png $fname --field=0 --zoom
done

# Convert them into a mp4
#ffmpeg -start_number 25 -framerate 10 ${FOLDER}/chgl*.png ${FOLDER}/movie.mp4