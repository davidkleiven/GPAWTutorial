a=$1
b=$2
PREFIX=/work/sophus/almgsi_ellipse2D/prec600K_${a}a_${b}b_
nice -19 python3 chgl_almgsi2D_mev.py --initfunc=ellipse --a=$a --b=$b --dx=0.5 --steps=1000 --update_freq=100 --prefix=$PREFIX