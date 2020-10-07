from ase.io.trajectory import TrajectoryReader
from ase.io import write
import os
from PIL import Image

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

folder = ".tmpmovie"
if not os.path.exists(folder):
    os.mkdir(folder)
atoms = TrajectoryReader("data/kmc_test.traj")
dangle = 0.1 # Degrees

bbox = [-10, -10, 50, 50]

counter = 0
for i, at in enumerate(atoms):
    if i > 1000 and i%10 != 0:
        continue

    fname = folder + f"/image{counter:04}.png"
    counter += 1
    
    a = at.copy()
    # rotate to to look in x direction
    a.rotate('z', 'x', rotate_cell=True)
    write(folder + '/x.png', a, bbox=bbox)

    a = at.copy()
    a.rotate('z', 'y', rotate_cell=True)
    write(folder + '/y.png', a, bbox=bbox)
    
    a = at.copy()
    write(folder + '/z.png', a, bbox=bbox)

    imx =  Image.open(folder + '/x.png')
    imy =  Image.open(folder + '/y.png')
    imz =  Image.open(folder + '/z.png')
    im = get_concat_h(imx, imy)
    im = get_concat_h(im, imz)
    im.save(fname)
