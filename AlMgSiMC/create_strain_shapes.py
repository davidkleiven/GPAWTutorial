from ase.io import read, write
from ase.build import cut
from ase.visualize import view

def main():
    atoms = read("data/mgsi100_fully_relaxed.xyz")
    
    # Create slab
    plane1 = atoms
    slab = cut(plane1, a=(1, 0, -1), b=(0, 1, 0))

    # Plane perp to planes
    plane = slab*(6, 1, 6)
    write("data/plate_perp2layers.xyz", plane)

    # Plane parallel to planes
    plane = slab*(6, 6, 1)
    write("data/plate_par2planes.xyz", plane)

    # Needles perp to planes
    needle = slab*(6, 2, 2)
    write("data/needle_perp2planes.xyz", needle)
    view(needle)

    # Needle par to planes
    needle = slab*(2, 2, 6)
    write("data/needle_par2planes.xyz", needle)
    view(needle)
    view(slab)

if __name__ == "__main__":
    main()

