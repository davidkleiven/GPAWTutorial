from ase import build
from ase.visualize import view

def main():
    aluminum = build.bulk( "Al", crystalstructure="fcc" )*4
    print (len(aluminum))

    # Extract 3 111 planes
    planes = build.cut( aluminum, (1,-1,0),(1,1,-2), nlayers=3 )
    view( planes, viewer="Avogadro" )

if __name__ == "__main__":
    main()
