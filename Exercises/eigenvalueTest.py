import numpy as np

def main():
    matrix = np.array( [[1,2,3],[4,5,6],[7,8,9]] )
    matrix = matrix+matrix.T
    eigval, eigvec = np.linalg.eigh(matrix)
    print (eigval)

if __name__ == "__main__":
    main()
