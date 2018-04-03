from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

seeds = None
if ( rank == 0 ):
    seeds = np.random.randint(0, high=100000, size=size)

seed = comm.scatter( seeds, root=0 )
print ("{}: {}".format(rank,seed))
comm.barrier()
np.random.seed(seed)

print ("{}: {}".format(rank,np.random.randint(0, high=10, size=10)))
