from mpi4py import MPI
comm = MPI.COMM_WORLD
# print(comm.Get_size())
print(comm.Get_rank())
