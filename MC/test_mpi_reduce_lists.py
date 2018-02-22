from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

my_list = [rank,rank**2]

all_lists = []
#comm.Barrier()
all_lists = comm.gather( my_list, root=0 )
my_list = []
if ( rank == 0 ):
    for sublist in all_lists:
        my_list += sublist
    print ( my_list )
