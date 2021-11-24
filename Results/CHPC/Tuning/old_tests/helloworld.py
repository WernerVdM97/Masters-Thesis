import os
import sys
#sys.path.insert(0, "../../..")

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD   
size = comm.Get_size()
rank = comm.Get_rank()
#name = MPI.Get_processor_name()
print("Hello, World! I am process %d of %d.\n"
    % (rank, size))
'''

path = "/home/wvandermerwe1/lustre/MTF/out/"

def write_file(filepath, run_time, size, rank):
        filetitle = "{0}_{1}.txt".format(size, rank)
        f = open(os.path.join(filepath, filetitle), 'w')
        f.write("{0}\n".format(run_time))
        f.flush()
        f.close()
st = time.time()

sys.stdout.write(
    "Hello, World! I am process %d of %d.\n"
    % (rank, size))
sys.stdout.write(str(sys.version))
sys.stdout.write("\n")
#sys.stdout.write(str(sys.version_info))

time = time.time() - st

write_file(path,time, size, (rank+1))
'''
