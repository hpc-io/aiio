#!/usr/bin/env python

# https://docs.nersc.gov/development/languages/python/parallel-python/
from mpi4py import MPI
import os
import subprocess
import sys


mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
if mpi_rank == 0:
    print("# of ranks = ", mpi_size)

comm = MPI.COMM_WORLD

comm.Barrier()

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

INPUT_ROT='/project/projectdirs/m2621/dbin/Cori_archive_2019_un_taz/2019/'
OUTPU_ROT='/project/projectdirs/m2621/dbin/Cori_archive_2019_un_taz_parsered/2019/'
OUTPU_MON='2'
TEMPR_ROT='/global/cscratch1/sd/dbin/iod-tempdir/'
n = len(sys.argv)
print(n)
print(sys.argv)
if n == 5:
    INPUT_ROT=sys.argv[1]
    OUTPU_ROT=sys.argv[2]
    OUTPU_MON=sys.argv[3]
    TEMPR_ROT=sys.argv[4]
    if mpi_rank == 0:
        print("Argument List:", str(sys.argv))
else:
    if mpi_rank == 0:
        print("Usage: paralle.py input_root_dir output_root_dir month temp_directory")
        print("  e.g., paralle.py /project/projectdirs/m2621/dbin/Cori_archive_2019_un_taz/2019/ /project/projectdirs/m2621/dbin/Cori_archive_2019_un_taz_parsered/2019/ 2 /global/cscratch1/sd/dbin/iod-tempdir/")
    exit()

INPUT_DIR=INPUT_ROT+OUTPU_MON+'/'
OUTPU_DIR=OUTPU_ROT+OUTPU_MON+'/'
OUTPU_MER=OUTPU_ROT+OUTPU_MON+'-merge.csv'

if mpi_rank == 0:
    os.makedirs(OUTPU_DIR, exist_ok=True)
comm.Barrier()

dir_list=get_immediate_subdirectories(INPUT_DIR)
dir_list.sort()
if mpi_rank == 0:
    print(dir_list)

if mpi_size < len(dir_list):
    print("Please run job with [", len(dir_list), " ] MPI ranks")
    exit()

if mpi_rank < len(dir_list):
    input_sub_dir=INPUT_DIR+dir_list[mpi_rank]
    #print('my rank : ', mpi_rank ,input_sub_dir)
    output_csv_file=OUTPU_DIR+dir_list[mpi_rank]+'.csv'
    temp_directory=TEMPR_ROT+'tempdir-'+str(dir_list[mpi_rank])
    #print('my rank : ', mpi_rank ,output_csv_file)
    ##print('my rank : ', mpi_rank , ", ",  input_sub_dir, ", ",  output_csv_file, ", ",  temp_directory)
    subprocess.run(["./parser.sh", input_sub_dir, output_csv_file, temp_directory])

###exit()

##comm.Barrier()

##if mpi_rank == 0:
##    print("Finish all days in a month and try to merge them")
##    print(OUTPU_MER)
##    subprocess.run(["cp", "/dev/null", OUTPU_MER])  ##cp /dev/null access.log
##    for i in range(len(dir_list)):
##        output_csv_file=OUTPU_DIR+dir_list[i]+'.csv'
##        subprocess.run(["cat", output_csv_file, " >>", OUTPU_MER])

##comm.Barrier()
    
