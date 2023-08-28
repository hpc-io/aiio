import csv
import os
import sys



##print(len(sys.argv))

## input :
##
##  sys.argv[1] :  temp directory for darshan-parser output
##                 input_dir
##  sys.argv[2] :  output file
##                 output_file
##


input_dir = "./tempdir"
output_file= "./Darshan_Total.csv"

#print(sys.argv)

#if len(sys.argv) == 3:
#    input_dir = sys.argv[1] 
#else:
#    input_dir= "./tempdir"

if len(sys.argv) == 3:
    input_dir = sys.argv[1]
    output_file = sys.argv[2] 
else:
    input_dir= "./tempdir"
    output_file= "./Darshan_Total.csv"

##print("input_dir = ", input_dir, ", output_file = ", output_file)

headerCol = []
headerValue = []
fileHeader = open('./headerTotal.csv','r')
Lines = fileHeader.readlines()
#print(len(Lines))
#print(Lines)
for line in Lines:
    #print(line)
    line = line[:-1]
    headerCol.append(line)
    headerValue.append('-1')
fileHeader.close()
fileHeader = open('./headerTotal.csv','r')


file_total_input = open(input_dir+'/parsed_total.txt','r')
file_total_output = open(output_file,'a+')

file_count = open(input_dir+'/count.txt','r')
count = file_count.read()
file_count.close()
##print(input_dir+'/count.txt, read count =',  count)
if int(count) == 0:
    Lines = fileHeader.readlines()
    row = ''
    for line in Lines:
        line = line[:-1]
        row = row + str(line) + ','
    row = row[:-1] + '\n'
    file_total_output.write(row)
    

LinesTotal = file_total_input.readlines()
row = ''
headList = ['POSIX','STDIO','MPIIO']

for line in LinesTotal:
    if line.startswith('# exe'):
        exe = str((line.partition("# exe: ")[2])[:-1]).replace(',',';')
        headerValue[headerCol.index("exe")] = str(exe)
    if line.startswith('# uid'):
        uid = str((line.partition("# uid: ")[2])[:-1])
        headerValue[headerCol.index("uid")] = str(uid) 
    if line.startswith('# jobid'):
        jobid = str((line.partition("# jobid: ")[2])[:-1])
        headerValue[headerCol.index("jobid")] = str(jobid)    
    if line.startswith('# darshan log version'):
        darshan_log_version = str((line.partition("# darshan log version: ")[2])[:-1])
        headerValue[headerCol.index("darshan_log_version")] = str(darshan_log_version)
    if line.startswith('# start_time:'):
        start_time = str((line.partition("# start_time: ")[2])[:-1])
        headerValue[headerCol.index('start_time')] = str(start_time)
    if line.startswith('# start_time_asci'):
        start_time_asci = str((line.partition("# start_time_asci: ")[2])[:-1])
        headerValue[headerCol.index('start_time_asci')] = str(start_time_asci)
    if line.startswith('# end_time:'):
        end_time = str((line.partition("# end_time: ")[2])[:-1])
        headerValue[headerCol.index('end_time')] = str(end_time)
    if line.startswith('# end_time_asci'):
        end_time_asci = str((line.partition("# end_time_asci: ")[2])[:-1])
        headerValue[headerCol.index('end_time_asci')] = str(end_time_asci)
    if line.startswith('# nprocs'):
        nprocs = str((line.partition("# nprocs: ")[2])[:-1])
        headerValue[headerCol.index('nprocs')] = str(nprocs)
    if line.startswith('# run time'):
        runtime = str((line.partition("# run time: ")[2])[:-1])
        headerValue[headerCol.index('runtime')] = str(runtime)
    if line.startswith(('total')):
        total = str((line.partition(": ")[2])[:-1])
        heading = str(line.partition(": ")[0])
        headerValue[headerCol.index(heading)] = str(total)
        # if not any(x in heading for x in headList):
        #     heading = heading + "\n"
        #     file5.write(heading)

##
## Get the performance number
##
#parsed_perf.txt
## agg_perf_by_slowest: 1062.704311 # MiB/s
perf_headList = ['POSIX_PERF_MIBS','MPIIO_PERF_MIBS','STDIO_PERF_MIBS']
perf_headList_index = 0
file_perf_input = open(input_dir+'/parsed_perf.txt','r')
LinesPerf = file_perf_input.readlines()
for line in LinesPerf:
    if line.startswith(('# POSIX module data')):
        perf_headList_index = 0
    if line.startswith(('# MPI-IO module data')):
        perf_headList_index = 1
    if line.startswith(('# STDIO module data')):
        perf_headList_index = 2
    if line.startswith(('# agg_perf_by_slowest')):
            total = str((line.partition(": ")[2])[:-1])
            perf=str(total.partition(" # ")[0])
            headerValue[headerCol.index(perf_headList[perf_headList_index])] = str(perf)
file_perf_input.close()

##
## Get the Luster ino
##
#
#LUSTRE_OSTS     248
#LUSTRE_MDTS     5
#LUSTRE_STRIPE_OFFSET    -1
#LUSTRE_STRIPE_SIZE      1048576
#LUSTRE_STRIPE_WIDTH     8

file_luster_input = open(input_dir+'/parsed_luster.txt','r')
LinesLustre = file_luster_input.readlines()
LUSTRE_STRIPE_WIDTH_list=[]
LUSTRE_STRIPE_SIZE_list=[]

for line in LinesLustre:
    if line.startswith(('LUSTRE_OSTS')): 
            value = str((line.partition("\t")[2])[:-1])
            headerValue[headerCol.index('LUSTRE_OSTS')] = str(value)
    if line.startswith(('LUSTRE_MDTS')):
            value = str((line.partition("\t")[2])[:-1])
            headerValue[headerCol.index('LUSTRE_MDTS')] = str(value)
    if line.startswith(('LUSTRE_STRIPE_OFFSET')):
            value = str((line.partition("\t")[2])[:-1])
            headerValue[headerCol.index('LUSTRE_STRIPE_OFFSET')] = str(value)
    if line.startswith(('LUSTRE_STRIPE_SIZE')):
            value = str((line.partition("\t")[2])[:-1])
            LUSTRE_STRIPE_SIZE_list.append(int(value))
            avg = int(sum(LUSTRE_STRIPE_SIZE_list)/len(LUSTRE_STRIPE_SIZE_list))
            headerValue[headerCol.index('LUSTRE_STRIPE_SIZE')] = str(avg)
    if line.startswith(('LUSTRE_STRIPE_WIDTH')):
            value = str((line.partition("\t")[2])[:-1])
            LUSTRE_STRIPE_WIDTH_list.append(int(value))
            avg = int(sum(LUSTRE_STRIPE_WIDTH_list)/len(LUSTRE_STRIPE_WIDTH_list))
            headerValue[headerCol.index('LUSTRE_STRIPE_WIDTH')] = str(avg)

file_luster_input.close()


row = ''
for value in headerValue:
    row = row + value + ',' 

row = row[:-1] + '\n'
file_total_output.write(row)

count = int(count) + 1
file_count = open(input_dir+'/count.txt','w')
file_count.write(str(count))
file_count.close()
    
file_total_output.close()
file_total_input.close()


# file5.close()

# for line in Lines:
#     splitted = []
#     row = ''
#     if line.startswith(('POSIX','STDIO','MPI','H5F','H5D','PNETCDF','BGQ','LUSTRE')):
#         splitted = line.split()
#         row = row + str(uid) + ',' + str(jobid) + ',' + str(start_time) + ',' + str(end_time)
#         for i in splitted:
#             row = row + ',' + str(i).replace(',',';')
#         row = row + '\n'
#         file3.write(row)

# file3.close()
# file1.close()

