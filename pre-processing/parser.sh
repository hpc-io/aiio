#!/bin/bash -l

# AIIO Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy) and Ohio State
# University. All rights reserved.

# https://stackoverflow.com/questions/238073/how-to-add-a-progress-bar-to-a-shell-script
# 1. Create ProgressBar function
# 1.1 Input is currentState($1) and totalState($2)
function ProgressBar {
# Process data
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
# Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

# 1.2 Build progressbar strings and print the ProgressBar line
# 1.2.1 Output example:                           
# 1.2.1.1 Progress : [########################################] 100%
printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"

}


if [ "$#" -lt 2 ]
then
        echo "Wrong input : " >&2
	echo  $@ >&2
	echo  $#
        echo "Usage:  parser.sh input-directory output-csv-file" >&2
        echo "example:  ./parser.sh log-test ./log-test.csv [temp-dir]"
        exit 2
fi


log_raw_dir=$1
output_file=$2

if [ "$#" -eq 3 ]
then
	tmp_dir=$3
else
	tmp_dir="./tempdir"
fi

s=10000000

#log_raw_dir="/Users/dbin/work/darshan-log/log-test"
#output_file="${log_raw_dir}_Darshan_Total.csv"
##tmp_dir="./tempdir"

echo -n "" > ${output_file}
mkdir -p ${tmp_dir}
touch ${tmp_dir}/count.txt
echo "0" > ${tmp_dir}/count.txt

file_name_list="${tmp_dir}/filelist.txt"
find ${log_raw_dir} -type f > ${file_name_list}

total_lines=$(wc -l ${file_name_list})
current_lines=0

while IFS= read -r filename
do
       #echo "$filename"
       # echo "$filename"
       #start=$(date +%s)
       ls -s $filename | awk '{print $1;}'i > ${tmp_dir}/filesize.txt
       size=$(head -n 1 ${tmp_dir}/filesize.txt)
       #end=$(date +%s)
       #echo "Elapsed Time to get size : $(($end-$start)) seconds"
       #echo "$size"
       if (( $size < $s )); then
                darshan-parser --total $filename > ${tmp_dir}/parsed_total.txt 2>/dev/null
                error=$?
                if [ $error -ne 0 ]
                then
                        #echo "Skip erroneous file : $filename"
                        continue
                fi
                darshan-parser --perf $filename > ${tmp_dir}/parsed_perf.txt 2>/dev/null
                error=$?
                if [ $error -ne 0 ]
                then
                        #echo "Skip erroneous file : $filename"
                        continue
                fi
                darshan-parser $filename  2>/dev/null  | grep '^LUSTRE'  | cut -d$'\t' -f 4-5 > ${tmp_dir}/parsed_luster.txt 
                error=$?
                if [ $error -ne 0 ]
                then
                        #echo "Skip erroneous file : $filename"
                        continue
                fi
		end2=$(date +%s)
		#echo "Elapsed Time for run darshan-parser 3 times : $(($end2-$end)) seconds"
                python parser.py ${tmp_dir} ${output_file}
 		#end3=$(date +%s)
		#echo "Elapsed Time merge results of darshan-parser : $(($end3-$end2)) seconds"
                
       fi
       ##ProgressBar ${current_lines} ${total_lines}
       ##current_lines=$((current_lines+1))
done < "$file_name_list"
echo "Done with $output_file"
#rm  -rf ${tmp_dir}

