#!/bin/bash

MAIN_PATH=../../../instances/hadoop
UNDIRECTED=True
MULTI_THREAD=False
TIME_LIMIT=3600 # seconds
MEM_LIMIT=80000 # 80 GB

SCRIPT=./netsolver-gurobi-hadoop-mod.py
result_path=timelogs
mkdir ${result_path}

pns=(us-mid1.pn
    us-mid2.pn
    us-west1.pn
    us-west2.pn)

vns=(instance_list_hadoop_4vm_rnd.txt
    instance_list_hadoop_6vm_rnd.txt
    instance_list_hadoop_8vm_rnd.txt
    instance_list_hadoop_10vm_rnd.txt
    instance_list_hadoop_12vm_rnd.txt
    instance_list_hadoop_14vm_rnd.txt)

for pn in ${pns[@]}; do
    for vn in ${vns[@]}; do
        out_fname=results.${pn}.${vn}.log
        runlim_fname=results.${pn}.${vn}.stdout
        echo --- starting: ${out_fname}
        cmd="runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT} ${MAIN_PATH}/hadoop_extra/${pn} ${MAIN_PATH}/${vn} ${UNDIRECTED} ${MULTI_THREAD} ${result_path}/${out_fname}"
        echo --- started: $cmd
        $cmd |& tee ${result_path}/${runlim_fname}
        echo +++ completed: $cmd 
        echo
    done
done

echo "+++ all done; bye +++"

