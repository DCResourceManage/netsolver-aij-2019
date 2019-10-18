#!/bin/bash

MAIN_PATH=../../../instances/hadoop/hadoop_extensions
UNDIRECTED=True
MULTI_THREAD=False
TIME_LIMIT=3600 # seconds
MEM_LIMIT=80000 # 80 GB

scripts=(servermin # ./netsolver-gurobi-servermin.py
    hotspot # ./netsolver-gurobi-hotspot.py
    softaffinity) # ./netsolver-gurobi-softaffinity.py

result_path=timelogs
mkdir ${result_path}

pns=(us-mid1.pn
    us-mid2.pn
    us-west1.pn
    us-west2.pn)

vns=(hadoop_4_1bw.vn
    hadoop_4_2bw.vn
    hadoop_10_1bw.vn
    hadoop_10_2bw.vn
    hadoop_15_1bw.vn
    hadoop_15_2bw.vn)

for script in ${scripts[@]}; do
    mkdir ${result_path}/${script}
    for pn in ${pns[@]}; do
        for vn in ${vns[@]}; do
            out_fname=results.${script}.${pn}.${vn}.log
            runlim_fname=results.${script}.${pn}.${vn}.stdout
            echo --- starting: ${out_fname}
            cmd="runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ./netsolver-gurobi-${script}.py ${MAIN_PATH}/${pn} ${MAIN_PATH}/${vn} ${UNDIRECTED} ${MULTI_THREAD} ${result_path}/${script}/${out_fname}"
            echo --- started: $cmd
            $cmd |& tee ${result_path}/${script}/${runlim_fname}
            echo +++ completed: $cmd 
            echo
        done
    done
done

echo "+++ all done; bye +++"

