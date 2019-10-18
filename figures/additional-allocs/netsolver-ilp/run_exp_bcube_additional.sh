#!/bin/bash

SCRIPT=./netsolver-gurobi-mod.py
MAIN_PATH=../../../instances/fattree-bcube
UNDIRECTED=False
MULTI_THREAD=False
TIME_LIMIT=3600 # seconds
MEM_LIMIT=80000 # 80 GB

result_path=timelogs
mkdir ${result_path}

pns=(BCube_n8_k2_bw10_cpu16.pn
    BCube_n10_k2_bw10_cpu16.pn
    BCube_n12_k2_bw10_cpu16.pn
    BCube_n16_k2_bw10_cpu16.pn
    Fattree_n8_k2_bw10_cpu16.pn
    Fattree_n12_k2_bw10_cpu16.pn
    Fattree_n16_k2_bw10_cpu16.pn)

vns=(instance_list_6vm_rnd.txt
    instance_list_9vm_rnd.txt
    instance_list_12vm_rnd.txt
    instance_list_15vm_rnd.txt)

for pn in ${pns[@]}; do
    for vn in ${vns[@]}; do
        out_fname=results.${pn}.${vn}.log
        runlim_fname=results.${pn}.${vn}.stdout
        echo --- starting: ${out_fname}
        cmd="runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT} ../secondnet/final_pns_json/final.${pn}.${vn}.pn.json ${MAIN_PATH}/${vn} ${UNDIRECTED} ${MULTI_THREAD} ${result_path}/${out_fname}"
        echo --- started: $cmd
        $cmd |& tee ${result_path}/${runlim_fname}
        echo +++ completed: $cmd 
        echo
    done
done

echo "+++ all done; bye +++"

