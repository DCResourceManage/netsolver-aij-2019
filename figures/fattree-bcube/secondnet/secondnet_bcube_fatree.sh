#!/bin/bash

#SCRIPT=./secondnet
#SCRIPT=./vdcalloc
#SCRIPT=./vdcalloc_undirected
SCRIPT=./vdcalloc_print_final_pn
MAIN_PATH=../../../instances/fattree-bcube
TIME_LIMIT=3600 # seconds
MEM_LIMIT=80000 # 80 GB

result_path=timelogs
mkdir ${result_path}

pns=(0,8,2,10,16,BCube_n8_k2_bw10_cpu16.pn
    0,10,2,10,16,BCube_n10_k2_bw10_cpu16.pn
    0,12,2,10,16,BCube_n12_k2_bw10_cpu16.pn
    0,16,2,10,16,BCube_n16_k2_bw10_cpu16.pn
    1,8,2,10,16,Fattree_n8_k2_bw10_cpu16.pn
    1,12,2,10,16,Fattree_n12_k2_bw10_cpu16.pn
    1,16,2,10,16,Fattree_n16_k2_bw10_cpu16.pn
    )

vns=(instance_list_6vm_rnd.txt
    instance_list_9vm_rnd.txt
    instance_list_12vm_rnd.txt
    instance_list_15vm_rnd.txt)

for pn in ${pns[@]}; do
    # turn e.g. '0,8,2,10,16,BCube_n8_k2_bw10_cpu16.pn' into
    # array ['0','8','2','10','16','BCube_n8_k2_bw10_cpu16.pn']
    IFS="," read -r -a arr <<< ${pn}
    pn_config="${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]} ${arr[4]}"
    pn_name=${arr[5]}
    for vn in ${vns[@]}; do
        out_fname=results.${pn_name}.${vn}.log
        runlim_fname=results.${pn_name}.${vn}.stdout
        echo --- starting: ${out_fname}
        cmd="runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} ${SCRIPT} ${MAIN_PATH}/${vn} ${pn_config} ${result_path}/${out_fname}"
        echo --- started: $cmd
        $cmd |& tee ${result_path}/${runlim_fname}
        echo +++ completed: $cmd 
        echo
    done
done

echo "+++ all done; bye +++"

