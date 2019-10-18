#!/bin/bash

SCRIPT=./vdcalloc_print_final_pn
MAIN_PATH=../../../instances/hadoop
TIME_LIMIT=3600 # seconds
MEM_LIMIT=80000 # 80 GB

result_path=timelogs
mkdir ${result_path}

pns=(3,0,0,0,0,us_west1.pn
    4,0,0,0,0,us_west2.pn
    5,0,0,0,0,us_mid1.pn
    6,0,0,0,0,us_mid2.pn)

vns=(instance_list_hadoop_4vm_rnd.txt
    instance_list_hadoop_6vm_rnd.txt
    instance_list_hadoop_8vm_rnd.txt
    instance_list_hadoop_10vm_rnd.txt
    instance_list_hadoop_12vm_rnd.txt
    instance_list_hadoop_14vm_rnd.txt)

for pn in ${pns[@]}; do
    # turn e.g. '3,0,0,0,0,us_west1.pn' into
    # array ['3', '0', '0', '0', '0', 'us_west1.pn']
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

