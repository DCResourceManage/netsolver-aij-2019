#!/bin/bash

TIME_LIMIT=3600
MEM_LIMIT=80000
SCRIPT_NAME=./netsolver-gurobi-incremental.py
UNDIRECTED=True
MULTI_THREAD=False
MAIN_PATH=../../../instances/fmcad/in_json

#tree16_2000

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_2000.20.pn ${MAIN_PATH}/instance_list_vn2_3.1 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_2000.20.vn3.1 &> results.tree16_2000.20.vn3.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_2000.20.pn ${MAIN_PATH}/instance_list_vn2_3.2 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_2000.20.vn3.2 &> results.tree16_2000.20.vn3.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_2000.20.pn ${MAIN_PATH}/instance_list_vn2_3.3 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_2000.20.vn3.3 &> results.tree16_2000.20.vn3.3.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_2000.20.pn ${MAIN_PATH}/instance_list_vn2_5.1 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_2000.20.vn5.1 &> results.tree16_2000.20.vn5.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_2000.20.pn ${MAIN_PATH}/instance_list_vn2_5.2 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_2000.20.vn5.2 &> results.tree16_2000.20.vn5.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_2000.20.pn ${MAIN_PATH}/instance_list_vn2_5.3 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_2000.20.vn5.3 &> results.tree16_2000.20.vn5.3.stdout

#tree16_400

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_400.20.pn ${MAIN_PATH}/instance_list_vn2_3.1 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_400.20.vn3.1 &> results.tree16_400.20.vn3.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_400.20.pn ${MAIN_PATH}/instance_list_vn2_3.2 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_400.20.vn3.2 &> results.tree16_400.20.vn3.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_400.20.pn ${MAIN_PATH}/instance_list_vn2_3.3 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_400.20.vn3.3 &> results.tree16_400.20.vn3.3.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_400.20.pn ${MAIN_PATH}/instance_list_vn2_5.1 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_400.20.vn5.1 &> results.tree16_400.20.vn5.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_400.20.pn ${MAIN_PATH}/instance_list_vn2_5.2 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_400.20.vn5.2 &> results.tree16_400.20.vn5.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_400.20.pn ${MAIN_PATH}/instance_list_vn2_5.3 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_400.20.vn5.3 &> results.tree16_400.20.vn5.3.stdout


#tree16_200

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_200.20.pn ${MAIN_PATH}/instance_list_vn2_3.1 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_200.20.vn3.1 &> results.tree16_200.20.vn3.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_200.20.pn ${MAIN_PATH}/instance_list_vn2_3.2 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_200.20.vn3.2 &> results.tree16_200.20.vn3.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_200.20.pn ${MAIN_PATH}/instance_list_vn2_3.3 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree16_200.20.vn3.3 &> results.tree16_200.20.vn3.3.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_200.20.pn ${MAIN_PATH}/instance_list_vn2_5.1 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_200.20.vn5.1 &> results.tree16_200.20.vn5.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_200.20.pn ${MAIN_PATH}/instance_list_vn2_5.2 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_200.20.vn5.2 &> results.tree16_200.20.vn5.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree16_200.20.pn ${MAIN_PATH}/instance_list_vn2_5.3 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree16_200.20.vn5.3 &> results.tree16_200.20.vn5.3.stdout



#tree_200

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree_200.20.pn ${MAIN_PATH}/instance_list_vn2_3.1 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree_200.20.vn3.1 &> results.tree_200.20.vn3.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree_200.20.pn ${MAIN_PATH}/instance_list_vn2_3.2 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree_200.20.vn3.2 &> results.tree_200.20.vn3.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree_200.20.pn ${MAIN_PATH}/instance_list_vn2_3.3 ${UNDIRECTED} 9 ${MULTI_THREAD} results.tree_200.20.vn3.3 &> results.tree_200.20.vn3.3.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree_200.20.pn ${MAIN_PATH}/instance_list_vn2_5.1 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree_200.20.vn5.1 &> results.tree_200.20.vn5.1.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree_200.20.pn ${MAIN_PATH}/instance_list_vn2_5.2 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree_200.20.vn5.2 &> results.tree_200.20.vn5.2.stdout

runlim -t ${TIME_LIMIT} -s ${MEM_LIMIT} python3.6 ${SCRIPT_NAME} ${MAIN_PATH}/tree_200.20.pn ${MAIN_PATH}/instance_list_vn2_5.3 ${UNDIRECTED} 15 ${MULTI_THREAD} results.tree_200.20.vn5.3 &> results.tree_200.20.vn5.3.stdout
