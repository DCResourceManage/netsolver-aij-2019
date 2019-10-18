#!/usr/bin/env python3
import os
import itertools
import tempfile
import subprocess
import sys
import re
os.environ['PYTHONHASHSEED']="1"
#os.environ['PYTHONPATH'] = os.getcwd()+"/solvers/monosat_py/" 
if not os.path.exists("timelogs"):
    os.makedirs("timelogs")
#this is the exact configuration used for the ijcai experiments
command = "runlim -t 3600 -s 80000 python3.6 ./vdcmapper_dist.py --configured --deallocate-prob=0 --no-detect-sat --ignore-ram --ignore-storage  --no-assert_all_physical_edges  --no-bitblast-addition --no-intrinsic_edge_constraints --intrinsic_edge_sets  --refine_flow --no-remove_flow_cycles --static-flow_caps --tree-addition --use_cover_for_refinement --no-anti-affinity --no-min-cores --no-affinity --directed "

instance_commands = ["--max-vms=6 --timelog=timelogs/config_results_vdcmapper_bcube8_6vm ../../../instances/fattree-bcube/BCube_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
		     "--max-vms=6 --timelog=timelogs/config_results_vdcmapper_bcube10_6vm ../../../instances/fattree-bcube/BCube_n10_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
		     "--max-vms=6 --timelog=timelogs/config_results_vdcmapper_bcube12_6vm ../../../instances/fattree-bcube/BCube_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
		     "--max-vms=6 --timelog=timelogs/config_results_vdcmapper_bcube16_6vm ../../../instances/fattree-bcube/BCube_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
		     "--max-vms=6 --timelog=timelogs/config_results_vdcmapper_fattree8_6vm ../../../instances/fattree-bcube/Fattree_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
		     "--max-vms=6 --timelog=timelogs/config_results_vdcmapper_fattree12_6vm ../../../instances/fattree-bcube/Fattree_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
		     "--max-vms=6 --timelog=timelogs/config_results_vdcmapper_fattree16_6vm ../../../instances/fattree-bcube/Fattree_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_6vm_rnd.txt",
  
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_bcube8_9vm ../../../instances/fattree-bcube/BCube_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_bcube10_9vm ../../../instances/fattree-bcube/BCube_n10_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_bcube12_9vm ../../../instances/fattree-bcube/BCube_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_bcube16_9vm ../../../instances/fattree-bcube/BCube_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_fattree8_9vm ../../../instances/fattree-bcube/Fattree_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_fattree12_9vm ../../../instances/fattree-bcube/Fattree_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		     "--max-vms=9 --timelog=timelogs/config_results_vdcmapper_fattree16_9vm ../../../instances/fattree-bcube/Fattree_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_9vm_rnd.txt",
		       
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_bcube8_12vm  ../../../instances/fattree-bcube/BCube_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_bcube10_12vm ../../../instances/fattree-bcube/BCube_n10_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_bcube12_12vm ../../../instances/fattree-bcube/BCube_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_bcube16_12vm ../../../instances/fattree-bcube/BCube_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_fattree8_12vm ../../../instances/fattree-bcube/Fattree_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_fattree12_12vm ../../../instances/fattree-bcube/Fattree_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     "--max-vms=12 --timelog=timelogs/config_results_vdcmapper_fattree16_12vm ../../../instances/fattree-bcube/Fattree_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_12vm_rnd.txt",
		     
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_bcube8_15vm ../../../instances/fattree-bcube/BCube_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt",
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_bcube10_15vm ../../../instances/fattree-bcube/BCube_n10_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt",
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_bcube12_15vm ../../../instances/fattree-bcube/BCube_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt",
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_bcube16_15vm ../../../instances/fattree-bcube/BCube_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt",
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_fattree8_15vm ../../../instances/fattree-bcube/Fattree_n8_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt",
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_fattree12_15vm ../../../instances/fattree-bcube/Fattree_n12_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt",
		     "--max-vms=15 --timelog=timelogs/config_results_vdcmapper_fattree16_15vm ../../../instances/fattree-bcube/Fattree_n16_k2_bw10_cpu16.pn ../../../instances/fattree-bcube/instance_list_15vm_rnd.txt"
		     ]

for inst in instance_commands:
  full_command = command + " " + inst
  print("Running: " + full_command)
  subprocess.call(full_command,shell=True,env=os.environ)
