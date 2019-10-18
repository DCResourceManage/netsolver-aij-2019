#!/usr/bin/env python3
import os
import itertools
import tempfile
import subprocess
import sys
import re

os.environ['PYTHONHASHSEED']="1"
os.environ['PYTHONPATH'] = os.getcwd()+"/solvers/monosat_py/" 
if not os.path.exists("timelogs"):
    os.makedirs("timelogs")

#this is the exact configuration used for the ijcai experiments
#note that ram is enforced, but storage is not, as described in the paper.

command = "runlim -t 3600 -s 80000 python3.6 vdcmapper_dist.py  --configured --no-detect-sat  --no-ignore-ram --ignore-storage  --no-assert_all_physical_edges  --no-bitblast-addition --no-intrinsic_edge_constraints --intrinsic_edge_sets --refine_flow --no-remove_flow_cycles --static-flow_caps --tree-addition --use_cover_for_refinement --no-anti-affinity --no-min-cores --no-affinity --no-directed "

instance_commands = [" --max-vms=4 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid1_4vm ../../../instances/hadoop/hadoop_extra/us-mid1.pn  ../../../instances/hadoop/instance_list_hadoop_4vm_rnd.txt",
		     " --max-vms=6 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid1_6vm ../../../instances/hadoop/hadoop_extra/us-mid1.pn  ../../../instances/hadoop/instance_list_hadoop_6vm_rnd.txt",
		     "--max-vms=8 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid1_8vm ../../../instances/hadoop/hadoop_extra/us-mid1.pn  ../../../instances/hadoop/instance_list_hadoop_8vm_rnd.txt",
		     " --max-vms=10 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid1_10vm ../../../instances/hadoop/hadoop_extra/us-mid1.pn  ../../../instances/hadoop/instance_list_hadoop_10vm_rnd.txt",
		     " --max-vms=12 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid1_12vm ../../../instances/hadoop/hadoop_extra/us-mid1.pn  ../../../instances/hadoop/instance_list_hadoop_12vm_rnd.txt",
		     
		     " --max-vms=4 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid2_4vm ../../../instances/hadoop/hadoop_extra/us-mid2.pn  ../../../instances/hadoop/instance_list_hadoop_4vm_rnd.txt",
		     " --max-vms=6 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid2_6vm ../../../instances/hadoop/hadoop_extra/us-mid2.pn  ../../../instances/hadoop/instance_list_hadoop_6vm_rnd.txt",
		     "--max-vms=8 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid2_8vm ../../../instances/hadoop/hadoop_extra/us-mid2.pn  ../../../instances/hadoop/instance_list_hadoop_8vm_rnd.txt",
		     " --max-vms=10 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid2_10vm ../../../instances/hadoop/hadoop_extra/us-mid2.pn  ../../../instances/hadoop/instance_list_hadoop_10vm_rnd.txt",
		     " --max-vms=12 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_mid2_12vm ../../../instances/hadoop/hadoop_extra/us-mid2.pn  ../../../instances/hadoop/instance_list_hadoop_12vm_rnd.txt",
		     
		    " --max-vms=4 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west2_4vm ../../../instances/hadoop/hadoop_extra/us-west2.pn  ../../../instances/hadoop/instance_list_hadoop_4vm_rnd.txt",
		     " --max-vms=6 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west2_6vm ../../../instances/hadoop/hadoop_extra/us-west2.pn  ../../../instances/hadoop/instance_list_hadoop_6vm_rnd.txt",
		     "--max-vms=8 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west2_8vm ../../../instances/hadoop/hadoop_extra/us-west2.pn  ../../../instances/hadoop/instance_list_hadoop_8vm_rnd.txt",
		     " --max-vms=10 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west2_10vm ../../../instances/hadoop/hadoop_extra/us-west2.pn  ../../../instances/hadoop/instance_list_hadoop_10vm_rnd.txt",
		     " --max-vms=12 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west2_12vm ../../../instances/hadoop/hadoop_extra/us-west2.pn  ../../../instances/hadoop/instance_list_hadoop_12vm_rnd.txt",
		    
		    " --max-vms=4 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west1_4vm ../../../instances/hadoop/hadoop_extra/us-west1.pn  ../../../instances/hadoop/instance_list_hadoop_4vm_rnd.txt",
		     " --max-vms=6 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west1_6vm ../../../instances/hadoop/hadoop_extra/us-west1.pn  ../../../instances/hadoop/instance_list_hadoop_6vm_rnd.txt",
		     "--max-vms=8 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west1_8vm ../../../instances/hadoop/hadoop_extra/us-west1.pn  ../../../instances/hadoop/instance_list_hadoop_8vm_rnd.txt",
		     " --max-vms=10 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west1_10vm ../../../instances/hadoop/hadoop_extra/us-west1.pn  ../../../instances/hadoop/instance_list_hadoop_10vm_rnd.txt",
		     " --max-vms=12 --max-cores=8 --max-ram=16 --timelog=timelogs/config_results_vdcmapper_hadoop_us_west1_12vm ../../../instances/hadoop/hadoop_extra/us-west1.pn  ../../../instances/hadoop/instance_list_hadoop_12vm_rnd.txt"
		     ]

for inst in instance_commands:
  full_command = command + " " + inst
  print("Running: " + full_command)
  subprocess.call(full_command,shell=True,env=os.environ)
