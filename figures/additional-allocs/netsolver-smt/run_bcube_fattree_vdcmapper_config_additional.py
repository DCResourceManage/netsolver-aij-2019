#!/usr/bin/env python3

import os
import itertools
import tempfile
import subprocess
import sys
import re
import glog

os.environ['PYTHONHASHSEED']="1"
if not os.path.exists("timelogs"):
    os.makedirs("timelogs")

#this is the exact configuration used for the ijcai experiments
command = "runlim -t 3600 -s 80000 python3.6 ./vdcmapper_dist.py --configured --deallocate-prob=0 --no-detect-sat --ignore-ram --ignore-storage  --no-assert_all_physical_edges  --no-bitblast-addition --no-intrinsic_edge_constraints --intrinsic_edge_sets  --refine_flow --no-remove_flow_cycles --static-flow_caps --tree-addition --use_cover_for_refinement --no-anti-affinity --no-min-cores --no-affinity --directed "

MAIN_PATH = '../secondnet/final_pns_json'
INSTANCE_PATH = '../../../instances/fattree-bcube'
vdc_size_config = {'6': '--max-vms=6', '9': '--max-vms=9',
        '12': '--max-vms=12', '15': '--max-vms=15'}
datacenters = {'bcube8': 'BCube_n8_k2_bw10_cpu16.pn',
        'bcube10': 'BCube_n10_k2_bw10_cpu16.pn',
        'bcube12': 'BCube_n12_k2_bw10_cpu16.pn',
        'bcube16': 'BCube_n16_k2_bw10_cpu16.pn',
        'fattree8': 'Fattree_n8_k2_bw10_cpu16.pn',
        'fattree12': 'Fattree_n12_k2_bw10_cpu16.pn',
        'fattree16': 'Fattree_n16_k2_bw10_cpu16.pn'}

for vdc_size, conf in vdc_size_config.items():
    for dc_name, dc_file in datacenters.items():
        pn = '{}/final.{}.instance_list_{}vm_rnd.txt.pn.json'.format(MAIN_PATH, dc_file, vdc_size)
        vn = '{}/instance_list_{}vm_rnd.txt'.format(INSTANCE_PATH, vdc_size)
        full_cmd = '{} {} --timelog=./timelogs/config_results_vdcmapper_{}_{}vm {} {}'.format(
                command, conf, dc_name, vdc_size, pn, vn)
        glog.info('--- started: {}'.format(full_cmd))
        sys.stdout.flush();
        sys.stderr.flush();
        subprocess.call(full_cmd, shell=True, env=os.environ)
        sys.stdout.flush();
        sys.stderr.flush();
        glog.info('+++ compelted: {}'.format(full_cmd))

glog.info('+++ all done, bye +++')

