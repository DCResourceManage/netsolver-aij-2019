#!/usr/bin/env python3
import os
import itertools
import tempfile
import subprocess
import sys
import re
import glog

RESULTS_DIR = './timelogs'
MAIN_PATH = '../../../instances/hadoop/hadoop_extensions'

os.environ['PYTHONHASHSEED']="1"
os.environ['PYTHONPATH'] = os.getcwd()+"/solvers/monosat_py/" 
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

command = 'runlim -t 3600 -s 80000 python3.6 vdcmapper_extensions.py --intrinsic_edge_sets --theory-order-vsids --rnd-theory-freq 0.99 --decide-opt-lits --binary-search --no-anti-affinity'

ext_configs = {'softaffinity': '--no-min-cores --no-min-servers --affinity --soft-affinity',
        'minserver': '--no-soft-affinity --no-min-cores --no-affinity --min-servers',
        'mincores': '--no-soft-affinity --no-min-servers --no-affinity --min-cores'}

vdc_size_config = {'4': '--max-vms=4 --max-cores=8 --max-ram=16',
        '10': '--max-vms=10 --max-cores=8 --max-ram=16',
        '15': '--max-vms=15 --max-cores=8 --max-ram=16'}
bw_vals = ['1bw', '2bw']
datacenters = ['us-west1']
#datacenters = ['us-west1', 'us-west2', 'us-mid1', 'us-mid2']

for extension, ext_config in ext_configs.items():
    if not os.path.exists(os.path.join(RESULTS_DIR, extension)):
        os.makedirs(os.path.join(RESULTS_DIR, extension))

    for bw_val in bw_vals:
        for vdc_size, conf in vdc_size_config.items():
            for dc in datacenters:
                pn = '{}/{}.pn'.format(MAIN_PATH, dc)
                vn = '{}/hadoop_{}_{}.vn'.format(MAIN_PATH, vdc_size, bw_val)
                full_cmd = '{} {} --timelog={}/{}/config_results_hadoop_{}_{}vm_{} {} {} {}'.format(
                        command, conf, RESULTS_DIR, extension, dc, vdc_size, bw_val, ext_config, pn, vn)
                glog.info('--- started: {}'.format(full_cmd))
                sys.stdout.flush();
                sys.stderr.flush();
                subprocess.call(full_cmd, shell=True, env=os.environ)
                sys.stdout.flush();
                sys.stderr.flush();
                glog.info('+++ compelted: {}'.format(full_cmd))

glog.info('+++ all done, bye +++')

