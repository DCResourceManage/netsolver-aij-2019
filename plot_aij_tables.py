import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glog

# lower bound on the time (in ms) for better visual plots
TIME_LOWER_BOUND = 0.001

def get_allocations_and_time(ifile):
    glog.debug('{}'.format(ifile))
    fd = open(ifile, 'r')
    lines = fd.readlines()
    start_read = False
    execution_time = []
    for line in lines:
        if not start_read and 'init' in line:
            start_read = True
            continue
        if 'done' in line:
            start_read = False
            break

        if start_read:
            # stop after cumulative 3600 seconds (one hour)
            cumul_time = float(line.split(' ')[2])
            if cumul_time > 3600:
                break
            execution_time.append(float(line.split(' ')[1]))

    glog.debug('len = {}'.format(len(execution_time)))
    if len(execution_time) > 0:
        ptile = round(np.percentile(execution_time, 50), 4)
    else:
        ptile = 0
    glog.debug('percentile = {}'.format(ptile))
    # log scale plotting does not accept zero. Make it one microsecond.
    if ptile <= TIME_LOWER_BOUND:
        ptile = TIME_LOWER_BOUND

    # number of allocations are equal to the number of recorded 
    # allocation times
    allocations = len(execution_time)
    glog.debug('allocations = {}'.format(allocations))

    fd.close()
    return (allocations, float(ptile))


def get_latex_stats(ifile):
    fd = open(ifile, 'r')
    lines = fd.readlines()
    start_read = False
    exceeded_thres = False
    execution_time = []
    for line in lines:
        if not start_read and 'init' in line:
            start_read = True
            continue
        if 'done' in line:
            start_read = False
            break

        if start_read:
            # stop after cumulative 3600 seconds (one hour)
            cumul_time = float(line.split(' ')[2])
            if cumul_time > 3600:
                exceeded_thres = True
                break
            execution_time.append(float(line.split(' ')[1]))

    # print('len = %s' % len(execution_time))

    ptile_min = round(min(execution_time), 3)
    ptile_mean = round(np.mean(execution_time), 3)
    ptile_med = round(np.percentile(execution_time, 50), 3)
    ptile_max = round(max(execution_time), 3)

    def _make_non_zero(ptile):
        # all numbers <0.001 should be made equal to 0.001 for consistency
        if ptile <= TIME_LOWER_BOUND:
            return '$<$0.001'
        else:
            return '%.3f' % ptile

    ptile_min = _make_non_zero(ptile_min)
    ptile_mean = _make_non_zero(ptile_mean)
    ptile_med = _make_non_zero(ptile_med)
    ptile_max = _make_non_zero(ptile_max)

    # number of allocs are equal to the number of recorded allocation times
    allocations = len(execution_time)
    glog.debug('allocations = {}'.format(allocations))

    if exceeded_thres:
        total_time = '$>$3600'
    else:
        total_time = '%.3f' % cumul_time

    fd.close()
    # return string format: & allocations & total_time & min & avg & med & max &
    return "& %d & %s & %s & %s & %s & %s" % (allocations, total_time,
            ptile_min, ptile_mean, ptile_med, ptile_max)


def print_latex(z3ar_files, secondnet_files, smt_files, ilp_files, vdc_names):
    format_str = 'alg_name & allocations & total_time & min & avg & med & max'
    z3ar_time = []
    secondnet_time = []
    smt_time = []
    ilp_time = []

    for index in range(len(vdc_names)):
        print('below are the results for %s (%s)' % (vdc_names[index], format_str))
        if len(z3ar_files) != 0:
            print('Z3-AR %s \\\\' % get_latex_stats(z3ar_files[index]))
        print('SecondNet %s \\\\' % get_latex_stats(secondnet_files[index]))
        print('\\netsolver-SMT %s \\\\' % get_latex_stats(smt_files[index]))
        print('\\netsolver-ILP %s \\\\' % get_latex_stats(ilp_files[index]))
        print # empty line for readability


def fig6():
    """ Figure that compares algorithms in FMCAD13 topologies """
    root_dir_prefix = './figures/fmcad-instances'
    servers_and_cores = [('200', '4'), ('200', '16'), ('400', '16'), 
                        ('2000', '16')]
    
    for (server, core) in servers_and_cores:
        plot_file_name = 'paper-plots/fig6_' + server + 'servers_' + core + 'cores.pdf'

        # 4 core files do not contain number of cores, only 16 core files do
        if core == '4':
            core_str = '_'
        else:
            core_str = core + '_'

        # z3 sample file name: fmcad13_z3_mode4_datacentertree_200.20_instance_vn2_3.1.log
        root_dir = os.path.join(root_dir_prefix, 'Z3-AR/timelogs/')
        prefix = 'fmcad13_z3_mode3_datacentertree'
        suffix = '.20_instance_vn2_'
        z3ar_files = [
            root_dir + prefix + core_str + server + suffix + '3.1.log',
            root_dir + prefix + core_str + server + suffix + '3.2.log',
            root_dir + prefix + core_str + server + suffix + '3.3.log',
            root_dir + prefix + core_str + server + suffix + '5.1.log',
            root_dir + prefix + core_str + server + suffix + '5.2.log',
            root_dir + prefix + core_str + server + suffix + '5.3.log']

        # secondnet sample file name: fmcad13_secondnet_datacentertree_200.20_instance_vn2_3.1.log
        root_dir = os.path.join(root_dir_prefix, 'secondnet/timelogs/')
        prefix = 'fmcad13_secondnet_datacentertree'
        suffix = '.20_instance_vn2_'
        secondnet_files = [
            root_dir + prefix + core_str + server + suffix + '3.1.log',
            root_dir + prefix + core_str + server + suffix + '3.2.log',
            root_dir + prefix + core_str + server + suffix + '3.3.log',
            root_dir + prefix + core_str + server + suffix + '5.1.log',
            root_dir + prefix + core_str + server + suffix + '5.2.log',
            root_dir + prefix + core_str + server + suffix + '5.3.log']

        
        # eg. fmcad13_vdcmapper_intrinsic-edge-sets-no-rnd-theory-order-theory-order-vsids-rnd-theory-freq-0-99_datacentertree_200.20.pn_instance_vn2_5.3.vn.log
        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/monosat-master/timelogs/')
        prefix = 'fmcad13_vdcmapper_intrinsic-edge-sets-no-rnd-theory-order-theory-order-vsids-rnd-theory-freq-0-99_datacentertree'
        suffix = '.20.pn_instance_vn2_'

        smt_files = [
            root_dir + prefix + core_str + server + suffix + '3.1.vn.log',
            root_dir + prefix + core_str + server + suffix + '3.2.vn.log',
            root_dir + prefix + core_str + server + suffix + '3.3.vn.log',
            root_dir + prefix + core_str + server + suffix + '5.1.vn.log',
            root_dir + prefix + core_str + server + suffix + '5.2.vn.log',
            root_dir + prefix + core_str + server + suffix + '5.3.vn.log']

        # ILP sample file name: results.tree_200.20.vn3.1
        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/')
        prefix = 'results.tree'
        suffix = '.20.vn'
        ilp_files = [
            root_dir + prefix + core_str + server + suffix + '3.1',
            root_dir + prefix + core_str + server + suffix + '3.2',
            root_dir + prefix + core_str + server + suffix + '3.3',
            root_dir + prefix + core_str + server + suffix + '5.1',
            root_dir + prefix + core_str + server + suffix + '5.2',
            root_dir + prefix + core_str + server + suffix + '5.3']

        vdc_names = ['T1 9VMs', 'T2 9VMs', 'T3 9VMs', 
                    'T1 15VMs', 'T2 15VMs', 'T3 15VMs']
        print('outputting data for {}'.format(plot_file_name))
        print_latex(z3ar_files, secondnet_files, smt_files, 
            ilp_files, vdc_names)


def fig7_fattree(vdc_sizes):
    root_dir_prefix = './figures/fattree-bcube'
    
    for size in vdc_sizes:
        if size == '8':
            servers = '128'
        elif size == '12':
            servers = '432'
        elif size == '16':
            servers = '1024'

        plot_file_name = 'paper-plots/fig7_fattree_' + servers + 'servers.pdf'

        root_dir = os.path.join(root_dir_prefix, 'secondnet/timelogs/')
        # sample: results.Fattree_n12_k2_bw10_cpu16.pn.instance_list_6vm_rnd.txt.log
        file_prefix = 'results.Fattree_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list'
        secondnet_files = [
            root_dir + file_prefix + size + file_suffix + '_6vm_rnd.txt.log',
            root_dir + file_prefix + size + file_suffix + '_9vm_rnd.txt.log',
            root_dir + file_prefix + size + file_suffix + '_12vm_rnd.txt.log',
            root_dir + file_prefix + size + file_suffix + '_15vm_rnd.txt.log']

        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/monosat-master/timelogs/')
        file_prefix = 'config_results_vdcmapper_fattree'
        smt_files = [
            root_dir + file_prefix + size + '_6vm',
            root_dir + file_prefix + size + '_9vm',
            root_dir + file_prefix + size + '_12vm',
            root_dir + file_prefix + size + '_15vm']

        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/')
        file_prefix = 'results.Fattree_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list_'
        ilp_files = [
            root_dir + file_prefix + size + file_suffix + '6vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '9vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '12vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '15vm_rnd.txt.log']

        vdc_names = ['6VMs', '9VMs', '12VMs', '15VMs']

        print('outputting data for {}'.format(plot_file_name))
        print_latex([], secondnet_files, smt_files, ilp_files, vdc_names)


def fig7_bcube(vdc_sizes):
    root_dir_prefix = './figures/fattree-bcube'
    
    for size in vdc_sizes:
        if size == '8':
            servers = '512'
        elif size == '10':
            servers = '1000'
        elif size == '12':
            servers = '1728'
        elif size == '16':
            servers = '4096'

        plot_file_name = 'paper-plots/fig7_bcube_' + servers + 'servers.pdf'
        root_dir = os.path.join(root_dir_prefix, 'secondnet/timelogs/')
        # sample: results.BCube_n8_k2_bw10_cpu16.pn.instance_list_6vm_rnd.txt.log
        file_prefix = 'results.BCube_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list'
        secondnet_files = [
            root_dir + file_prefix + size + file_suffix + '_6vm_rnd.txt.log',
            root_dir + file_prefix + size + file_suffix + '_9vm_rnd.txt.log',
            root_dir + file_prefix + size + file_suffix + '_12vm_rnd.txt.log',
            root_dir + file_prefix + size + file_suffix + '_15vm_rnd.txt.log']

        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/monosat-master/timelogs/')
        file_prefix = 'config_results_vdcmapper_bcube'
        smt_files = [
            root_dir + file_prefix + size + '_6vm',
            root_dir + file_prefix + size + '_9vm',
            root_dir + file_prefix + size + '_12vm', 
            root_dir + file_prefix + size + '_15vm']

        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/')
        file_prefix = 'results.BCube_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list_'
        ilp_files = [
            root_dir + file_prefix + size + file_suffix + '6vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '9vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '12vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '15vm_rnd.txt.log']

        vdc_names = ['6VMs', '9VMs', '12VMs', '15VMs']

        print('outputting data for {}'.format(plot_file_name))
        print_latex([], secondnet_files, smt_files, ilp_files, vdc_names)


def fig8():
    root_dir_prefix = './figures/hadoop'
    azs = ['mid1', 'mid2','west1', 'west2']
    
    for az in azs:
        plot_file_name = 'paper-plots/fig8_' + az + '.pdf'
        root_dir = os.path.join(root_dir_prefix, 'secondnet/timelogs/')
        # sample: results.us_mid1.pn.instance_list_hadoop_4vm_rnd.txt.log
        file_prefix = 'results.us_'
        file_suffix = '.pn.instance_list_hadoop'
        secondnet_files = [
            root_dir + file_prefix + az + file_suffix + '_4vm_rnd.txt.log',
            root_dir + file_prefix + az + file_suffix + '_6vm_rnd.txt.log',
            root_dir + file_prefix + az + file_suffix + '_8vm_rnd.txt.log',
            root_dir + file_prefix + az + file_suffix + '_10vm_rnd.txt.log',
            root_dir + file_prefix + az + file_suffix + '_12vm_rnd.txt.log']

        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/monosat-master/timelogs/')
        # sample: config_results_vdcmapper_hadoop_us_mid1_4vm
        file_prefix = 'config_results_vdcmapper_hadoop_us_'
        smt_files = [
            root_dir + file_prefix + az + '_4vm',
            root_dir + file_prefix + az + '_6vm',
            root_dir + file_prefix + az + '_8vm',
            root_dir + file_prefix + az + '_10vm',
            root_dir + file_prefix + az + '_12vm']

        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/')
        # sample: results.us-mid1.pn.instance_list_hadoop_4vm_rnd.txt.log
        file_prefix = 'results.us-'
        ilp_files = [
            root_dir + file_prefix + az + '.pn.instance_list_hadoop_4vm_rnd.txt.log',
            root_dir + file_prefix + az + '.pn.instance_list_hadoop_6vm_rnd.txt.log',
            root_dir + file_prefix + az + '.pn.instance_list_hadoop_8vm_rnd.txt.log',
            root_dir + file_prefix + az + '.pn.instance_list_hadoop_10vm_rnd.txt.log',
            root_dir + file_prefix + az + '.pn.instance_list_hadoop_12vm_rnd.txt.log']

        # `H-` stands for `Hadoop` (because Hadoop does not fit into fig label)
        vdc_names = ['H-4VM', 'H-6VM', 'H-8VM', 'H-10VM', 'H-12VM']

        print('outputting data for {}'.format(plot_file_name))
        print_latex([], secondnet_files, smt_files, ilp_files, vdc_names)


def mc_fattree(vdc_sizes):
    root_dir_prefix = './figures/fattree-bcube'
    for size in vdc_sizes:
        if size == '8':
            servers = '128'
        elif size == '12':
            servers = '432'
        elif size == '16':
            servers = '1024'

        total_allocs = {'sc': [], 'mc': []}
        total_time = {'sc': [], 'mc': []}

        plot_file_name = 'mc_fattree_' + servers + 'servers.pdf'
        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/')
        file_prefix = 'results.Fattree_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list_'
        sc_files = [
            root_dir + file_prefix + size + file_suffix + '6vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '9vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '12vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '15vm_rnd.txt.log']

        for file in sc_files:
            allocation, time = get_allocations_and_time(file)
            total_allocs['sc'].append(allocation)
            total_time['sc'].append(max(time, TIME_LOWER_BOUND))

        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp-mct/timelogs/')
        file_prefix = 'results.Fattree_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list_'
        mc_files = [
            root_dir + file_prefix + size + file_suffix + '6vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '9vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '12vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '15vm_rnd.txt.log']

        for file in mc_files:
            allocation, time = get_allocations_and_time(file)
            total_allocs['mc'].append(allocation)
            total_time['mc'].append(max(time, TIME_LOWER_BOUND))

        glog.info('{} total_allocs = {}'.format(plot_file_name, total_allocs))
        glog.info('{} total_time = {}'.format(plot_file_name, total_time))


        vdc_names = ['6VMs', '9VMs', '12VMs', '15VMs']


def mc_bcube(vdc_sizes):
    root_dir_prefix = './figures/fattree-bcube'

    for size in vdc_sizes:
        if size == '8':
            servers = '512'
        elif size == '10':
            servers = '1000'
        elif size == '12':
            servers = '1728'
        elif size == '16':
            servers = '4096'

        total_allocs = {'sc': [], 'mc': []}
        total_time = {'sc': [], 'mc': []}

        plot_file_name = 'mc_bcube_' + servers + 'servers.pdf'
        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/')
        file_prefix = 'results.BCube_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list_'
        sc_files = [
            root_dir + file_prefix + size + file_suffix + '6vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '9vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '12vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '15vm_rnd.txt.log']

        for file in sc_files:
            allocation, time = get_allocations_and_time(file)
            total_allocs['sc'].append(allocation)
            total_time['sc'].append(max(time, TIME_LOWER_BOUND))

        root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp-mct/timelogs/')
        file_prefix = 'results.BCube_n'
        file_suffix = '_k2_bw10_cpu16.pn.instance_list_'
        mc_files = [
            root_dir + file_prefix + size + file_suffix + '6vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '9vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '12vm_rnd.txt.log', 
            root_dir + file_prefix + size + file_suffix + '15vm_rnd.txt.log']

        for file in mc_files:
            allocation, time = get_allocations_and_time(file)
            total_allocs['mc'].append(allocation)
            total_time['mc'].append(max(time, TIME_LOWER_BOUND))

        glog.info('{} total_allocs = {}'.format(plot_file_name, total_allocs))
        glog.info('{} total_time = {}'.format(plot_file_name, total_time))

        vdc_names = ['6VMs', '9VMs', '12VMs', '15VMs']


if __name__ == "__main__":
    """ Output table data as latex. Run as $ python plot_aij_tables.py --h """
    CLI = argparse.ArgumentParser(
        description='Choose the figure to output the latex data for')
    group = CLI.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-f6',
        '--figure6',
        action='store_true',
        default=False,
        help='Compare algorithms in FMCAD13 topologies')

    group.add_argument(
        '-f7',
        '--figure7',
        action='store_true',
        default=False,
        help='Compare algorithms in FatTree and BCube topologies')

    group.add_argument(
        '-f8',
        '--figure8',
        action='store_true',
        default=False,
        help='Compare algorithms in commercial datacenter topologies using Hadoop instances')

    group.add_argument(
        '-mc',
        '--multi-core',
        action='store_true',
        default=False,
        help='Extract ILP results with and without multiple cores')

    ARGS = CLI.parse_args()

    if ARGS.figure6:
        fig6()
    elif ARGS.figure7:
        ft_sizes = ['8', '12', '16']
        fig7_fattree(ft_sizes)
         
        bc_sizes = ['8', '10', '12']
        fig7_bcube(bc_sizes)

    elif ARGS.figure8:
        fig8()
    elif ARGS.multi_core:
        ft_sizes = ['8', '12', '16']
        mc_fattree(ft_sizes)
         
        bc_sizes = ['8', '10', '12']
        mc_bcube(bc_sizes)

    else:
        glog.error('ERROR: invalid option.')
        sys.exit(1)
