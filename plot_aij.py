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


def plot3bars(plot_file_name,
    secondnet_allocs, smt_allocs, ilp_allocs,
    secondnet_time, smt_time, ilp_time,
    vdc_names, time_range, allocs_range):
    glog.info('started plotting')
    n_groups = len(secondnet_allocs)
    fig, ax = plt.subplots(figsize=(8, 2.25))

    bar_width = 0.18
    index = np.arange(n_groups)
    opacity = 0.4

    secondnet_rects = plt.bar(index, secondnet_allocs, bar_width,
                            alpha=opacity,
                            color='r',
                            hatch='--',
                            label='SecondNet')

    smt_rects = plt.bar(index + bar_width, smt_allocs, 
                            bar_width,
                            alpha=opacity,
                            color='g',
                            hatch='\\\\',
                            label='NetSolver-SMT')

    ilp_rects = plt.bar(index + 2*bar_width, ilp_allocs, bar_width,
                            alpha=opacity,
                            color='b',
                            hatch='//',
                            label='NetSolver-ILP')

    axes = plt.gca()
    plt.ylabel('Number of VDC allocations')
    plt.yticks(allocs_range)

    plt.xlabel('VDC type')
    plt.xticks(index + 2*bar_width, vdc_names)

    # plot right-vertical axis for execution time
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(time_range)

    ax2.plot(index+bar_width/2, secondnet_time, 'r^', markersize=10, 
            label='SecondNet')
    ax2.plot(index+3*bar_width/2, smt_time, 'gs', markersize=8,
            label='NetSolver-SMT')
    ax2.plot(index+5*bar_width/2, ilp_time, 'b*', markersize=10, label='NetSolver-ILP')

    ax2.set_ylabel('Median per-VDC allocation time (s)')

    # put legends
    ax.legend(loc='upper left', bbox_to_anchor=(0, 2.2), numpoints=1, ncol=1,
        title='Left Scale, VDC Allocations', frameon=False)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 2.2),
        numpoints=1, ncol=1, frameon=False,
        title='Right Log Scale, Time Per Allocation')

    plt.draw()
    final_figure = plt.gcf()
    final_figure.savefig(plot_file_name, bbox_inches='tight', dpi=200)
    glog.info('plotting done, see {}'.format(plot_file_name))
    plt.close(fig)


def plot3bars_ext(plot_file_name,
    affinity_allocs, minserver_allocs, hotspot_allocs,
    affinity_time, minserver_time, hotspot_time,
    vdc_names, time_range, allocs_range):
    glog.info('started plotting')
    n_groups = len(affinity_allocs)
    fig, ax = plt.subplots(figsize=(8, 2.25))

    bar_width = 0.18
    index = np.arange(n_groups)
    opacity = 0.4

    affinity_rects = plt.bar(index, affinity_allocs, bar_width,
                            alpha=opacity,
                            color='r',
                            hatch='--',
                            label='affinity')

    minserver_rects = plt.bar(index + bar_width, minserver_allocs, 
                            bar_width,
                            alpha=opacity,
                            color='g',
                            hatch='\\\\',
                            label='min-server')

    hotspot_rects = plt.bar(index + 2*bar_width, hotspot_allocs,
                            bar_width,
                            alpha=opacity,
                            color='b',
                            hatch='//',
                            label='no-hotspot')

    axes = plt.gca()
    plt.ylabel('Number of VDC allocations')
    plt.yticks(allocs_range)

    plt.xlabel('VDC type')
    plt.xticks(index + 2*bar_width, vdc_names)

    # plot right-vertical axis for execution time
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(time_range)

    ax2.plot(index+bar_width/2, affinity_time, 'r+', markersize=10, 
            label='affinity')
    ax2.plot(index+3*bar_width/2, minserver_time, 'gx', markersize=8,
            label='min-server')
    ax2.plot(index+5*bar_width/2, hotspot_time, 'bd', markersize=10, 
            label='no-hotspot')

    ax2.set_ylabel('Median per-VDC allocation time (s)')

    # put legends
    ax.legend(loc='upper left', bbox_to_anchor=(0, 2.2), numpoints=1, ncol=1,
        title='Left Scale, VDC Allocations', frameon=False)    
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 2.2), 
        numpoints=1, ncol=1, frameon=False,
        title='Right Log Scale, Time Per Allocation')

    plt.draw()
    final_figure = plt.gcf()
    final_figure.savefig(plot_file_name, bbox_inches='tight', dpi=200)
    glog.info('plotting done, see {}'.format(plot_file_name))
    plt.close(fig)


def plot4bars(plot_file_name,
    z3ar_allocs, secondnet_allocs, smt_allocs, ilp_allocs,
    z3ar_time, secondnet_time, smt_time, ilp_time,
    vdc_names, time_range, allocs_range):
    glog.info('started plotting')
    n_groups = len(secondnet_allocs)
    fig, ax = plt.subplots(figsize=(8, 2.25))

    bar_width = 0.18
    index = np.arange(n_groups)
    opacity = 0.4

    z3ar_rects = plt.bar(index, z3ar_allocs, 
                            bar_width,
                            alpha=opacity,
                            color='y',
                            hatch='//',
                            label='Z3-AR')

    secondnet_rects = plt.bar(index + bar_width, secondnet_allocs, bar_width,
                            alpha=opacity,
                            color='r',
                            hatch='--',
                            label='SecondNet')

    smt_rects = plt.bar(index + 2*bar_width, smt_allocs, 
                            bar_width,
                            alpha=opacity,
                            color='g',
                            hatch='\\\\',
                            label='NetSolver-SMT')

    ilp_rects = plt.bar(index + 3*bar_width, ilp_allocs, bar_width,
                            alpha=opacity,
                            color='b',
                            hatch='//',
                            label='NetSolver-ILP')

    axes = plt.gca()
    plt.ylabel('Number of VDC allocations')
    plt.yticks(allocs_range)

    plt.xlabel('VDC type')
    plt.xticks(index + 2*bar_width, vdc_names)

    # plot right-vertical axis for execution time
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(time_range)

    ax2.plot(index+bar_width/2, z3ar_time, 'y.', markersize=15,
            label='Z3-AR')
    ax2.plot(index+3*bar_width/2, secondnet_time, 'r^', markersize=10, 
            label='SecondNet')
    ax2.plot(index+5*bar_width/2, smt_time, 'gs', markersize=8,
            label='NetSolver-SMT')
    ax2.plot(index+7*bar_width/2, ilp_time, 'b*', markersize=10, label='NetSolver-ILP')

    ax2.set_ylabel('Median per-VDC allocation time (s)')

    # put legends
    ax.legend(loc='upper left', bbox_to_anchor=(0, 2.2), numpoints=1, ncol=1,
        title='Left Scale, VDC Allocations', frameon=False)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 2.2), 
        numpoints=1, ncol=1, frameon=False,
        title='Right Log Scale, Time Per Allocation')

    plt.draw()
    final_figure = plt.gcf()
    final_figure.savefig(plot_file_name, bbox_inches='tight', dpi=200)
    glog.info('plotting done, see {}'.format(plot_file_name))
    plt.close(fig)


def plot5bars(plot_file_name,
    sp_allocs, ps_allocs, secondnet_allocs, smt_allocs, ilp_allocs,
    sp_time, ps_time, secondnet_time, smt_time, ilp_time,
    vdc_names, time_range, allocs_range):
    glog.info('started plotting')
    n_groups = len(secondnet_allocs)
    fig, ax = plt.subplots(figsize=(8, 2.25))

    bar_width = 0.18
    index = np.arange(n_groups)
    opacity = 0.4

    sp_rects = plt.bar(index, sp_allocs,
                            bar_width,
                            alpha=opacity,
                            color='c',
                            hatch='|',
                            label='GAR-SP')

    ps_rects = plt.bar(index + bar_width, ps_allocs,
                            bar_width,
                            alpha=opacity,
                            color='y',
                            hatch='//',
                            label='GAR-PS')

    secondnet_rects = plt.bar(index + 2*bar_width, secondnet_allocs,
                            bar_width,
                            alpha=opacity,
                            color='r',
                            hatch='--',
                            label='SecondNet')

    netsolver_rects = plt.bar(index + 3*bar_width, smt_allocs,
                            bar_width,
                            alpha=opacity,
                            color='g',
                            hatch='\\\\',
                            label='NetSolver-SMT')

    gurobi_rects = plt.bar(index + 4 * bar_width, ilp_allocs,
                              bar_width,
                              alpha=opacity,
                              color='b',
                              hatch='//',
                              label='NetSolver-ILP')

    # put final metadata and plot
    axes = plt.gca()
    plt.xlabel('VDC type')
    plt.ylabel('Number of VDC allocations')
    plt.xticks(index + 1.5*bar_width, vdc_names, rotation=0)
    plt.yticks(allocs_range)

    # plot right-vertical axis for execution time
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(time_range)

    ax2.plot(index+1*bar_width/2, sp_time, 'c+', markersize=8, label='GAR-SP')
    ax2.plot(index+3*bar_width/2, ps_time, 'y.', markersize=12, label='GAR-PS')
    ax2.plot(index+5*bar_width/2, secondnet_time, 'r^', markersize=8, 
        label='SecondNet')
    ax2.plot(index+7*bar_width/2, smt_time, 'gs', markersize=8,
        label='NetSolver-SMT')
    ax2.plot(index+9*bar_width/2, ilp_time, 'b*', markersize=8,
        label='NetSolver-ILP')

    ax2.set_ylabel('Median per-VDC allocation time (s)')

    # put legends
    ax.legend(loc='upper left', bbox_to_anchor=(0, 2.2), numpoints=1, ncol=1,
        title='Left Scale, VDC Allocations', frameon=False)    
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 2.2), 
        numpoints=1, ncol=1, frameon=False,
        title='Right Log Scale, Time Per Allocation')

    plt.draw()
    final_figure = plt.gcf()
    final_figure.savefig(plot_file_name, bbox_inches='tight', dpi=200)
    glog.info('plotting done, see {}'.format(plot_file_name))
    plt.close(fig)


def plot_topology(plot_file_name, z3ar_files, secondnet_files, smt_files, 
                ilp_files, vdc_names, time_range, allocs_range):
    z3ar_allocs, secondnet_allocs, smt_allocs, ilp_allocs = [], [], [], []
    z3ar_time, secondnet_time, smt_time, ilp_time = [], [], [], []

    # some tables will not have z3ar data, that's okay. We don't need to treat 
    # them separately.
    for file in z3ar_files:
        allocation, time = get_allocations_and_time(file)
        z3ar_allocs.append(allocation)
        z3ar_time.append(time)

    for file in secondnet_files:
        allocation, time = get_allocations_and_time(file)
        secondnet_allocs.append(allocation)
        secondnet_time.append(time)

    for file in smt_files:
        allocation, time = get_allocations_and_time(file)
        smt_allocs.append(allocation)
        smt_time.append(time)

    for file in ilp_files:
        allocation, time = get_allocations_and_time(file)
        ilp_allocs.append(allocation)
        ilp_time.append(time)

    glog.info('z3ar_allocs = {}'.format(z3ar_allocs))
    glog.info('z3ar_time = {}'.format(z3ar_time))

    glog.info('secondnet_allocs = {}'.format(secondnet_allocs))
    glog.info('secondnet_time = {}'.format(secondnet_time))

    glog.info('smt_allocs = {}'.format(smt_allocs))
    glog.info('smt_time = {}'.format(smt_time))

    glog.info('ilp_allocs = {}'.format(ilp_allocs))
    glog.info('ilp_time = {}'.format(ilp_time))

    if len(z3ar_files) == 0:
        plot3bars(plot_file_name, secondnet_allocs, smt_allocs, ilp_allocs,
            secondnet_time, smt_time, ilp_time,
            vdc_names, time_range, allocs_range)
    else:
        plot4bars(plot_file_name,
            z3ar_allocs, secondnet_allocs, smt_allocs, ilp_allocs,
            z3ar_time, secondnet_time, smt_time, ilp_time,
            vdc_names, time_range, allocs_range)


def fig6():
    """ Figure that compares algorithms in FMCAD13 topologies """
    root_dir_prefix = './figures/fmcad-instances'
    time_range = [TIME_LOWER_BOUND,100]
    servers_and_cores = [('200', '4'), ('200', '16'), ('400', '16'), 
                        ('2000', '16')]
    # predefine ranges here to have a determenistic (and nicely rounded)
    # vertical axis for number of VDC allocations. We get these values after
    # running the code once to print actual range values.
    allocs_ranges = [np.arange(0, 101, 25), np.arange(0, 401, 100),
            np.arange(0, 801, 200), np.arange(0, 4001, 1000)]
    range_index = 0

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
        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/')
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
        plot_topology(plot_file_name, z3ar_files, secondnet_files, smt_files, 
                ilp_files, vdc_names, time_range, allocs_ranges[range_index])
        range_index += 1


def fig7_fattree(vdc_sizes):
    root_dir_prefix = './figures/fattree-bcube'
    time_range = [TIME_LOWER_BOUND, 100]
    # predefine ranges here to have a determenistic (and nicely rounded)
    # vertical axis for number of VDC allocations. We get these values after
    # running the code once to print actual range values.
    allocs_ranges = [np.arange(0, 401, 100), np.arange(0, 1201, 300),
        np.arange(0, 1801, 300)]
    range_index = 0

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

        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/')
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

        plot_topology(plot_file_name, [], secondnet_files, smt_files, 
            ilp_files, vdc_names, time_range, allocs_ranges[range_index])
        range_index += 1


def fig7_bcube(vdc_sizes):
    root_dir_prefix = './figures/fattree-bcube'
    time_range = [TIME_LOWER_BOUND, 100]
    # predefine ranges here to have a determenistic (and nicely rounded)
    # vertical axis for number of VDC allocations. We get these values after
    # running the code once to print actual range values.
    allocs_ranges = [np.arange(0, 1601, 400), np.arange(0, 2801, 700),
            np.arange(0, 4801, 1200)]
    range_index = 0

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

        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/')
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

        plot_topology(plot_file_name, [], secondnet_files, smt_files, 
            ilp_files, vdc_names, time_range, allocs_ranges[range_index])
        range_index += 1


def fig8():
    root_dir_prefix = './figures/hadoop'
    time_range = [TIME_LOWER_BOUND, 10]
    azs = ['mid1', 'mid2','west1', 'west2']
    # predefine ranges here to have a determenistic (and nicely rounded)
    # vertical axis for number of VDC allocations. We get these values after
    # running the code once to print actual range values.
    allocs_ranges = [np.arange(0, 161, 40), np.arange(0, 321, 80),
        np.arange(0, 481, 120), np.arange(0, 121, 30)]
    range_index = 0

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

        root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/')
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

        plot_topology(plot_file_name, [], secondnet_files, smt_files, 
                ilp_files, vdc_names, time_range, allocs_ranges[range_index])
        range_index += 1


def fig9():
    azs = ['west1', 'west2']
    time_range = [TIME_LOWER_BOUND, 100]
    allocs_range = []

    for az in azs:
        plot_file_name = 'paper-plots/fig9_' + az + '.pdf'
        if az == 'west1':
            secondnet_allocs = [2400, 1740, 786, 240, 480, 0]
            smt_allocs = [2399, 2399, 943, 932, 613, 600]
            ilp_allocs = [2399, 2397, 960, 707, 640, 320]
            sp_allocs = [632, 355, 241, 121, 81, 1]
            ps_allocs = [1297, 617, 239, 119, 79, 0]

            secondnet_time = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
            smt_time = [0.1, 0.1, 0.2, 0.2, 0.3, 0.4]
            ilp_time = [0.125294, 0.126046, 0.33834, 1.49471, 0.561857, 5.45]
            sp_time = [0.5, 0.6, 1.3, 1.1, 1.9, 2]
            ps_time = [3.4, 3.7, 10.5, 10.5, 17.5, 0]
            
            allocs_range = np.arange(0, 2501, 500)

        elif az == 'west2':
            secondnet_allocs = [560, 406, 184, 56, 112, 0]
            smt_allocs = [560, 560, 221, 217, 146, 143]
            ilp_allocs = [558, 554, 223, 145, 149, 49]
            sp_allocs = [185, 97, 56, 29, 20, 1]
            ps_allocs = [361, 170, 54, 27, 18, 0]

            secondnet_time = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
            smt_time = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]           
            ilp_time = [0.0314379, 0.0343081, 0.0916209, 0.192709, 0.144476, 0.276443]
            sp_time = [0.001, 0.001, 0.001, 0.001, 0.1, 0.3]
            ps_time = [0.2, 0.2, 0.7, 0.7, 1, 0]

            allocs_range = np.arange(0, 601, 100)

        vdc_names = ['Hadoop\n4VMs, 1Gb', 'Hadoop\n4VMs, 2Gb', 'Hadoop\n10VMs, 1Gb', 'Hadoop\n10VMs, 2Gb','Hadoop\n15VMs, 1Gb', 'Hadoop\n15VMs, 2Gb']

        plot5bars(plot_file_name,
            sp_allocs, ps_allocs, secondnet_allocs, smt_allocs, ilp_allocs,
            sp_time, ps_time, secondnet_time, smt_time, ilp_time,
            vdc_names, time_range, allocs_range)


def plot_robustness(plot_file_name, secondnet_allocs, secondnet_time,
    smt_flag, extra_allocs, extra_time, vdc_names, time_range, allocs_range):
    
    glog.info('started plotting')
    n_groups = len(vdc_names)
    fig, ax = plt.subplots(figsize=(8, 2.25))

    bar_width = 0.25
    bar_space = bar_width + 0.0
    index = np.arange(n_groups)

    opacity = 0.6

    ft_secondnet_rects = plt.bar(index, secondnet_allocs['fat_tree'], 
            bar_width,
            alpha=opacity,
            color='r',
            hatch='--',
            label='SecondNet')

    bcube_secondnet_rects = plt.bar(index+bar_space, secondnet_allocs['bcube'],
            bar_width,
            alpha=opacity,
            color='r',
            hatch='--')

    if smt_flag:
        fig_label = 'NetSolver-SMT'
        fig_color = 'g'
        fig_hatch = '\\\\'
    else:
        fig_label = 'NetSolver-ILP'
        fig_color = 'b'
        fig_hatch = '//'

    ft_extra_rects = plt.bar(index, extra_allocs['fat_tree'],
            bar_width,
            bottom=secondnet_allocs['fat_tree'],
            alpha=opacity,
            color=fig_color,
            hatch=fig_hatch)

    bcube_extra_rects = plt.bar(index + bar_space, extra_allocs['bcube'],
            bar_width,
            bottom=secondnet_allocs['bcube'],
            color=fig_color,
            hatch=fig_hatch,
            label=fig_label)

    # put final metadata and plot
    axes = plt.gca()
    plt.xlabel('VDC type')
    plt.ylabel('Number of VDC allocations')
    xticks = [0.15, 1.15, 2.15, 3.15]
    plt.xticks(xticks, ['FatTree   BCube\n'+ii for ii in vdc_names])
    plt.yticks(allocs_range)

    # plot right-vertical axis for execution time
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(time_range)

    ax2.plot(index+bar_space/2, secondnet_time['fat_tree'], 'r^',
        markersize=10)
    ax2.plot(index+3*bar_space/2, secondnet_time['bcube'], 'r^',
        markersize=10, label='SecondNet')
    if smt_flag:
        ax2.plot(index+bar_space/2, extra_time['fat_tree'],'gs',
            markersize=10)
        ax2.plot(index+3*bar_space/2, extra_time['bcube'],'gs',
            markersize=10, label='NetSolver-SMT')
    else:
        ax2.plot(index+bar_space/2, extra_time['fat_tree'],'b*',
            markersize=10)
        ax2.plot(index+3*bar_space/2, extra_time['bcube'],'b*',
            markersize=10, label='NetSolver-ILP')

    ax2.set_ylabel('Median per-VDC allocation time (s)')

    # put legends
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.80), numpoints=1, ncol=1,
        title='Left Scale, VDC Allocations', frameon=False)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 1.80),
        numpoints=1, ncol=1, frameon=False,
        title='Right Log Scale, Time Per Allocation')

    plt.draw()
    final_figure = plt.gcf()
    final_figure.savefig(plot_file_name, bbox_inches='tight', dpi=100)
    glog.info('plotting done, see {}'.format(plot_file_name))
    plt.close(fig)


def fig10():
    time_range = [TIME_LOWER_BOUND, 100]
    vdc_names = ['6 VMs', '9 VMs', '12 VMs', '15 VMs']

    ### Get allocations from SecondNet ###
    secondnet_allocs = {'bcube': [], 'fat_tree': []}
    secondnet_time = {'bcube': [], 'fat_tree': []}
    root_dir_prefix = './figures/additional-allocs/'
    root_dir = os.path.join(root_dir_prefix, 'secondnet/timelogs/')

    # sample: results.BCube_n8_k2_bw10_cpu16.pn.instance_list_6vm_rnd.txt.log
    file_prefix = 'results.BCube_n8_k2_bw10_cpu16.pn.instance_list'
    file_suffix = '_rnd.txt.log'
    secondnet_files = [
        root_dir + file_prefix + '_6vm' + file_suffix, 
        root_dir + file_prefix + '_9vm' + file_suffix, 
        root_dir + file_prefix + '_12vm' + file_suffix, 
        root_dir + file_prefix + '_15vm' + file_suffix]

    for file in secondnet_files:
        allocation, time = get_allocations_and_time(file)
        secondnet_allocs['bcube'].append(allocation)
        secondnet_time['bcube'].append(max(time, time_range[0]))

    # sample: results.Fattree_n12_k2_bw10_cpu16.pn.instance_list_6vm_rnd.txt.log
    file_prefix = 'results.Fattree_n12_k2_bw10_cpu16.pn.instance_list'
    file_suffix = '_rnd.txt.log'
    secondnet_files = [
        root_dir + file_prefix + '_6vm' + file_suffix, 
        root_dir + file_prefix + '_9vm' + file_suffix, 
        root_dir + file_prefix + '_12vm' + file_suffix, 
        root_dir + file_prefix + '_15vm' + file_suffix]

    for file in secondnet_files:
        allocation, time = get_allocations_and_time(file)
        secondnet_allocs['fat_tree'].append(allocation)
        secondnet_time['fat_tree'].append(max(time, time_range[0]))
        
    glog.debug('secondnet_allocs = {}'.format(secondnet_allocs))
    glog.debug('secondnet_time = {}'.format(secondnet_time))

    ### Get addtional allocations from SMT ###
    smt_extra_allocs = {'bcube': [], 'fat_tree': []}
    smt_time = {'bcube': [], 'fat_tree': []}

    root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/')
    # sample: config_results_vdcmapper_bcube8_6vm
    file_prefix = 'config_results_vdcmapper_bcube8'
    smt_files = [
        root_dir + file_prefix + '_6vm',
        root_dir + file_prefix + '_9vm',
        root_dir + file_prefix + '_12vm',
        root_dir + file_prefix + '_15vm']

    for file in smt_files:
        allocation, time = get_allocations_and_time(file)
        smt_extra_allocs['bcube'].append(allocation)
        smt_time['bcube'].append(time)

    # sample: config_results_vdcmapper_fattree12_6vm
    file_prefix = 'config_results_vdcmapper_fattree12'
    smt_files = [
        root_dir + file_prefix + '_6vm',
        root_dir + file_prefix + '_9vm',
        root_dir + file_prefix + '_12vm',
        root_dir + file_prefix + '_15vm']

    for file in smt_files:
        allocation, time = get_allocations_and_time(file)
        smt_extra_allocs['fat_tree'].append(allocation)
        smt_time['fat_tree'].append(time)

    glog.debug('smt_extra_allocs = {}'.format(smt_extra_allocs))
    glog.debug('smt_time = {}'.format(smt_time))

    ### Get addtional allocations from ILP ###
    ilp_extra_allocs = {'bcube': [], 'fat_tree': []}
    ilp_time = {'bcube': [], 'fat_tree': []}

    root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/')
    # sample: results.BCube_n8_k2_bw10_cpu16.pn.instance_list_6vm_rnd.txt.log
    file_prefix = 'results.BCube_n8_k2_bw10_cpu16.pn.instance_list'
    file_suffix = '_rnd.txt.log'
    ilp_files = [
        root_dir + file_prefix + '_6vm' + file_suffix,
        root_dir + file_prefix + '_9vm' + file_suffix,
        root_dir + file_prefix + '_12vm' + file_suffix,
        root_dir + file_prefix + '_15vm' + file_suffix]

    for file in ilp_files:
        allocation, time = get_allocations_and_time(file)
        ilp_extra_allocs['bcube'].append(allocation)
        ilp_time['bcube'].append(time)

    # sample: results.Fattree_n12_k2_bw10_cpu16.pn.instance_list_6vm_rnd.txt.log
    file_prefix = 'results.Fattree_n12_k2_bw10_cpu16.pn.instance_list'
    file_suffix = '_rnd.txt.log'
    ilp_files = [
        root_dir + file_prefix + '_6vm' + file_suffix,
        root_dir + file_prefix + '_9vm' + file_suffix,
        root_dir + file_prefix + '_12vm' + file_suffix,
        root_dir + file_prefix + '_15vm' + file_suffix]

    for file in ilp_files:
        allocation, time = get_allocations_and_time(file)
        ilp_extra_allocs['fat_tree'].append(allocation)
        ilp_time['fat_tree'].append(time)

    # predefine ranges here to have a determenistic (and nicely rounded)
    # vertical axis for number of VDC allocations. We get these values after
    # running the code once to print actual range values.
    allocs_range = np.arange(0, 1601, 400)

    plot_file_name = 'paper-plots/fig10-smt.pdf'
    smt_flag = True
    plot_robustness(plot_file_name, secondnet_allocs, secondnet_time,
        smt_flag, smt_extra_allocs, smt_time,
        vdc_names, time_range, allocs_range)

    plot_file_name = 'paper-plots/fig10-ilp.pdf'
    smt_flag = False
    plot_robustness(plot_file_name, secondnet_allocs, secondnet_time,
        smt_flag, ilp_extra_allocs, ilp_time,
        vdc_names, time_range, allocs_range)


def fig12(topologies):
    time_range = [0.05, 200]
    vdc_names = ['Hadoop\n4VMs, 1Gb', 'Hadoop\n4VMs, 2Gb',
        'Hadoop\n10VMs, 1Gb', 'Hadoop\n10VMs, 2Gb',
        'Hadoop\n15VMs, 1Gb', 'Hadoop\n15VMs, 2Gb']

    def get_allocs_and_time_inner(smt_flag, root_dir, file_prefix, az):
        allocs, times = [], []

        if smt_flag:
            # sample: config_results_vdcmapper_hadoop_us_west1_4vm_1bw
            data_files = [
                root_dir + file_prefix + az + '_4vm_1bw',
                root_dir + file_prefix + az + '_4vm_2bw',
                root_dir + file_prefix + az + '_10vm_1bw',
                root_dir + file_prefix + az + '_10vm_2bw',
                root_dir + file_prefix + az + '_15vm_1bw',
                root_dir + file_prefix + az + '_15vm_2bw']
        else:
            # sample: results.softaffinity.us-mid1.pn.hadoop_4_1bw.vn.log
            data_files = [
                root_dir + file_prefix + az + '.pn.hadoop_4_1bw.vn.log',
                root_dir + file_prefix + az + '.pn.hadoop_4_2bw.vn.log',
                root_dir + file_prefix + az + '.pn.hadoop_10_1bw.vn.log',
                root_dir + file_prefix + az + '.pn.hadoop_10_2bw.vn.log',
                root_dir + file_prefix + az + '.pn.hadoop_15_1bw.vn.log',
                root_dir + file_prefix + az + '.pn.hadoop_15_2bw.vn.log']

        for file in data_files:
            alloc, time = get_allocations_and_time(file)
            allocs.append(alloc)
            times.append(time)

        glog.info('allocs = {}, times = {}'.format(allocs, times))
        return allocs, times


    root_dir_prefix = './figures/extensions/'

    for topology in topologies:
        if topology == 'west1':
            smt_flag = True
            file_prefix = 'config_results_hadoop_us-'
            plot_file_name = 'paper-plots/fig12_' + topology + '_smt.pdf'
            root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/softaffinity/')
            affinity_allocs, affinity_time = get_allocs_and_time_inner(
                smt_flag, root_dir, file_prefix, topology)

            root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/minserver/')
            minserver_allocs, minserver_time = get_allocs_and_time_inner(
                smt_flag, root_dir, file_prefix, topology)
            
            root_dir = os.path.join(root_dir_prefix, 'netsolver-smt/timelogs/mincores/')
            hotspot_allocs, hotspot_time = get_allocs_and_time_inner(
                smt_flag, root_dir, file_prefix, topology)

            allocs_range = np.arange(0, 2501, 500)
            plot3bars_ext(plot_file_name,
                affinity_allocs, minserver_allocs, hotspot_allocs,
                affinity_time, minserver_time, hotspot_time,
                vdc_names, time_range, allocs_range)


            # make the identical plot for ILP
            smt_flag = False
            plot_file_name = 'paper-plots/fig12_' + topology + '_ilp.pdf'
            allocs_range = np.arange(0, 2501, 500)

            root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/softaffinity/')
            file_prefix = 'results.softaffinity.us-'
            affinity_allocs, affinity_time = get_allocs_and_time_inner(
                smt_flag, root_dir, file_prefix, topology)

            root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/servermin/')
            file_prefix = 'results.servermin.us-'
            minserver_allocs, minserver_time = get_allocs_and_time_inner(
                smt_flag, root_dir, file_prefix, topology)
            
            root_dir = os.path.join(root_dir_prefix, 'netsolver-ilp/timelogs/hotspot/')
            file_prefix = 'results.hotspot.us-'
            hotspot_allocs, hotspot_time = get_allocs_and_time_inner(
                smt_flag, root_dir, file_prefix, topology)

            plot3bars_ext(plot_file_name,
                affinity_allocs, minserver_allocs, hotspot_allocs,
                affinity_time, minserver_time, hotspot_time,
                vdc_names, time_range, allocs_range)
        
        elif topology == 'west2':
            allocs_range = np.arange(0, 1001, 200)
            affinity_allocs = [560, 560, 220, 216, 144, 142]
            minserver_allocs = [280, 280, 218, 215, 143, 139]
            hotspot_allocs = [556, 487, 81, 157, 56, 133]

            affinity_time = [0.001, 0.001, 0.1, 0.1, 1.5, 0.6]
            minserver_time = [0.001, 0.001, 0.1, 0.2, 4, 5]
            hotspot_time = [0.6, 0.6, 2.2, 0.8, 7, 9]

        else:
            glog.error('ERROR: unsupported DC topology {}. Exit.'.format(topology))
            sys.exit(1)


if __name__ == "__main__":
    """ Plot figures in AIJ paper. Run as $ python plot_aij.py --h """
    CLI = argparse.ArgumentParser(
        description='Choose the plot to generate')
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
        '-f9',
        '--figure9',
        action='store_true',
        default=False,
        help='Compare VNE algorithms in commercial datacenter topologies using Hadoop instances')
    
    group.add_argument(
        '-f10',
        '--figure10',
        action='store_true',
        default=False,
        help='Robustness experiments: SMT and ILP make additional allocations from where SecondNet left')

    group.add_argument(
        '-f12',
        '--figure12',
        action='store_true',
        default=False,
        help='NetSolver with extensions: affinity, min-server, no-hotspot')

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
    elif ARGS.figure9:
        fig9()
    elif ARGS.figure10:
        fig10()
    elif ARGS.figure12:
        topologies = ['west1']
        fig12(topologies)
    else:
        glog.error('ERROR: invalid option.')
        sys.exit(1)
