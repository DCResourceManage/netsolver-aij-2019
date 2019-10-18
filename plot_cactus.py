import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import sys
import os
import glog
from matplotlib.ticker import FormatStrFormatter

root_data_location = './figures/'


def get_allocations_and_time(ifile):
    # print(ifile)
    fd = open(ifile, 'r')
    lines = fd.readlines()
    start_read = False
    execution_time = []
    allocations = []
    total_allocs = 0
    for line in lines:
        if not start_read and 'init' in line:
            start_read = True
            continue
        if 'init' in line:
            continue
        if 'done' in line:
            start_read = False
            break

        if start_read:
            # stop after cumulative 3600 seconds (one hour)
            cumul_time = float(line.split(' ')[2])
            if cumul_time > 3600:
                break
            total_allocs += 1
            execution_time.append(float(line.split(' ')[2]))
            allocations.append(total_allocs)

    fd.close()
    return (allocations, execution_time)


def cactus_plot(plot_file_name, datacenterName, vdcName, datasets,
    allocs_range, time_range):
    print('started plotting')
    # fig, ax = plt.subplots(figsize=(3.25, 3.25))
    fig, ax = plt.subplots(figsize=(4, 2.25))

    n_groups = len(datasets)
    lines = []
    axes = plt.gca()
    n_of_markers = 4
    #plt.title(datacenterName + " " + vdcName)
    for index, (name, color, marker, msize, file) in enumerate(datasets):
        if os.path.exists(file):
            allocations, execution_times = get_allocations_and_time(file)
            if(len(allocations)==0):
                print("Warning: No allocations found for " + file)
            mspace = allocs_range[-1]/n_of_markers
            line = ax.plot(execution_times, allocations, color=color, 
                marker=marker, markevery=mspace, markersize=msize, label=name)
        else:
            print("Warning: Missing file: " + file)

    
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.8), numpoints=2, ncol=2,
        frameon=False)

    plt.xlabel('CPU Time (s)')
    plt.ylabel('Number of VDC allocations')
    plt.yticks(allocs_range)

    # set time axis a bit forward, to make sure that secondnet runs (that are very close to 0) are visible
    xshift = max(3, time_range[1]/10)
    ax.set_xlim(-xshift)
    plt.xticks(time_range)

    fig.tight_layout()
    plt.draw()
    final_figure = plt.gcf()
    final_figure.savefig(plot_file_name, bbox_inches='tight', dpi=200)
    print('plotting done, see %s' % plot_file_name)
    plt.close()


def fig6():
    # included in the paper.pdf
    # fig6_cactus_2000_16_T1_9VMs.pdf
    # fig6_cactus_2000_16_T3_15VMs.pdf

    servers_and_cores = [('200', '4'), #('200', '16'), ('400', '16'),
                        ('2000', '16')]

    allocs_ranges = [np.arange(0, 101, 25),
        np.arange(0, 101, 25),
        np.arange(0, 101, 25),
        np.arange(0, 61, 20),
        np.arange(0, 61, 20),
        np.arange(0, 61, 20),

        np.arange(0, 4001, 1000),
        np.arange(0, 3601, 900),
        np.arange(0, 3001, 750),
        np.arange(0, 1201, 300),
        np.arange(0, 1201, 300),
        np.arange(0, 1201, 300)]

    time_ranges = [np.arange(0, 61, 15),
        np.arange(0, 81, 20),
        np.arange(0, 81, 20),
        np.arange(0, 141, 35),
        np.arange(0, 141, 35),
        np.arange(0, 141, 35),

        np.arange(0, 3601, 900),
        np.arange(0, 3601, 900),
        np.arange(0, 3601, 900),
        np.arange(0, 3601, 900),
        np.arange(0, 3601, 900),
        np.arange(0, 3601, 900)]

    range_index = 0
    for (server, core) in servers_and_cores:
        # files for 4 core data do not contain numbers of cores, only 16 core does
        if core == '4':
            core_str = '_';
        else:
            core_str = core + '_'
        # secondnet sample file name: fmcad13_secondnet_datacentertree_200.20_instance_vn2_3.1.log
        root_dir = root_data_location + 'fmcad-instances/secondnet/timelogs/'
        prefix = 'fmcad13_secondnet_datacentertree'
        suffix = '.20_instance_vn2_'
        secondnet_files = [
            root_dir + prefix + core_str + server + suffix + '3.1.log',
            root_dir + prefix + core_str + server + suffix + '3.2.log',
            root_dir + prefix + core_str + server + suffix + '3.3.log',
            root_dir + prefix + core_str + server + suffix + '5.1.log',
            root_dir + prefix + core_str + server + suffix + '5.2.log',
            root_dir + prefix + core_str + server + suffix + '5.3.log']

        # netsolver sample file name: fmcad13_vdcmapper_intrinsic-edge-sets-no-rnd-theory-order-theory-order-vsids-rnd-theory-freq-0-99_datacentertree_200.20.pn_instance_vn2_3.1.vn
        root_dir = root_data_location + 'fmcad-instances/netsolver-smt/monosat-master/timelogs/'
        prefix = 'fmcad13_vdcmapper_intrinsic-edge-sets-no-rnd-theory-order-theory-order-vsids-rnd-theory-freq-0-99_datacentertree'
        suffix = '.20.pn_instance_vn2_'
        netsolver_files = [
            root_dir + prefix + core_str + server + suffix + '3.1.vn.log',
            root_dir + prefix + core_str + server + suffix + '3.2.vn.log',
            root_dir + prefix + core_str + server + suffix + '3.3.vn.log',
            root_dir + prefix + core_str + server + suffix + '5.1.vn.log',
            root_dir + prefix + core_str + server + suffix + '5.2.vn.log',
            root_dir + prefix + core_str + server + suffix + '5.3.vn.log']

        root_dir = root_data_location + 'fmcad-instances/netsolver-ilp/'
        prefix = 'results.tree'
        suffix = '.20.vn'
        gurobi_files = [
            root_dir + prefix + core_str + server + suffix + '3.1',
            root_dir + prefix + core_str + server + suffix + '3.2',
            root_dir + prefix + core_str + server + suffix + '3.3',
            root_dir + prefix + core_str + server + suffix + '5.1',
            root_dir + prefix + core_str + server + suffix + '5.2',
            root_dir + prefix + core_str + server + suffix + '5.3']


        # z3 sample file name: fmcad13_z3_mode4_datacentertree_200.20_instance_vn2_3.1.log
        root_dir = root_data_location + 'fmcad-instances/Z3-AR/timelogs/'
        prefix = 'fmcad13_z3_mode3_datacentertree'
        suffix = '.20_instance_vn2_'
        z3_files = [
            root_dir + prefix + core_str + server + suffix + '3.1.log',
            root_dir + prefix + core_str + server + suffix + '3.2.log',
            root_dir + prefix + core_str + server + suffix + '3.3.log',
            root_dir + prefix + core_str + server + suffix + '5.1.log',
            root_dir + prefix + core_str + server + suffix + '5.2.log',
            root_dir + prefix + core_str + server + suffix + '5.3.log']

        # vdc_names = ['vn3.1', 'vn3.2', 'vn3.3', 'vn5.1', 'vn5.2', 'vn5.3']
        vdc_names = ['T1 9VMs', 'T2 9VMs', 'T3 9VMs',
                    'T1 15VMs', 'T2 15VMs', 'T3 15VMs']

        for i, vdc_name in enumerate(vdc_names):
            plot_file_name = "./paper-plots/cactus/fig6_cactus_" + server + "_" + str(core) + "_" + vdc_name.replace(
                " ", "_") +".pdf"
            cactus_plot(plot_file_name, server + core, vdc_name, [
               ("Z3-AR", 'y', '.', 8, z3_files[i]),
               ("SecondNet", 'r', '^', 8, secondnet_files[i]),
               ("NetSolver-SMT", 'g', 's', 8, netsolver_files[i]),
               ("NetSolver-ILP", 'b', '*', 8, gurobi_files[i])],
               allocs_ranges[range_index], time_ranges[range_index])

            range_index += 1


def fig7_bcube():
    # included in the paper.pdf
    # fig7_cactus_bcube512_12VMs.pdf

    dc_sizes = ['8', '10', '12', '16']
    vn_sizes = ['6','9','12','15']

    allocs_range = np.arange(0, 721, 180)
    time_range = np.arange(0, 3601, 900)

    for size in dc_sizes:
        if size == '8':
            servers = '512'
        elif size == '10':
            servers = '1000'
        elif size == '12':
            servers = '1728'
        elif size == '16':
            servers = '4096'

        root_dir = root_data_location + 'fattree-bcube/secondnet/timelogs/'
        file_prefix = 'results.BCube_n'
        file_infix = "_k2_bw10_cpu16.pn.instance_list_"
        file_suffix = "vm_rnd.txt.log";

        secondnet_files = [root_dir + file_prefix + size + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        root_dir = root_data_location + 'fattree-bcube/netsolver-smt/monosat-master/timelogs/'
        file_prefix = 'config_results_vdcmapper_bcube'
        file_infix = "_"
        file_suffix = "vm";

        netsolver_files = [root_dir + file_prefix + size + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        root_dir = root_data_location + 'fattree-bcube/netsolver-ilp/timelogs/'
        file_prefix = 'results.BCube_n'
        file_infix = "_k2_bw10_cpu16.pn.instance_list_"
        file_suffix = "vm_rnd.txt.log";

        gurobi_files = [root_dir + file_prefix + size + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        vdc_names = ['6VMs', '9VMs', '12VMs', '15VMs']

        for i, vdc_name in enumerate(vdc_names):
            plot_file_name = "paper-plots/cactus/fig7_cactus_" + "bcube" + servers + "_" + vdc_name +".pdf"
            cactus_plot(plot_file_name, "bcube" + servers + " ", vdc_name, [
               ("SecondNet", 'r', '^', 8, secondnet_files[i]),
               ("NetSolver-SMT", 'g', 's', 8, netsolver_files[i]),
               ("NetSolver-ILP", 'b', '*', 8, gurobi_files[i])],
               allocs_range, time_range)


def fig7_fattree():
    # included in the paper.pdf
    # fig7_cactus_fattree432_12VMs.pdf

    allocs_range = np.arange(0, 721, 180)
    time_range = np.arange(0, 3601, 900)

    dc_sizes = ['8', '12', '16']
    vn_sizes = ['6', '9', '12', '15']
    for size in dc_sizes:
        if size == '8':
            servers = '128'
        elif size == '12':
            servers = '432'
        elif size == '16':
            servers = '1024'


        root_dir = root_data_location + 'fattree-bcube/secondnet/timelogs/'
        file_prefix = 'results.Fattree_n'
        file_infix = "_k2_bw10_cpu16.pn.instance_list_"
        file_suffix = "vm_rnd.txt.log";

        secondnet_files = [root_dir + file_prefix + size + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        root_dir = root_data_location + 'fattree-bcube/netsolver-smt/monosat-master/timelogs/'
        file_prefix = 'config_results_vdcmapper_fattree'
        file_infix = "_"
        file_suffix = "vm";

        netsolver_files = [root_dir + file_prefix + size + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        root_dir = root_data_location + 'fattree-bcube/netsolver-ilp/timelogs/'
        file_prefix = 'results.Fattree_n'
        file_infix = "_k2_bw10_cpu16.pn.instance_list_"
        file_suffix = "vm_rnd.txt.log";

        gurobi_files = [root_dir + file_prefix + size + file_infix + vn_size +file_suffix for vn_size in vn_sizes]


        vdc_names = ['6VMs', '9VMs', '12VMs', '15VMs']

        for i, vdc_name in enumerate(vdc_names):
            plot_file_name = "paper-plots/cactus/fig7_cactus_" + "fattree" + servers + "_" + vdc_name +".pdf"
            cactus_plot(plot_file_name, "fattree" + servers + " " , vdc_name, [
               ("SecondNet", 'r', '^', 8, secondnet_files[i]),
               ("NetSolver-SMT", 'g', 's', 8, netsolver_files[i]),
               ("NetSolver-ILP", 'b', '*', 8, gurobi_files[i])],
               allocs_range, time_range)


def fig8():
    # included in the paper.pdf
    # fig8_cactus_mid1_H-10VM.pdf
    # fig8_cactus_west1_H-10VM.pdf

    allocs_ranges = [np.arange(0, 101, 25),
        np.arange(0, 301, 75)]

    time_ranges = [np.arange(0, 110, 35),
        np.arange(0, 1041, 260)]

    azs = ['west1', 'west2', 'mid1', 'mid2']
    vn_sizes = ['4','6','8','10', '12']
    range_index = 0

    for az in azs:
        root_dir = root_data_location + 'hadoop/secondnet/timelogs/'
        file_prefix = 'results.us_'
        file_infix = ".pn.instance_list_hadoop_"
        file_suffix = "vm_rnd.txt.log";

        secondnet_files = [root_dir + file_prefix + az + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        root_dir = root_data_location + 'hadoop/netsolver-smt/monosat-master/timelogs/'
        file_prefix = 'config_results_vdcmapper_hadoop_us_'
        file_infix = "_"
        file_suffix = "vm";

        netsolver_files = [root_dir + file_prefix + az + file_infix + vn_size +file_suffix for vn_size in vn_sizes]

        root_dir = root_data_location + 'hadoop/netsolver-ilp/timelogs/'
        file_prefix = 'results.us-'
        file_infix = ".pn.instance_list_hadoop_"
        file_suffix = "vm_rnd.txt.log";

        gurobi_files = [root_dir + file_prefix + az + file_infix + vn_size +file_suffix for vn_size in vn_sizes]


        vdc_names = ['H-4VM', 'H-6VM', 'H-8VM', 'H-10VM', 'H-12VM']
        if az == 'mid1':
            range_index = 0
        elif az == 'west1':
            range_index = 1

        for i, vdc_name in enumerate(vdc_names):
            plot_file_name = "paper-plots/cactus/fig8_cactus_" + az + "_" + vdc_name +".pdf"

            cactus_plot(plot_file_name, az, vdc_name, [
               ("SecondNet", 'r', '^', 8, secondnet_files[i]),
               ("NetSolver-SMT", 'g', 's', 8, netsolver_files[i]),
               ("NetSolver-ILP", 'b', '*', 8, gurobi_files[i])],
               allocs_ranges[range_index], time_ranges[range_index])


if __name__ == "__main__":
    """ Make cactus plots to compare different algorithms. Run as $ python plot_cactus.py --h """
    CLI = argparse.ArgumentParser(
        description='Choose the figure to output the cactus plot for')
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

    ARGS = CLI.parse_args()

    if ARGS.figure6:
        fig6()
    elif ARGS.figure7:
        fig7_fattree()
        fig7_bcube()

    elif ARGS.figure8:
        fig8()
    
    else:
        glog.error('ERROR: invalid option.')
        sys.exit(1)
