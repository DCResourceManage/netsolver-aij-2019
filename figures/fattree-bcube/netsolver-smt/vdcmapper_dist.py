import argparse
from collections import OrderedDict
from fractions import gcd
from functools import reduce
import functools
from itertools import tee
import json
import math
from networkx.algorithms.approximation.vertex_cover import \
    min_weighted_vertex_cover
import networkx.algorithms.components
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.flow import maxflow
import networkx.algorithms.flow
from networkx.algorithms.shortest_paths import generic
from networkx.relabel import convert_node_labels_to_integers
import os
from random import shuffle
import random
import shutil
import sys
import time

from monosat import *
import networkx as nx
import time



#if "PYTHONHASHSEED" not in os.environ or os.environ['PYTHONHASHSEED'] != "1":
#    print(
#        "Python 3.4+ randomizes dictionary hashes by default (using PYTHONHASHSEED as its random seed).\n Set PYTHONHASHSEED=1 in the runtime environment (e.g., '$export PYTHONHASHSEED=1') in order to reproduce experiments (currently, PYTHONHASHSEED is %s)." % (
#        os.environ['PYTHONHASHSEED'] if 'PYTHONHASHSEED' in os.environ else "UNSET"))
#    sys.exit(1)
if __name__ == "__main__":
    from os import path
    import os
    import sys

    # Setup PYTHONPATH... if anyone knows a better way to do this without a shell script, I'm all ears..
    sys.path.append(os.path.abspath(os.path.join(path.dirname(__file__), os.pardir)))

random.seed(1)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def printGraph(G, key="weight"):
    print("graph{")
    for (s1, s2, data) in G.edges_iter(data=true):
        print(s1 + " -- " + s2 + ("[xlabel=\"%d Mbps\"]" % (data[key]) if key is not None else ""))
    print("}")


def printDiGraph(G, key="weight"):
    print("digraph{")
    for (s1, s2, data) in G.edges_iter(data=true):
        print(s1 + " -> " + s2 + "[xlabel=\"%d Mbps\"]" % (data[key]))
    print("}")


start_time = time.clock()

parser = argparse.ArgumentParser(description='VDCMapper')

parser.add_argument("--logfile", type=str, help="Output filename", default=None)

parser.add_argument("--timelog", type=str, help="Output filename", default=None)


parser.add_argument("--output",type=str,help="Output filename, to store VN mapping in JSON format",default=None)

parser.add_argument('--static-flow_caps', dest='static_flow_caps', help="", action='store_true')
parser.add_argument('--no-static-flow_caps', dest='static_flow_caps', help="", action='store_false')
parser.set_defaults(static_flow_caps=True)

parser.add_argument('--tree-addition', dest='tree_addition',
                    help="Build multi-argument addition constraints in a tree, instead of linear chain",
                    action='store_true')
parser.add_argument('--no-tree-addition', dest='tree_addition', help="", action='store_false')
parser.set_defaults(tree_addition=True)

parser.add_argument('--bitblast-addition', dest='bitblast_addition', help="", action='store_true')
parser.add_argument('--no-bitblast-addition', dest='bitblast_addition', help="", action='store_false')
parser.set_defaults(bitblast_addition=False)

parser.add_argument('--bitblast_addition_shadow', dest='bitblast_addition_shadow', help="", action='store_true')
parser.add_argument('--no-bitblast_addition_shadow', dest='bitblast_addition_shadow', help="", action='store_false')
parser.set_defaults(bitblast_addition_shadow=False)

parser.add_argument('--separate_reach_constraints', dest='separate_reach_constraints', help="", action='store_true')
parser.add_argument('--no-separate_reach_constraints', dest='separate_reach_constraints', help="", action='store_false')
parser.set_defaults(separate_reach_constraints=False)

parser.add_argument('--assert_all_physical_edges', dest='assert_all_physical_edges', help="", action='store_true')
parser.add_argument('--no-assert_all_physical_edges', dest='assert_all_physical_edges', help="", action='store_false')
parser.set_defaults(assert_all_physical_edges=True)

parser.add_argument('--ignore-ram', dest='ignore_ram_constraints', help="", action='store_true')
parser.add_argument('--no-ignore-ram', dest='ignore_ram_constraints', help="", action='store_false')
parser.set_defaults(ignore_ram_constraints=False)

parser.add_argument('--ignore-storage', dest='ignore_storage_constraints', help="", action='store_true')
parser.add_argument('--no-ignore-storage', dest='ignore_storage_constraints', help="", action='store_false')
parser.set_defaults(ignore_storage_constraints=True)

parser.add_argument('--prop-assumps', dest='prop_assumps', help="", action='store_true')
parser.add_argument('--no-prop-assumps', dest='prop_assumps', help="", action='store_false')
parser.set_defaults(prop_assumps=True)

parser.add_argument('--force_virtual_switches', dest='force_virtual_switches', help="", action='store_true')
parser.add_argument('--no-force_virtual_switches', dest='force_virtual_switches', help="", action='store_false')
parser.set_defaults(force_virtual_switches=False)

parser.add_argument('--allow_virtual_switches_in_switches', dest='allow_virtual_switches_in_switches', help="",
                    action='store_true')
parser.add_argument('--no-allow_virtual_switches_in_switches', dest='allow_virtual_switches_in_switches', help="",
                    action='store_false')
parser.set_defaults(allow_virtual_switches_in_switches=True)

parser.add_argument('--intrinsic_edge_constraints', dest='intrinsic_edge_constraints', help="", action='store_true')
parser.add_argument('--no-intrinsic_edge_constraints', dest='intrinsic_edge_constraints', help="", action='store_false')
parser.set_defaults(intrinsic_edge_constraints=False)

parser.add_argument('--intrinsic_edge_sets', dest='intrinsic_edge_sets', help="", action='store_true')
parser.add_argument('--no-intrinsic_edge_sets', dest='intrinsic_edge_sets', help="", action='store_false')
parser.set_defaults(intrinsic_edge_sets=True)

parser.add_argument('--flush-pb', dest='flush_pb_theory', help="", action='store_true')
parser.add_argument('--no-flush-pb', dest='flush_pb_theory', help="", action='store_false')
parser.set_defaults(flush_pb_theory=False)

parser.add_argument('--refine_flow', dest='refine_flow', help="", action='store_true')
parser.add_argument('--no-refine_flow', dest='refine_flow', help="", action='store_false')
parser.set_defaults(refine_flow=True)

parser.add_argument('--remove_flow_cycles', dest='remove_flow_cycles', help="", action='store_true')
parser.add_argument('--no-remove_flow_cycles', dest='remove_flow_cycles', help="", action='store_false')
parser.set_defaults(remove_flow_cycles=False)

parser.add_argument('--remove_flow_cycles_python', dest='remove_flow_cycles_python', help="", action='store_true')
parser.add_argument('--no-remove_flow_cycles_python', dest='remove_flow_cycles_python', help="", action='store_false')
parser.set_defaults(remove_flow_cycles_python=False)

parser.add_argument('--remove_all_flow_cycles', dest='remove_all_flow_cycles', help="", action='store_true')
parser.add_argument('--no-remove_all_flow_cycles', dest='remove_all_flow_cycles', help="", action='store_false')
parser.set_defaults(remove_all_flow_cycles=False)

parser.add_argument('--use_min_cost_max_flow', dest='use_min_cost_max_flow', help="", action='store_true')
parser.add_argument('--no-use_min_cost_max_flow', dest='use_min_cost_max_flow', help="", action='store_false')
parser.set_defaults(use_min_cost_max_flow=False)

parser.add_argument('--use_cover_for_refinement', dest='use_cover_for_refinement', help="", action='store_true')
parser.add_argument('--no-use_cover_for_refinement', dest='use_cover_for_refinement', help="", action='store_false')
parser.set_defaults(use_cover_for_refinement=True)

parser.add_argument('--directed', dest='directed_links',
                    help='Assume data-center capacities are available separately in both directions',
                    action='store_true')
parser.add_argument('--no-directed', dest='directed_links',
                    help='Assume data-center capacities are shared beween both directions', action='store_false')
parser.set_defaults(directed_links=False)

parser.add_argument('--debug', dest='debug', help="", action='store_true')
parser.add_argument('--no-debug', dest='debug', help="", action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--rnd-theory-freq', dest='rnd_theory_freq', type=float, help="", default=0.99)

parser.add_argument('--rnd-theory-order', dest='rnd_theory_order', help="", action='store_true')
parser.add_argument('--no-rnd-theory-order', dest='rnd_theory_order', help="", action='store_false')
parser.set_defaults(rnd_theory_order=False)

parser.add_argument('--theory-order-vsids', dest='theory_order_vsids', help="", action='store_true')
parser.add_argument('--no-theory-order-vsids', dest='theory_order_vsids', help="", action='store_false')
parser.set_defaults(theory_order_vsids=True)

parser.add_argument('--decide-opt-lits', dest='decide_opt_lits', help="", action='store_true')
parser.add_argument('--no-decide-opt-lits', dest='decide_opt_lits', help="", action='store_false')
parser.set_defaults(decide_opt_lits=True)

parser.add_argument('--binary-search', dest='binary_search', help="", action='store_true')
parser.add_argument('--no-binary-search', dest='binary_search', help="", action='store_false')
parser.set_defaults(binary_search=True)

parser.add_argument('--vsids-both', dest='vsids_both', help="", action='store_true')
parser.add_argument('--no-vsids-both', dest='vsids_both', help="", action='store_false')
parser.set_defaults(vsids_both=True)

parser.add_argument('--rnd-restart', dest='rnd_restart', help="", action='store_true')
parser.add_argument('--no-rnd-restart', dest='rnd_restart', help="", action='store_false')
parser.set_defaults(rnd_restart=True)

parser.add_argument('--vsids-balance', dest='vsids_balance', type=float, help="", default=1)

parser.add_argument('--popcount', dest='popcount', type=str, help="", default="BV")

parser.add_argument('--optimization-conflict-limit', dest='opt_conflict_limit', type=int, help="", default=0)
parser.add_argument('--conflict-limit', dest='conflict_limit', type=int, help="", default=-1)

parser.add_argument('--optimization-time-limit', dest='opt_time_limit', type=int, help="", default=-1)

parser.add_argument('--detect-sat', dest='opt_detect_sat', help="", action='store_true')
parser.add_argument('--no-detect-sat', dest='opt_detect_sat', help="", action='store_false')
parser.set_defaults(opt_detect_sat=False)

parser.add_argument('--affinity', dest='affinity', help="", action='store_true')
parser.add_argument('--no-affinity', dest='affinity', help="", action='store_false')
parser.set_defaults(affinity=True)

parser.add_argument('--soft-affinity', dest='affinity_soft', help="", action='store_true')
parser.add_argument('--no-soft-affinity', dest='affinity_soft', help="", action='store_false')
parser.set_defaults(affinity_soft=True)

parser.add_argument('--anti-affinity', dest='anti_affinity', help="", action='store_true')
parser.add_argument('--no-anti-affinity', dest='anti_affinity', help="", action='store_false')
parser.set_defaults(anti_affinity=True)

parser.add_argument('--min-servers', dest='min_servers', help="", action='store_true')
parser.add_argument('--no-min-servers', dest='min_servers', help="", action='store_false')
parser.set_defaults(min_servers=False)

parser.add_argument('--min-cores', dest='min_cores', help="", action='store_true')
parser.add_argument('--no-min-cores', dest='min_cores', help="", action='store_false')
parser.set_defaults(min_cores=False)

parser.add_argument('--max-cores', dest='max_cores', type=int, help="", default=1)
parser.add_argument('--max-ram', dest='max_ram', type=int, help="", default=1)
parser.add_argument('--max-storage', dest='max_storage', type=int, help="", default=1)

parser.add_argument("--deallocate-prob", dest='deallocate_prob', type=float,
                    help="Probability, after each allocation, that a single, randomly chosen previously allocated VDC will be deallocated",
                    default=0)

parser.add_argument('--max-vms', dest='max_vms', type=int, help="", default=10)

parser.add_argument('--configured', dest='configured', help="", action='store_true')
parser.set_defaults(configured=False)

parser.add_argument('--ijcai-settings', dest='ijcai_settings', help="", action='store_true')
parser.set_defaults(ijcai_settings=False)


parser.add_argument('PN', metavar='PN', type=str,
                    help='Physical data center to allocate the virtual data center to')

parser.add_argument('VDC_LIST', metavar='VDC', type=str,
                    help='List of virtual data centers to allocate')

args = parser.parse_args()

if args.configured:
    #override command line settings to match configuration experiments
    args.assert_all_physical_edges = False
    args.bitblast_addition=False
    args.intrinsic_edge_constraints=False
    args.intrinsic_edge_sets=True
    args.refine_flow=True
    args.remove_flow_cycles=False
    args.static_flow_caps=True
    args.tree_addition=True
    args.use_cover_for_refinement=True



print("Settings: " + str(args))

log_file = args.logfile

log_time_file = args.timelog

static_flow_caps = args.static_flow_caps
tree_addition = args.tree_addition
bitblast_addition = args.bitblast_addition
bitblast_addition_shadow = args.bitblast_addition_shadow
separate_reach_constraints = args.separate_reach_constraints
assert_all_physical_edges = args.assert_all_physical_edges
# remove_redundant_ram_cpu_constraints=True
ignore_ram_constraints = args.ignore_ram_constraints
ignore_storage_constraints = args.ignore_storage_constraints

force_virtual_switches = args.force_virtual_switches
allow_virtual_switches_in_switches = args.allow_virtual_switches_in_switches
intrinsic_edge_constraints = args.intrinsic_edge_constraints
intrinsic_edge_sets = args.intrinsic_edge_sets
flush_pb_theory = args.flush_pb_theory
remove_flow_cycles = args.remove_flow_cycles
remove_all_flow_cycles = args.remove_all_flow_cycles
refine_flow = args.refine_flow
use_min_cost_max_flow = args.use_min_cost_max_flow
use_cover_for_refinement = args.use_cover_for_refinement
remove_flow_cycles_python = args.remove_flow_cycles_python
directed_links = args.directed_links
anti_affinity = args.anti_affinity
min_servers = args.min_servers
decide_opt_lits = args.decide_opt_lits
binary_search = args.binary_search
min_cores = args.min_cores
vsids_both = args.vsids_both
vsids_balance = args.vsids_balance
popcount = args.popcount


output_file = args.output

if output_file:
    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass

# all_instances = json.load(open(args.VDC_LIST))

vdc_allocation_map = dict()

allocated_vdcs = set()

only_instantiate_networks_on_demand = False
max_cores = args.max_cores
max_ram = args.max_ram
max_storage = args.max_storage
max_virt_nodes = args.max_vms  # maximum virtual nodes supported by this encoding
max_vdc_bandwidth = 32
vms = ["vm_%d" % (n) for n in range(max_virt_nodes)]
extra_vms = []  # ["vm_%d"%(n) for n in range(max_virt_nodes,25)]
if not force_virtual_switches:
    refine_flow = False
print("Modelling up to %d vms, with up to %d cores, %d ram, %d storage per vm" % (
len(vms), max_cores, max_ram, max_storage))
rnd_theory_order = args.rnd_theory_order
theory_order_vsids = args.theory_order_vsids
debug = args.debug
rnd_theory_freq = args.rnd_theory_freq
print("Setting random decision freq to %f" % (rnd_theory_freq))

if args.configured:
    # override command line settings to match configuration experiments
    monosat_settings = "-adaptive-history-clear=25 -conflict-min-cut-maxflow  -no-conflict-min-cut -decide-bv-bitwise -decide-graph-bv -no-decide-graph-rnd -decide-opt-lits -decide-theories -detect-pure-lits -no-reach-underapprox-cnf -rnd-theory-freq=0.99 -theory-order-vsids -no-theory-prop-during-assumps -theory-prop-during-fast-simp -no-theory-prop-during-simp -vsids-balance=0.99 -vsids-both"
elif args.ijcai_settings:
    monosat_settings = "-adaptive-history-clear=10 -conflict-min-cut -conflict-min-cut-maxflow -no-decide-bv-bitwise -decide-graph-bv -no-decide-graph-rnd -decide-opt-lits -decide-theories -detect-pure-lits -reach-underapprox-cnf -rnd-theory-freq=0.99 -theory-order-vsids -theory-prop-during-assumps -theory-prop-during-fast-simp -no-theory-prop-during-simp -vsids-balance=0.99 -no-vsids-both "
else:
    monosat_settings = " %s -detect-pure-lits -adaptive-history-clear=10 -no-theory-prop-during-simp -no-theory-prop-during-fast-simp  -opt-time-limit=%d  -verb=0 %s %s %s %s %s -rnd-theory-freq=%f -vsids-balance=%f -no-decide-bv-intrinsic  -decide-bv-bitwise  -decide-graph-bv -decide-theories -no-decide-graph-rnd   -lazy-maxflow-decisions -no-conflict-min-cut -conflict-min-cut-maxflow -reach-underapprox-cnf -check-solution " % (
    " -detect-sat-predicates=1" if args.opt_detect_sat else "", args.opt_time_limit, " " if rnd_theory_order else " ",
    "-theory-order-vsids" if theory_order_vsids else "-no-theory-order-vsids",
    "-decide-opt-lits" if decide_opt_lits else "-no-decide-opt-lits", " " if binary_search else " ",
    "-vsids-both" if vsids_both else "-no-vsids-both", rnd_theory_freq, vsids_balance) + (
                       " -rnd-restart " if args.rnd_restart else " -no-rnd-restart ") + (
                       " -theory-prop-during-assumps " if args.prop_assumps else " -no-theory-prop-during-assumps ")
print("Monosat Settings: " + monosat_settings)
Monosat().init(monosat_settings)

# Monosat().init("-verb=0 -rnd-theory-freq=%f -verb-time=0 -no-decide-bv-intrinsic  -decide-bv-bitwise  -decide-graph-bv -decide-theories -no-decide-graph-rnd   -lazy-maxflow-decisions -no-conflict-min-cut -no-conflict-min-cut-maxflow -reach-underapprox-cnf -no-check-solution "%(rnd_theory_freq))

# from graphcircuit import Graph,GraphManager
# Remember to convert from AIG to CNF using the command: aigtocnf -m --no-pg
print("begin encode");

pn_file = args.PN
vn_instance_list = args.VDC_LIST
next_instance_to_allocate = 0
vdc_instances = []

for line in open(vn_instance_list):
    #test if this is a file or a list of files
    line = line.strip()
    if len(line)==0:
        continue
    if not os.path.isfile(line):
        print("Interpreting " + str(vn_instance_list) + " as a single VDC instance")
        vdc_instances = [vn_instance_list]
        break;
    vdc_instances.append(line)

instance = dict()


def selectVDC():
    global next_instance_to_allocate
    # instance_file ="../" +  random.choice(vdc_files)
    if (next_instance_to_allocate>=len(vdc_instances)):
        next_instance_to_allocate =0;
    instance_file = vdc_instances[next_instance_to_allocate]
    print("VDC is " + instance_file)
    next_instance_to_allocate += 1
    vdc_name = os.path.splitext(os.path.basename(instance_file))[0]
    return json.load(open(instance_file)),vdc_name
    #
    #
    # return open(instance_file)
    # if(next_instance_to_allocate>=len(all_instances)):
    #     print("Allocated all vdcs");
    #     return None
    # inst = all_instances[next_instance_to_allocate]
    # next_instance_to_allocate+=1
    # return inst


# vn= json.load(open(vn_file))
print("Loading PN...")
pn = json.load(open(pn_file))
instance["PN"] = sorted(pn["PN"])  # for reproducibility
instance["Servers"] = pn["Servers"]
# instance["VN"]=sorted(vn["VN"]) #for reproducibility
# instance["VMs"]=vn["VMs"]

# instance_name=os.path.basename(vn_file) + "_to_"  + os.path.basename(pn_file)
instance_name = "assign_to_" + os.path.basename(pn_file)

if not directed_links:
    seen_link = dict()
    to_remove = []
    keep_inst = []
    for (s1, s2, bandwidth) in instance["PN"]:
        if (s2, s1) in seen_link.keys():
            i = seen_link[(s2, s1)]
            (s1, s2, bandwidth_o) = keep_inst[i]
            keep_inst[i] = (s1, s2, bandwidth_o + bandwidth)
        else:
            seen_link[(s1, s2)] = len(keep_inst)
            keep_inst.append((s1, s2, bandwidth))

    instance["PN"] = keep_inst

n_removed_cycles = 0
use_cover = True
if directed_links:
    use_cover_for_refinement = False
    use_cover = False

    # double up the VDC (since unlike the datacenter, it should be undirected), to match our secondnet experiments

    # for vm1,vm2,bw in list(instance["VN"]):
    #    instance["VN"].append([vm2,vm1,bw])


def count(vector):
    for i, v in enumerate(vector):
        if v.value():
            return i + 1
    return 0


def _getFlowPaths(edge_flows, source, sink, maximum_bandwidth, capacity='capacity'):
    if source == sink:
        yield ([source, sink], maximum_bandwidth)
        return
    flow_graph = nx.DiGraph()
    for s, flows in edge_flows.items():
        for t, flow in flows.items():
            if flow > 0:
                if not flow_graph.has_edge(s, t):
                    flow_graph.add_edge(s, t, weight=flow)

    # yield from
    for item in decomposeIntoSimplePaths(flow_graph, source, sink, maximum_bandwidth, capacity): yield item


def remove_cycle(g, start, end, stack, weight='weight'):
    global n_removed_cycles
    n_removed_cycles += 1
    max_weight = g[start][end][weight]
    prev = end
    # assert(start in stack)
    # assert(end in stack)
    # assert(stack[-1]==start)
    for p, _ in reversed(stack):
        w = g[p][prev][weight]
        prev = p
        max_weight = min(max_weight, w)
        if p == end:
            break
    prev = end
    earliest = end
    assert (max_weight > 0)
    for p, _ in reversed(stack):
        assert (g[p][prev][weight] >= max_weight)
        g[p][prev][weight] -= max_weight
        if g[p][prev][weight] == 0:
            # g.remove_edge(p,prev)
            earliest = p
        prev = p
        if p == end:
            break
    return earliest


def remove_cycles(g, n, done_set, weight='weight'):
    stack = []
    seen = set()
    seen.add(n)
    stack.append((n, g.edges_iter([n], data=True)))

    while (len(stack) > 0):
        n, edges = stack[-1]
        try:
            (n, c, data) = next(edges)
            if data[weight] > 0:
                if c in seen:
                    # cycle
                    # it is safe to not consider c again here, because either the edge (n,c) is removed, or an earlier edge is removed, in which case we will backtrack.
                    backtrack_to = remove_cycle(g, n, c, stack, weight)
                    if backtrack_to != stack[-1][0]:
                        while (stack[-1][0] != backtrack_to):
                            seen.remove(stack[-1][0])
                            stack.pop()
                        continue
                else:
                    seen.add(c)
                    stack.append((c, g.edges_iter([c], data=True)))

        except StopIteration:
            stack.pop()
            seen.remove(n)
            done_set.add(n)


def remove_all_cycles(g, weight='weight'):
    print("Removing cycles...")
    original_removed = n_removed_cycles
    done_set = set()

    for n in g.nodes():
        if n not in done_set:
            remove_cycles(g, n, done_set, weight)

    if n_removed_cycles > original_removed:
        # remove any 0-weight edges
        remove = []
        for (u, v, data) in g.edges(data=True):
            if data[weight] == 0:
                remove.append((u, v))
        g.remove_edges_from(remove)
    # printDiGraph(g)
    print("Done removing cycles...")
    if debug:
        assert (is_directed_acyclic_graph(g))
        print("Removed %d (of %d) cycles" % (n_removed_cycles - original_removed, n_removed_cycles))


# Decompose the edge flow into (acyclic) flows
# iti = 0
def getFlowPaths(edge_flows, source, sink, maximum_bandwidth, capacity='capacity'):
    # global iti
    if source == sink:
        yield ([source, sink], maximum_bandwidth)
        return
    flow_graph = nx.DiGraph()
    for s, flows in edge_flows.items():
        for t, flow in flows.items():
            if flow > 0:
                if not flow_graph.has_edge(s, t):
                    flow_graph.add_edge(s, t, {capacity: flow})
    if source not in flow_graph.nodes() or sink not in flow_graph.nodes():
        return

    # while there is a cycle, remove it
    # There is probably a better way to do this!
    # Do a simple dfs cycle detection and removal
    # printDiGraph(flow_graph)
    # iti +=1
    # print(iti)
    # if iti==48:
    #    pass
    if remove_all_flow_cycles:
        if debug:
            expected_flow, _ = networkx.algorithms.flow.maxflow.maximum_flow(flow_graph, source, sink, capacity)
        remove_all_cycles(flow_graph, capacity)
        # printDiGraph(flow_graph)
        if debug:
            remaining_flow, _ = networkx.algorithms.flow.maxflow.maximum_flow(flow_graph, source, sink, capacity)
            assert (expected_flow == remaining_flow)

    # yield from
    for item in decomposeIntoSimplePaths(flow_graph, source, sink, maximum_bandwidth, capacity): yield item


# decompose the graph into simple paths from source to sink, ignoring any cycles.
# note: the 'capacity' field will be altered by this process!
def decomposeIntoSimplePaths(graph, source, sink, maximum_bandwidth, capacity='capacity'):
    if source == sink:
        yield ([source, sink], maximum_bandwidth)
        return
    if debug:
        original_graph = graph.copy()
    done = False
    total_flow = 0
    while (not done and total_flow < maximum_bandwidth):
        try:
            residual_path = nx.shortest_path(graph, source, sink)
            pathflow = min((graph[u][v][capacity] for (u, v) in pairwise(residual_path)))
            pathflow = min(maximum_bandwidth - total_flow, pathflow)
            total_flow += pathflow
            assert (pathflow > 0)
        except networkx.exception.NetworkXNoPath:
            # no s1 -- s2 paths exist in the residual network
            if total_flow != maximum_bandwidth:
                if debug:
                    printDiGraph(original_graph, capacity)
                print("Total flow along flow paths from %s to %s  was %d, but expected %d" % (
                source, sink, total_flow, maximum_bandwidth), file=sys.stderr)
                sys.exit(3)
            done = True
            break
            # done

        for (u, v) in pairwise(residual_path):
            graph[u][v][capacity] -= pathflow
            if graph[u][v][capacity] == 0:
                graph.remove_edge(u, v)
        yield ((residual_path, pathflow))


# need to replace dicts and sets with sorted dicts/sets, to make experiments reproducible in Python 3.3+
# pad the servers with extra info if they dont have it (only some of our instances list combined bandwidth)
sorted_servers = OrderedDict()  # for reproducibility
extended_server_resources = OrderedDict()
n_extended_resources = 0
for servername in sorted(instance["Servers"]):
    data = instance["Servers"][servername]
    extended_server_resources[servername] = []
    if len(data) == 5:  # Support for the nfv instance

        cores, ram, bandwidth, video_cores, fast_storage = data
        sorted_servers[servername] = [cores, ram, 0, 0]
        extended_server_resources[servername] = [video_cores, fast_storage]
        n_extended_resources = 2
    elif len(data) == 4:
        cores, ram, storage, bandwidth = data
        sorted_servers[servername] = [cores, ram, storage, bandwidth]
    else:
        assert (len(data) == 3)
        cores, ram, storage = data
        sorted_servers[servername] = [cores, ram, storage, 0]  # 0 to fill in for missing storage capacity.
instance["Servers"] = sorted_servers

if n_extended_resources > 0:
    print("Interpreting datacenter as an NFV instance!")

if ignore_ram_constraints:
    print("Warning - ignoring RAM constraints")
print(instance_name)

# for each maxflow requirement, create a copy of the physical network (slightly altered to convert multiple destinations into a single destination)
bvwidth = 1
# for u,v,bandwidth in instance["VN"]:
#     if (bandwidth>= 1<<bvwidth):
#         bvwidth = math.ceil(math.log(bandwidth+1)/math.log(2))
for s1, s2, bandwidth in instance["PN"]:
    if (bandwidth >= 1 << bvwidth):
        bvwidth = math.ceil(math.log(bandwidth + 1) / math.log(2))
bvwidth += 1

vm_new_to_original_name_map=dict()
vm_original_to_new_name_map = dict()
def getOriginalVMName(vm):
    return vm_unmap[vm_new_to_original_name_map[vm]]
# enforce bandwidth constraints
# to do this, first split the virtual network into source and destination nodes.

def renameVN(instance, original_maxflow_requirements):
    global vm_new_to_original_name_map
    global m_original_to_new_name_map
    vm_new_to_original_name_map = dict()
    vm_original_to_new_name_map = dict()
    # vm's will be renamed in order to put the ones that are designated as 'sources' first

    used_sources = set()
    for source in original_maxflow_requirements:
        if source in original_maxflow_requirements:
            flow_requirements = original_maxflow_requirements[source]
            if len(flow_requirements) == 0:
                continue;
            used_sources.add(source)
    next_unused = 0
    for vm in used_sources:
        assert (vm not in vm_original_to_new_name_map)
        vm_original_to_new_name_map[vm] = vms[next_unused]
        vm_new_to_original_name_map[vms[next_unused]] = vm
        next_unused += 1

    for vm in vms:
        if vm not in vm_original_to_new_name_map:
            vm_original_to_new_name_map[vm] = vms[next_unused]
            vm_new_to_original_name_map[vms[next_unused]] = vm
            next_unused += 1

    sorted_vms = OrderedDict()  # for reproducibility
    extended_vm_resources = OrderedDict()
    for vm in sorted(instance["VMs"]):
        data = instance["VMs"][vm]
        vm_name = vm_original_to_new_name_map[vm]
        sorted_vms[vm_name] = data

    instance["VMs"] = sorted_vms
    vdc = []
    for u, v, bandwidth in instance["VN"]:
        vdc.append([vm_original_to_new_name_map[u], vm_original_to_new_name_map[v], bandwidth])

    instance["VN"] = vdc
    maxflow_requirements = dict()
    for vm in vms:
        original_vm = vm_new_to_original_name_map[vm]
        if original_vm in original_maxflow_requirements.keys():
            original_req = original_maxflow_requirements[original_vm]
            new_req = []
            for (u, sum_bandwidth) in original_req:
                new_req.append((vm_original_to_new_name_map[u], sum_bandwidth))
            maxflow_requirements[vm] = new_req

    return instance, maxflow_requirements


def loadVN(instance):
    global vm_unmap
    vm_unmap = dict()
    vm_map = dict()
    if (len(instance["VMs"]) > len(vms)):
        print("Insufficient vms modelled (modelled %d, required %d), increase --max-vms" % (
        len(vms), len(instance["VMs"])))
        assert (False)
    unused_vms = list(vms)
    unused_vms.reverse()
    sorted_vms = OrderedDict()  # for reproducibility
    extended_vm_resources = OrderedDict()
    for vm in sorted(instance["VMs"]):
        data = instance["VMs"][vm]
        vm_name = unused_vms[-1]
        vm_map[vm] = vm_name
        vm_unmap[vm_name] = vm
        unused_vms.pop()
        if len(data) == 5:
            cores, ram, bandwidth, video_cores, fast_storage = data
            sorted_vms[vm_name] = [cores, ram, 0, 0]
            extended_vm_resources[vm_name] = [video_cores, fast_storage]
        elif len(data) == 4:
            sorted_vms[vm_name] = instance["VMs"][vm]
        else:
            assert (len(data) == 3)
            cores, ram, total_bandwidth = data
            sorted_vms[vm_name] = [cores, ram, 0, 0]
            if (cores > max_cores):
                print("Increase max vm cores (has %d, required %d)" % (max_cores, cores))
                assert (False)
            if (ram > max_ram):
                print("Increase max vm cores (has %d, required %d)" % (max_ram, ram))
                assert (False)

    instance["VMs"] = sorted_vms

    if (len(sorted_vms) > max_virt_nodes):
        print("Too many vms (%d)" % (len(sorted_vms)))
        sys.exit(1)

    vdc = []
    for u, v, bandwidth in instance["VN"]:
        vdc.append([vm_map[u], vm_map[v], bandwidth])

    instance["VN"] = vdc

    virt_network = nx.Graph() if not directed_links else nx.DiGraph()
    for u, v, bandwidth in instance["VN"]:
        virt_network.add_edge(u, v)
    # printGraph(virt_network,None)
    virtual_switches = set()
    maxflow_requirements = dict()
    for n in virt_network:
        maxflow_requirements[n] = []

    original_vn = list(instance["VN"])

    refinement_maxflow_requirements = dict()
    if not force_virtual_switches:
        # find an approximate vertex cover
        if use_cover:
            cover = min_weighted_vertex_cover(virt_network);

            # for each node in the vertex cover, find the set of all edges incident to that node in the virtual network that are not yet covered;
            # those form a single maxflow query.


            covered_edges = set()
            for n in cover:
                refinement_maxflow_requirements[n] = dict()
                refinement_maxflow_requirements[n][n] = []
                for u, v, bandwidth in instance["VN"]:

                    if (u, v) not in covered_edges and (u == n or v == n):
                        covered_edges.add((u, v))

                        assert (u == n or v == n)
                        if u == n:
                            dest = v
                        else:
                            dest = u
                        maxflow_requirements[n].append((dest, bandwidth))
                        refinement_maxflow_requirements[n][n].append((dest, bandwidth))
        else:
            for n in virt_network:
                refinement_maxflow_requirements[n] = dict()
                refinement_maxflow_requirements[n][n] = []
                for u, v, bandwidth in instance["VN"]:
                    if u == n:
                        maxflow_requirements[n].append((v, bandwidth))
                        refinement_maxflow_requirements[n][n].append((v, bandwidth))


                        # if use_cover_for_refinement:
                        #    refinement_maxflow_requirements = maxflow_requirements.copy()

    else:
        components = networkx.algorithms.components.connected_components(virt_network)
        # original_vn=list(instance["VN"])
        if use_cover_for_refinement:
            cover = min_weighted_vertex_cover(virt_network);
        for component in components:
            # introduce a switch, IF the component is not already covered by a single node.
            covering_node = None
            component_nodes = set(component)
            intersect = component_nodes
            for edge in virt_network.edges(component):
                intersect = intersect & (set(edge))
            if len(intersect) > 0:
                # intersect can have at most two nodes
                covering_node = next(iter(intersect))
            if covering_node is not None:
                for u, v, bandwidth in instance["VN"]:
                    if u == covering_node:
                        maxflow_requirements[covering_node].append((v, bandwidth))
                    elif v == covering_node:
                        maxflow_requirements[covering_node].append((u, bandwidth))

            elif covering_node is None:
                # introduce a virtual switch, and modify the VN to run all the bandwidth in this component through it.
                sw = "virtual_switch%d" % (len(virtual_switches))
                virtual_switches.add(sw)
                covering_node = sw
                maxflow_requirements[sw] = []
                sum_bw = OrderedDict()
                for n in component:
                    sum_bw[n] = 0
                newvn = []
                for u, v, bandwidth in instance["VN"]:

                    if u in component_nodes:
                        assert (v in component_nodes)
                        sum_bw[u] += bandwidth
                        sum_bw[v] += bandwidth
                    else:
                        assert (v not in component_nodes)
                        newvn.append((u, v, bandwidth))
                total_bandwidth = 0
                for u, sum_bandwidth in sum_bw.items():
                    newvn.append((u, sw, sum_bandwidth))
                    maxflow_requirements[sw].append((u, sum_bandwidth))
                    total_bandwidth += sum_bandwidth

                instance["VMs"][sw] = [0, 0, 0,
                                       total_bandwidth]  # A virtual switch is a vm with neither ram nor core requirements (and hence doesn't need to be allocated to any particular server).
                instance["VN"] = newvn

            refinement_maxflow_requirements[covering_node] = dict()
            for n in component:
                refinement_maxflow_requirements[covering_node][n] = []
            if use_cover_for_refinement:
                covered_edges = set()
                for n in cover:
                    if n in component:
                        # this can be improved
                        for u, v, bandwidth in original_vn:

                            if (u, v) not in covered_edges and (u == n or v == n):
                                covered_edges.add((u, v))
                                assert (u in component)
                                assert (v in component)
                                assert (u == n or v == n)
                                if u == n:
                                    dest = v
                                else:
                                    dest = u
                                refinement_maxflow_requirements[covering_node][n].append((dest, bandwidth))
            else:
                for u, v, bandwidth in original_vn:
                    if u in component or v in component:
                        assert (v in component)
                        assert (u in component)
                        refinement_maxflow_requirements[covering_node][u].append((v, bandwidth))

        if len(virtual_switches) > 0:
            print("Added %d virtual switches" % (len(virtual_switches)))

    # if len(maxflow_requirements)==0:
    #     print("Warning: no flow requirements!")
    #     sys.exit(1)
    for source in instance["VMs"]:
        # for source, flow_requirements in maxflow_requirements.items():
        sum_bandwidth = 0
        if source in maxflow_requirements:
            flow_requirements = maxflow_requirements[source]
            if len(flow_requirements) == 0:
                continue;
            sum_bandwidth = 0
            for dest, bandwidth in flow_requirements:
                sum_bandwidth += bandwidth
        if (sum_bandwidth >= 1 << bvwidth):
            print("BV is too small: sum is %d, width is %d" % (sum_bandwidth, bvwidth))
            sys.exit(1)
            # bvwidth = math.ceil(math.log(sum_bandwidth+1)/math.log(2))

    return renameVN(instance, maxflow_requirements)


# using networkx to find the cover (so first need to construct the networkx virtual graph):



print("Bitvector width is %d" % (bvwidth))

monosat.bvtheory.BVManager().bitblast_addition = bitblast_addition
monosat.bvtheory.BVManager().bitblast_addition_shadow = bitblast_addition_shadow
physical_graph = Graph()

# we have a set of physical servers, connected by a fixed topology of switches, with fixed bandwidth
physical_servers = []  # (node,servername,[cores,ram,bandwidth])
hostdict = dict()
switches = []
switchdict = dict()
servercores = dict()
serverram = dict()
serverstorage = dict()
serverbandwidth = dict()

vmcores = dict()
vmram = dict()
vmstorage = dict()

# vmbandwidth=dict()#aggregate bandwidth requirements of the vm, equal to the sum of all its bandwidth requirements in the virtual network

physical_nodes = dict()
for servername in instance["Servers"]:
    # List of servers that can host virtual machines
    data = instance["Servers"][servername]
    data = list(map(int, data))
    cores, ram, storage, bandwidth = data
    servercores[servername] = cores
    serverram[servername] = ram
    serverstorage[servername] = storage
    serverbandwidth[servername] = bandwidth

    node = physical_graph.addNode(servername)
    physical_nodes[servername] = node
    hostdict[servername] = len(physical_servers)
    physical_servers.append((node, servername, data))

# print("graph{\nrankdir = BT")
# for s in instance["Servers"]:
#    print(s + " [label = \"c%d m%d d%d \"]"%(servercores[s],serverram[s],serverstorage[s]))
switches = []
pn_edges = []
for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
    # each connection is of the form ["s0", "tor0", 1000]
    bandwidth = int(bandwidth)

    if s1 in physical_nodes:
        n1 = physical_nodes[s1]
    else:
        n1 = physical_graph.addNode(s1)
        physical_nodes[s1] = n1
        switches.append(s1)

    if s2 in physical_nodes:
        n2 = physical_nodes[s2]
    else:
        n2 = physical_graph.addNode(s2)
        physical_nodes[s2] = n2
        switches.append(s2)
    if not directed_links:
        e = physical_graph.addUndirectedEdge(n1, n2, bandwidth)
    else:
        e = physical_graph.addEdge(n1, n2, bandwidth)
    pn_edges.append((e, (s1, s2, bandwidth)))
    if assert_all_physical_edges:
        Assert(e)
# print(s1 + " -- " + s2 + "[xlabel=\"%d Mbps\"]"%(bandwidth))
# print("}")


assume_false = []

virt_nodes_out = dict()
virt_nodes_in = dict()
max_virtbandwidth = dict()

for n in vms:
    # cores,ram,storage,aggregatebandwidth =  instance["VMs"][n]
    v = physical_graph.addNode("virt_out_" + str(n))
    virt_nodes_out[n] = v
    v2 = physical_graph.addNode("virt_in_" + str(n))
    virt_nodes_in[n] = v2

    vmcores[n] = [Var() for n in range(max_cores)]  # BitVector(vm_bv_width) #cores
    if not ignore_ram_constraints:
        vmram[n] = [Var() for n in range(max_ram)]  # BitVector(vm_bv_width) #ram
    else:
        vmram[n] = []
    if not ignore_storage_constraints:
        vmstorage[n] = [Var() for n in range(max_storage)]  # BitVector(vm_bv_width) #storage
    else:
        vmstorage[n] = []

    # vmbandwidth[n] = BitVector(vm_bv_width) #=maxbandwidth
    max_virtbandwidth[n] = max_vdc_bandwidth  # BitVector(vm_bv_width) #=maxbandwidth
#
for n in extra_vms:

    vmcores[n] = [Var() for n in range(max_cores)]
    if not ignore_ram_constraints:
        vmram[n] = [Var() for n in range(max_ram)]
    else:
        vmram[n] = []

    if not ignore_storage_constraints:
        vmstorage[n] = [Var() for n in range(max_storage)]
    else:
        vmstorage[n] = []

    for v in vmcores[n]:
        assume_false.append(~v)

    for v in vmram[n]:
        assume_false.append(~v)

    for v in vmstorage[n]:
        assume_false.append(~v)

vm_server_assignments = dict()
server_vm_assignments = dict()
switch_vm_assignments = dict()

unused_vm_edges = dict()

for host in instance["Servers"]:
    server_vm_assignments[host] = dict()
for switch in switches:
    switch_vm_assignments[switch] = dict()
all_edge_sets = []
edge_sets = dict()
for vm in vms:
    edge_sets[vm] = []
    start_vm = time.clock()
    server_edges = []
    virtnode_out = virt_nodes_out[vm]
    virtnode_in = virt_nodes_in[vm]
    maxbandwidth = max_virtbandwidth[vm]

    for host in instance["Servers"]:
        e = physical_graph.addEdge(virtnode_out, physical_nodes[host], maxbandwidth)
        # e.setSymbol("virt_"+str(vm) + "to_host_" + str(host))

        server_edges.append((e, host))
        server_vm_assignments[host][vm] = e
        e2 = physical_graph.addEdge(physical_nodes[host], virtnode_in, maxbandwidth)
        AssertEq(e, e2)

    # this is a virtual switch, and so can be allocated to switches in the network
    if vmcores[
        vm] is None and allow_virtual_switches_in_switches:  # and vmram[vm]==0  and vmstorage[vm]==0 and allow_virtual_switches_in_switches:

        for host in switches:
            e = physical_graph.addEdge(virtnode_out, physical_nodes[host], maxbandwidth)
            # e.setSymbol("virt_"+str(vm) + "to_switch_" + str(host))

            server_edges.append((e, host))
            switch_vm_assignments[host][vm] = e
            # server_vm_assignments[host][vm]=e
            e2 = physical_graph.addEdge(physical_nodes[host], virtnode_in, maxbandwidth)
            AssertEq(e, e2)

    # Each virtual machine is assigned to exactly one physical server, if the vm is used, and exactly no servers if the vm is not used.
    unused_vm_edge = physical_graph.addEdge(virtnode_out, virtnode_in, 0)
    unused_vm_edges[vm] = unused_vm_edge

    edge_vars = [x[0] for x in server_edges] + [unused_vm_edge]
    vm_server_assignments[vm] = server_edges

    if (intrinsic_edge_sets):
        physical_graph.newEdgeSet(edge_vars)
        all_edge_sets.append(edge_vars)
        edge_sets[vm].append(edge_vars)
    elif (intrinsic_edge_constraints):
        AssertExactlyOne(edge_vars);
    else:
        AssertEqualPB(edge_vars, 1);
        if flush_pb_theory:  # In large physical networks, these PB constraints can become excessively expensive to encode (because minisat+ is trying very hard to optimize them); forcing them into clauses immediately keeps minisat+ from trying those optimizations  (which are probably not very helpful here anyhow.)
            PBManager().flush()

print(Monosat().nVars())
print("Enforcing server core/ram counts")

server_core_assignments = dict()
server_ram_assignments = dict()
server_storage_assignments = dict()
server_extended_assignments = dict()
# Now, ensure that no server is assigned more virtual machines than it has the capacity to host.
# For now, we'll ignore ram, and model server capacity as just the number of cores.
# In the fmcad paper, each server has a capacity of 4 virtual machines.
for host in instance["Servers"]:
    cores, ram, storage, bandwidth = servercores[host], serverram[host], serverstorage[host], serverbandwidth[host]
    extended_server_res = extended_server_resources[host]
    assert (len(extended_server_res) == n_extended_resources)
    vm_assigns = server_vm_assignments[host]
    core_edges = []
    core_weights = []
    ram_edges = []
    ram_weights = []
    storage_edges = []
    storage_weights = []
    extended_edges = []

    edges = []
    used_cores = [Var() for n in range(cores)]  # BitVector(vm_bv_width)
    used_core_weights = list(range(1, cores + 1))

    if not ignore_ram_constraints:
        used_ram = [Var() for n in range(ram)]  # BitVector(vm_bv_width)
        used_ram_weights = list(range(1, ram + 1))
    if not ignore_storage_constraints:
        used_storage = [Var() for n in range(storage)]  # BitVector(vm_bv_width)
        used_storage_weights = list(range(1, storage + 1))
    used_ex = []
    for r in range(n_extended_resources):
        extended_edges.append([]);
        used_ex.append([Var() for n in range(extended_server_res[r])])

    for vmname in vms:
        if vmname in vm_assigns:
            edge = vm_assigns[vmname]
            edges.append(edge)
            nvmcores, nvmram, nvstorage = vmcores[vmname], vmram[vmname], vmstorage[vmname]

            for i, v in enumerate(nvmcores):
                core_edges.append(And(edge, v))
                core_weights.append(i + 1)

            if not ignore_ram_constraints:
                for v in nvmram:
                    ram_edges.append(And(edge, v))
                    ram_weights.append(i + 1)
            if not ignore_storage_constraints:
                for v in nvstorage:
                    storage_edges.append(And(edge, v))
                    storage_weights.append(i + 1)
            for r in range(n_extended_resources):
                extr = extended_vm_resources[vmname]
                w = extr[r]
                for v in extr:
                    extended_edges[r].append(And(edge, v))

                    # extended_weights[r].append((If(edge, w,0)))

    for vmname in extra_vms:

        edge = Var()
        assume_false.append(~edge)
        nvmcores, nvmram, nvstorage = vmcores[vmname], vmram[vmname], vmstorage[vmname]

        for i, v in enumerate(nvmcores):
            core_edges.append(And(edge, v))
            core_weights.append(i + 1)
        if not ignore_ram_constraints:
            for i, v in enumerate(nvmram):
                ram_edges.append(And(edge, v))
                ram_weights.append(i + 1)
        if not ignore_storage_constraints:
            for i, v in enumerate(nvstorage):
                storage_edges.append(And(edge, v))
                storage_weights.append(i + 1)

    server_core_assignments[host] = (edges, cores, used_cores)
    if not ignore_ram_constraints:
        server_ram_assignments[host] = (edges, ram, used_ram)
    else:
        server_ram_assignments[host] = ([], [], [])
    if not ignore_storage_constraints:
        server_storage_assignments[host] = (edges, storage, used_storage)
    else:
        server_storage_assignments[host] = ([], [], [])

    server_extended_assignments[host] = (edges, extended_server_res, used_ex)

    # print("core weights")
    AssertLessEqPB(core_edges + used_cores, cores, core_weights + used_core_weights);  # Weighting t
    # if flush_pb_theory: #In large physical networks, these PB constraints can become excessively expensive to encode (because minisat+ is trying very hard to optimize them); forcing them into clauses immediately keeps minisat+ from trying those optimizations  (which are probably not very helpful here anyhow.)
    #    PBManager().flush()
    # print("ram" + str(ram_weights+used_ram_weights))
    if not ignore_ram_constraints:
        AssertLessEqPB(ram_edges + used_ram, ram, ram_weights + used_ram_weights);
    # if flush_pb_theory: #In large physical networks, these PB constraints can become excessively expensive to encode (because minisat+ is trying very hard to optimize them); forcing them into clauses immediately keeps minisat+ from trying those optimizations  (which are probably not very helpful here anyhow.)
    #    PBManager().flush()
    # print("storage" + str(storage_weights+used_storage_weights))
    if not ignore_storage_constraints:
        AssertLessEqPB(storage_edges + used_storage, storage, storage_weights + used_storage_weights);
    # if flush_pb_theory: #In large physical networks, these PB constraints can become excessively expensive to encode (because minisat+ is trying very hard to optimize them); forcing them into clauses immediately keeps minisat+ from trying those optimizations  (which are probably not very helpful here anyhow.)
    #    PBManager().flush()
    # print("extended")
    for r in range(n_extended_resources):
        AssertLessEqPB(extended_edges[r] + used_ex[r], extended_server_res[r]);
    if flush_pb_theory:  # In large physical networks, these PB constraints can become excessively expensive to encode (because minisat+ is trying very hard to optimize them); forcing them into clauses immediately keeps minisat+ from trying those optimizations  (which are probably not very helpful here anyhow.)
        PBManager().flush()

        # Note: need to be careful here, because two vms in the same server have free bandwidth between each other, which may impact the above bandwidth requirement calculation
        # As a result, this constraint is not valid:
        # AssertLessEqPB(edges,bandwidth,bandwidth_weights);

# We will then assert that
# a) all physical network copies have the same edge assignment, and
# b) the sum of the bandwidth of all physical network copy edges is <= the allowed bandwidth for that edge

physical_graph_copies = []
all_edges = [[] for i in instance["PN"]]

flow_requirement_lits = dict()
enforced_sums = False


def getFlowRequirement(source, buildIfNotExists=False, forceGraphConstruction=False, unusedSource=False):
    if source in flow_requirement_lits:
        (
        physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, potential_dests,
        sum_bandwidth) = flow_requirement_lits[source]
        if physical_graph_copy is not None:
            return flow_requirement_lits[source]
    else:
        flow_requirement_lits[source] = (None, None, None, None, None, None, None)
    (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, potential_dests,
     sum_bandwidth) = flow_requirement_lits[source]
    if not buildIfNotExists:
        return flow_requirement_lits[source]
    if internal_edge_bandwidths is None:
        # build the bitvectors first
        print("Building edges for source %s" % (source))
        assert (not (enforced_sums))
        # then add the graph, only if required
        sum_bandwidth = BitVector(bvwidth)
        internal_edge_bandwidths = []
        internal_edge_bandwidths_limits = []
        for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
            u = physical_nodes[s1]
            v = physical_nodes[s2]

            bv2 = BitVector(bvwidth)
            bv2.setSymbol("bw_" + str(u) + "to" + str(v) + "_source_" + str(source))

            internal_edge_bandwidths.append(bv2)
            Assert(bv2 <= bandwidth)
            #
            # if not unusedSource:
            all_edges[i].append(bv2)
            internal_edge_bandwidths_limits.append(bv2 <= 0)
        potential_dests = dict()
        for dest in vms:
            if dest == source:
                continue

            bandwidth = BitVector(bvwidth)
            potential_dests[dest] = (bandwidth)

        flow_requirement_lits[source] = (
        None, internal_edge_bandwidths, internal_edge_bandwidths_limits, None, None, potential_dests, sum_bandwidth)

    if not forceGraphConstruction:
        return flow_requirement_lits[source]

    print("maxflow for " + source)
    # if not unusedSource:
    physical_graph_copy = Graph()
    for n in range(physical_graph.nodes):
        physical_graph_copy.addNode()
    # else:
    #    physical_graph_copy = None
    # add all the edges of the physical network, with bitvector weights.
    # and add the edges for the _relevant_ virtual machines for this flow query.
    copy_edges = dict()

    for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):

        u = physical_nodes[s1]
        v = physical_nodes[s2]
        bv2 = internal_edge_bandwidths[i]

        # if not unusedSource:
        e2, (s1_check, s2_check, bandwidth_check) = pn_edges[i]
        assert (s1 == s1_check)  # sanity checks
        assert (s2 == s2_check)
        # assert(bandwidth==bandwidth_check)


        if not directed_links:
            eforward = physical_graph_copy.addEdge(u, v, bv2)
            eback = physical_graph_copy.addEdge(v, u, bv2)
            AssertEq(eforward, eback)
        else:
            eback = None
            eforward = physical_graph_copy.addEdge(u, v, bv2)
        # e = physical_graph_copy.addUndirectedEdge(u,v,bv2)
        AssertEq(eforward,
                 e2)  # ensure that this edge in the physical network copy is enabled only if it is enabled in the master physical network.

        copy_edges[i] = (eforward, eback)
        # This edge can only have non-zero weight if it is enabled.
        if not assert_all_physical_edges:
            AssertImplies(bv2 > 0, eforward)

    # if not unusedSource:
    edgelitmap = dict()
    # add edges for the virtual machine source
    for var, host in vm_server_assignments[source]:
        h = physical_nodes[host]
        vm = virt_nodes_out[source]
        e = physical_graph_copy.addEdge(vm, h, sum_bandwidth)
        AssertEq(e, var);
        edgelitmap[var.getLit()] = e
    vm_unused = physical_graph_copy.addEdge(virt_nodes_out[source], virt_nodes_in[source], BitVector(bvwidth, 0))
    original_vm_unused_edge = unused_vm_edges[source]
    AssertEq(vm_unused, original_vm_unused_edge)
    edgelitmap[original_vm_unused_edge.getLit()] = vm_unused

    # if not unusedSource:
    if (intrinsic_edge_sets):
        for edgeset in edge_sets[source]:
            copyedgeset = [edgelitmap[e.getLit()] for e in edgeset]
            physical_graph_copy.newEdgeSet(copyedgeset)

    for dest in vms:
        if dest == source:
            continue

        bandwidth = potential_dests[dest]

        # if not unusedSource:
        vm = virt_nodes_in[dest]
        # assert(bandwidth<=max_virtbandwidth[dest])
        for var, host in vm_server_assignments[dest]:
            h = physical_nodes[host]
            e = physical_graph_copy.addEdge(h, vm, bandwidth)
            AssertEq(e, And(var, Not(original_vm_unused_edge)))
            # else:
            #    potential_dests[dest] = (BitVector(bvwidth,0))
            # if there are multiple flow requirements for this source, collect them all into one, new sink node

    sum_dest = physical_graph_copy.addNode()
    # if not unusedSource:
    for dest, bandwidth in potential_dests.items():
        # dests.append(dest)
        virt_dest = virt_nodes_in[dest]
        Assert(physical_graph_copy.addEdge(virt_dest, sum_dest, bandwidth))

    # if not unusedSource:
    virt_source = virt_nodes_out[source]
    mf = physical_graph_copy.maxFlowGreaterOrEqualTo(virt_source, sum_dest, sum_bandwidth)
    # else:
    #    mf = Var()
    flow_requirement_lits[source] = (
    physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, potential_dests,
    sum_bandwidth)
    Assert(mf)
    # all_flow_requirements[source] =(mf,sum_bandwidth)
    return flow_requirement_lits[source]

    # for source in vms:
    #    getFlowRequirement(source)


used_sources = set()

# only instantiate up to max_vms/2 (as that is the maximum possible required number of flow sources)
for i, source in enumerate(vms):
    # for undirected links, only n//2 possible vm sources are required
    (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, potential_dests,
     sum_bandwidth) = getFlowRequirement(source, True, False)

print("enforcing sums")
sums = dict()
# ensure the sum of all edges in the network copies add up to less or equal to the actual bandwidth of that edge
for i, (s1, s2, wt) in enumerate(instance["PN"]):

    n1 = physical_nodes[s1]
    n2 = physical_nodes[s2]
    assert (len(all_edges[i]) > 0);
    # if not bitblast_pn_addition:
    if not tree_addition:
        sm = BitVector(bvwidth, 0)

        for bv2 in all_edges[i]:
            sm += bv2

    else:
        elements = list(all_edges[i])
        while (len(elements) > 1):
            if (len(elements) % 2 == 1):
                elements.append(BitVector(bvwidth, 0))

            next_elements = []
            for a, b in zip(elements[::2], elements[1::2]):
                next_elements.append(a + b)
            elements = next_elements
        assert (len(elements) == 1);
        sm = elements[0]
    sums[i] = (sm, wt)
    Assert(sm <= wt)
enforced_sums = True


min_bvs = []
assumptions = []






print("solving...")

total_core_assignment = dict()
total_ram_assignment = dict()
total_storage_assignment = dict()
total_extended_resource_assignment = dict()
for host in instance["Servers"]:
    total_core_assignment[host] = 0
    total_ram_assignment[host] = 0
    total_storage_assignment[host] = 0
    total_extended_resource_assignment[host] = []
    for r in range(n_extended_resources):
        total_extended_resource_assignment[host].append(0)

residual_network = nx.Graph() if not directed_links else nx.DiGraph()

for u, v, bandwidth in instance["PN"]:
    residual_network.add_edge(u, v, weight=bandwidth, cost=1)
    if not directed_links:
        assert (residual_network[v][u]['weight'] == bandwidth)
    else:
        if not residual_network.has_edge(v, u):
            residual_network.add_edge(v, u, weight=0, cost=1)
n_solutions = 0

elapsed = 0
elapsed_out = 0
used_hosts = set()
print("Done Encoding..")
t = time.clock()
log = None
if log_file is not None:
    log = open(log_file, "w")

if log_time_file is not None:
    timefile = open(log_time_file, "w")
    timefile.write("# instance " + instance_name + "\n")
    timefile.write("# " + "Settings: " + str(args) + "\n")
    prev_time = t
    timefile.write("init %f %f\n" % (t - start_time, t - start_time))

# os.system("callgrind_control -i on")

bandwidth_assump_lits = dict()

for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
    bandwidth_assump_lits[i] = true()

# set the used_cores, etc
for host in instance["Servers"]:
    assigned_cores = 0
    assigned_ram = 0
    assigned_storage = 0
    assigned_ext = [0] * n_extended_resources

    used_cores, used_ram, used_storage, used_ex = server_core_assignments[host][-1], server_ram_assignments[host][-1], \
                                                  server_storage_assignments[host][-1], \
                                                  server_extended_assignments[host][-1]
    for v in used_cores:
        assumptions.append(~v)
    for v in used_ram:
        assumptions.append(~v)
    for v in used_storage:
        assumptions.append(~v)
    for res in used_ex:
        for v in res:
            assumptions.append(~v)
            # assumptions.append(used_cores== 0)
            # assumptions.append(used_ram== 0)
            # assumptions.append(used_storage== 0)
            # for r in used_ex:
            #     assumptions.append(r== 0)
inst,vdc_name = selectVDC()
vm_instance, maxflow_requirements = loadVN(inst)

unused_sources = set(vms)
for source in vms:
    # for source, flow_requirements in maxflow_requirements.items():

    if source in maxflow_requirements:
        flow_requirements = maxflow_requirements[source]
        if len(flow_requirements) == 0:
            continue;
        sum_flow = 0
        for dest, bandwidth in flow_requirements:
            sum_flow += bandwidth
        (
        physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, potential_dests,
        sum_bandwidth) = getFlowRequirement(source, True, True)

        non_dests = set(vms)
        non_dests.remove(source)
        for dest, bandwidth in flow_requirements:
            assigned_bandwidth = potential_dests[dest]
            assumptions.append(assigned_bandwidth == bandwidth)
            non_dests.remove(dest)
        for dest in non_dests:
            assigned_bandwidth = potential_dests[dest]
            assumptions.append(assigned_bandwidth <= 0)
        print("Setting sum flow for graph  for source " + str(source) + " to " + str(sum_flow))
        unused_sources.remove(source)
        assumptions.append(sum_bandwidth == sum_flow)
for source in unused_sources:
    if not only_instantiate_networks_on_demand:

        # disable this maxflow requirement


        (
        physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, potential_dests,
        sum_bandwidth) = getFlowRequirement(source)
        # constraint the maxflow at the begining of the assumptions - this is important!
        if sum_bandwidth is not None:
            assumptions.append(sum_bandwidth <= 0)  # since maximum flow is always greater or equal to 0 in any graph,
        # the maxflow>=sum_bandwidth constraint becomes a tautology is sum_bandwidth is <=0, and ideally can be optimized
        # away by the solver.
        if potential_dests is not None:
            for vm, bandwidth in potential_dests.items():
                assumptions.append(bandwidth <= 0)
        if internal_edge_bandwidths_limits is not None:
            for bandwidth_zero in internal_edge_bandwidths_limits:
                assumptions.append(bandwidth_zero)

used_vms = set(vm_instance["VMs"].keys())
unused_vms = set(vms)
unused_vms = unused_vms.difference(used_vms)
for vm in used_vms:
    cores, ram, storage, aggregatebandwidth = vm_instance["VMs"][vm]
    assumptions.append(~unused_vm_edges[vm])  # force the vm to be assigned to a server
    assert (cores <= max_cores)
    assert (ram <= max_ram)
    assert (storage <= max_storage)

    for i in range(max_cores):
        v = vmcores[vm][i]
        if i + 1 == cores:
            assumptions.append(v)
        else:
            assumptions.append(~v)
    if not ignore_ram_constraints:
        for i in range(max_ram):
            v = vmram[vm][i]
            if i + 1 == ram:
                assumptions.append(v)
            else:
                assumptions.append(~v)
    if not ignore_storage_constraints:
        for i in range(max_storage):
            v = vmstorage[vm][i]
            if i + 1 == storage:
                assumptions.append(v)
            else:
                assumptions.append(~v)

for vm in unused_vms:
    assumptions.append(unused_vm_edges[vm])  # prevent the vm from being assigned to a server
    for i in range(max_cores):
        v = vmcores[vm][i]
        assumptions.append(~v)
    if not ignore_ram_constraints:
        for i in range(max_ram):
            v = vmram[vm][i]
            assumptions.append(~v)
    if not ignore_storage_constraints:
        for i in range(max_storage):
            v = vmstorage[vm][i]
            assumptions.append(~v)
    for host in instance["Servers"]:
        assumptions.append(~server_vm_assignments[host][vm])


        # for  n in vm_instance["VMs"]:
        #     cores,ram,storage,aggregatebandwidth =  vm_instance["VMs"][n]
        #
        #
        #     maxbandwidth =0
        #     #speed this up:
        #     for vm1, vm2,bandwidth in vm_instance["VN"]:
        #         if vm1 == n or vm2 == n:
        #             maxbandwidth+=int(bandwidth)
        #     max_virtbandwidth[n]=maxbandwidth
        #     assumptions.append(vmcores[n]==cores);
        #     assumptions.append(vmram[n]==ram);
        #     assumptions.append(vmstorage[n]==storage);
        # extended resources go here...
json_data = dict()
n_allocations=0
assumptions.extend(assume_false)
de_allocating = True
while (de_allocating):
    while Solve(assumptions, bvs_to_minimize=min_bvs):
        # os.system("callgrind_control -i off")
        et = time.clock()
        elapsed += et - t
        assumptions = []

        solution_data = dict()
        allocation_data = list()

        solution_data["assignment"] = allocation_data
        bandwidth_data = list()
        solution_data["bandwidth"] = bandwidth_data
        json_data["allocation_%s_%d" % (vdc_name,n_allocations)] = solution_data
        n_solutions+=1
        n_allocations+=1
        print("Solved %d"%(n_solutions))
        found_optimal = FoundOptimal()

        # shutil.copy(filename,filename[:-3]+str(n_solutions) + ".gnf")
        if log:
            log.write("# vdc allocation %d\n" % (n_solutions))
            if (found_optimal):
                log.write("# proved optimal allocation\n")
            else:
                log.write("# failed to prove optimal allocation\n")
            log.write("vdc\n")

        if log_time_file:
            timefile.flush()

        vdc_allocation_map[n_solutions] = dict()
        allocated_vdcs.add(n_solutions)

        vm_assignments = dict()
        server_assigned_cores = dict()
        server_assigned_ram = dict()
        server_assigned_storage = dict()
        server_assigned_ext = dict()
        for host in instance["Servers"]:
            assigned_cores = 0
            assigned_ram = 0
            assigned_storage = 0
            assigned_ext = [0] * n_extended_resources
            for vm in vm_instance["VMs"]:
                e = server_vm_assignments[host][vm]

                if e.value():
                    # vm was assigned to host
                    # so, we need to reduce the number of available cores and ram on the host.
                    vm_assignments[vm] = host
                    cores = vmcores[vm]
                    ram = vmram[vm]
                    storage = vmstorage[vm]
                    ncores = count(cores)
                    assert (ncores > 0)
                    assigned_cores += ncores

                    assigned_ram += count(ram)
                    assigned_storage += count(storage)
                    for r in range(n_extended_resources):
                        extr = extended_vm_resources[vm]
                        assigned_ext[r] += count(extr[r])
                    if log:
                        log.write("%s -> %s\n" % (vm, host))
                    if(output_file):
                        allocation_data.append((getOriginalVMName(vm),host))
                    # print("vm " + vm + " assigned to " + host)
                    used_hosts.add(host)

            server_assigned_cores[host] = assigned_cores
            server_assigned_ram[host] = assigned_ram
            server_assigned_storage[host] = assigned_storage
            server_assigned_ext[host] = assigned_ext
        # for virtual_switch in virtual_switches:
        #     if virtual_switch not in  vm_assignments:
        #         #the virtual switch was attached to a switch, not a server
        #         for switch in switches:
        #             e = switch_vm_assignments[switch][virtual_switch]
        #             if e.value():
        #                 vm_assignments[virtual_switch]=switch
        #                 print("vswitch " + vm + " assigned to " + host)
        for vm in vm_instance["VMs"]:
            if vm not in vm_assignments:
                print("Error: unassigned vm " + vm)
            assert (vm in vm_assignments)

        changed = dict()
        additional_flow = dict()
        for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
            changed[i] = 0
            additional_flow[(s1, s2)] = 0
            additional_flow[(s2, s1)] = 0
        directed_flow_nets = dict()

        edge_assigns = dict()
        for i, bvs in enumerate(all_edges):
            edge_assigns[i] = []
            for b in bvs:
                edge_assigns[i].append(b.value())

        for source in vm_instance["VMs"]:
            # for source, flow_requirements in maxflow_requirements.items():

            if source in maxflow_requirements:
                refine_this_flow = refine_flow and source in virtual_switches
                flow_requirements = maxflow_requirements[source]
                if len(flow_requirements) == 0:
                    continue;

                (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, dests,
                 required_flow) = getFlowRequirement(source)

                flow = physical_graph_copy.getMaxFlow(mf)
                source_dest_flow = flow
                required_flow_val = required_flow.value()
                assert (flow >= required_flow_val)
                assert (required_flow_val > 0)
                assert (flow > 0)

                flow_req_dict = dict()
                for (dest, bandwidth) in flow_requirements:
                    flow_req_dict[dest] = bandwidth

                directed_flow_net = nx.DiGraph()
                for a, b, bw in instance["PN"]:
                    directed_flow_net.add_node(a)
                    directed_flow_net.add_node(b)

                for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
                    efor = copy_edges[i][0]
                    eback = copy_edges[i][1]
                    ffor = physical_graph_copy.getEdgeFlow(mf, efor, remove_flow_cycles)
                    fback = physical_graph_copy.getEdgeFlow(mf, eback, remove_flow_cycles) if eback is not None else 0

                    if ffor > 0:
                        if not directed_flow_net.has_edge(s1, s2):
                            directed_flow_net.add_edge(s1, s2, weight=0)
                        directed_flow_net[s1][s2]['weight'] += ffor
                    if fback > 0:
                        if not directed_flow_net.has_edge(s2, s1):
                            directed_flow_net.add_edge(s2, s1, weight=0)
                        directed_flow_net[s2][s1]['weight'] += fback
                    # edge flows in PN can be 0, or less than the total flow, if some of the vms were assigned to the same server.
                    assert (ffor >= 0)
                    assert (fback >= 0)

                    f = ffor - fback
                    if not directed_links:
                        if f < 0:
                            f = -f
                            s1, s2 = s2, s1

                    assert (f >= 0)
                    assert (f <= bandwidth)

                    if f > 0:

                        if not residual_network.has_edge(s1, s2):
                            residual_network.add_edge(s1, s2, weight=0, cost=1)

                        assert (residual_network[s1][s2]['weight'] >= f)
                        residual_network[s1][s2]['weight'] -= f
                        # print("source %s: Allocated %d flow to %s -> %s (%d remains)"%(source,f,s1,s2,residual_network[s1][s2]['weight']))
                        if not directed_links:
                            assert (residual_network[s1][s2]['weight'] == residual_network[s2][s1]['weight'])

                        if residual_network[s1][s2]['weight'] == 0:
                            residual_network.remove_edge(s1, s2)
                # printDiGraph(directed_flow_net)
                directed_flow_nets[source] = directed_flow_net
                if remove_flow_cycles and debug:
                    if (not is_directed_acyclic_graph(directed_flow_net)):
                        printDiGraph(directed_flow_net)
                        assert (False)

        for source in vm_instance["VMs"]:
            if source in maxflow_requirements:
                refine_this_flow = refine_flow and source in virtual_switches
                flow_requirements = maxflow_requirements[source]
                if len(flow_requirements) == 0:
                    continue;
                (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf, dests,
                 required_flow) = flow_requirement_lits[source]
                flow = physical_graph_copy.getMaxFlow(mf)
                source_dest_flow = flow
                assert (flow > 0)

                # this can be improved
                flow_req_dict = dict()
                for (dest, bandwidth) in flow_requirements:
                    flow_req_dict[dest] = bandwidth

                directed_flow_net = directed_flow_nets[source]

                # before any refinement can happen, ALL flows from all flow sources must be removed from the residual network.
                if refine_this_flow:
                    assert (force_virtual_switches)
                    assert (source in virtual_switches)
                    # Starting with the _RESIDUAL_ network
                    # find shortest paths between connections.
                    # while such a path exists, move flow from its old path (which went through source) to the new one.
                    # this is always safe to do, and is guaranteed not to remove the connections that the solver found (as we always operate on the residual network).
                    # this can be short-cut for vm's assigned to the same server.
                    print("Refinining...")
                    # printGraph(flow_network)


                    # Should only consider the flows req that this maxflow is covering... especially for virtual switches.
                    # generalize this later...
                    extra_node = "extra_node"
                    extra_dest_node = "extra_dest_node"
                    extra_source_node = "extra_source_node"
                    directed_flow_net.add_node(extra_source_node)
                    directed_flow_net.add_node(extra_node)
                    directed_flow_net.add_node(extra_dest_node)

                    outer_edge = directed_flow_net.add_edge(extra_node, vm_assignments[source], weight=0)
                    for vm1 in refinement_maxflow_requirements[source]:
                        for vm2, bandwidth in refinement_maxflow_requirements[source][vm1]:
                            s1 = vm_assignments[vm1]
                            s2 = vm_assignments[vm2]
                            sVirt = vm_assignments[source]
                            found_path = []
                            if s1 == s2:
                                if sVirt != s1:
                                    directed_flow_net[extra_node][sVirt]["weight"] = bandwidth
                                    # (that flow had to pass through the virtual switch)
                                    maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(
                                        directed_flow_net, extra_node, s1, capacity="weight")
                                    # note: this _requires_ networkx 1.9+; the old ford_fulkerson code in networkx 1.8 had a subtly different interface that will break the code below!
                                    assert (maxflow == bandwidth)
                                    # remove that flow from the returned flow-net
                                    for (residual_path, pathflow) in getFlowPaths(edge_flows, sVirt, s1, bandwidth,
                                                                                  capacity='weight'):
                                        for (u, v) in pairwise(residual_path):
                                            if u != extra_node and v != extra_node:
                                                assert (pathflow > 0)
                                                assert (directed_flow_net[u][v]['weight'] >= pathflow)
                                                directed_flow_net[u][v]['weight'] -= pathflow
                                                if directed_flow_net[u][v]['weight'] == 0:
                                                    directed_flow_net.remove_edge(u, v)
                                                if not residual_network.has_edge(u, v):
                                                    residual_network.add_edge(u, v, weight=0, cost=1)
                                                residual_network[u][v]['weight'] += pathflow

                                if sVirt != s2:
                                    directed_flow_net[extra_node][sVirt]["weight"] = bandwidth
                                    # (that flow had to pass through the virtual switch)
                                    maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(
                                        directed_flow_net, extra_node, s2, capacity="weight")
                                    # note: this _requires_ networkx 1.9+; the old ford_fulkerson code in networkx 1.8 had a subtly different interface that will break the code below!
                                    assert (maxflow == bandwidth)
                                    # remove that flow from the returned flow-net
                                    # print(edge_flows)

                                    # need to decompose into (simple) flow paths to avoid cycles here...
                                    for (residual_path, pathflow) in getFlowPaths(edge_flows, sVirt, s2, bandwidth,
                                                                                  capacity='weight'):
                                        for (u, v) in pairwise(residual_path):
                                            if u != extra_node and v != extra_node:
                                                assert (pathflow > 0)
                                                assert (directed_flow_net[u][v]['weight'] >= pathflow)
                                                directed_flow_net[u][v]['weight'] -= pathflow
                                                if directed_flow_net[u][v]['weight'] == 0:
                                                    directed_flow_net.remove_edge(u, v)
                                                if not residual_network.has_edge(u, v):
                                                    residual_network.add_edge(u, v, weight=0, cost=1)
                                                residual_network[u][v]['weight'] += pathflow

                    # printGraph(directed_flow_net)
                    for vm1 in refinement_maxflow_requirements[source]:
                        for vm2, bandwidth in refinement_maxflow_requirements[source][vm1]:

                            s1 = vm_assignments[vm1]
                            s2 = vm_assignments[vm2]
                            sVirt = vm_assignments[source]

                            if s1 != s2:  # and s1 != sVirt and s2 != sVirt:
                                # printGraph(directed_flow_net)
                                directed_flow_net.add_edge(s1, extra_dest_node, weight=bandwidth)
                                directed_flow_net.add_edge(s2, extra_dest_node, weight=bandwidth)
                                maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(directed_flow_net,
                                                                                                    sVirt,
                                                                                                    extra_dest_node,
                                                                                                    capacity="weight")
                                # note: this _requires_ networkx 1.9+; the old ford_fulkerson code in networkx 1.8 had a subtly different interface that will break the code below!
                                if (maxflow != bandwidth * 2):
                                    printDiGraph(directed_flow_net)
                                    print("%s %s" % (s1, s2));
                                assert (maxflow == bandwidth * 2)
                                # remove that flow from the returned flow-net
                                # print(edge_flows)

                                for (residual_path, flow) in getFlowPaths(edge_flows, sVirt, extra_dest_node,
                                                                          bandwidth * 2, capacity='weight'):
                                    for (s, t) in pairwise(residual_path):
                                        if s != extra_dest_node and t != extra_dest_node:
                                            assert (flow > 0)
                                            assert (directed_flow_net[s][t]['weight'] >= flow)
                                            directed_flow_net[s][t]['weight'] -= flow
                                            if directed_flow_net[s][t]['weight'] == 0:
                                                directed_flow_net.remove_edge(s, t)
                                            # if not directed_flow_net.has_edge(t,s):
                                            #    directed_flow_net.add_edge(t,s,weight=0)
                                            # directed_flow_net[t][s]['weight']+=flow

                                            if not residual_network.has_edge(s, t):
                                                residual_network.add_edge(s, t, weight=0, cost=1)
                                            # print(str(residual_network[s][t]['weight']))
                                            residual_network[s][t]['weight'] += flow

                                # Now that we have removed the current flow from the flow graph and added it to the residual graph, a valid s1 s2 flow of bandwidth is guaranteed to exist in bandwidth.
                                # use min_cost_max_flow to find the shortest such flow (which, in the worst case, is the same one we just added to the residual graph)
                                directed_flow_net.remove_edge(s1, extra_dest_node)
                                directed_flow_net.remove_edge(s2, extra_dest_node)

                                if not use_cover_for_refinement:

                                    s1 = vm_assignments[vm1]
                                    s2 = vm_assignments[vm2]
                                    sVirt = vm_assignments[source]
                                    assert (s1 != s2)
                                    residual_network.add_edge(s2, extra_dest_node, weight=bandwidth, cost=0)
                                    if use_min_cost_max_flow:
                                        maxflow, edge_flows = networkx.algorithms.flow.mincost.max_flow_min_cost(
                                            residual_network, s1, extra_dest_node, capacity="weight", weight="cost")
                                    else:
                                        maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(
                                            residual_network, s1, extra_dest_node, capacity="weight")

                                    residual_network.remove_edge(s2, extra_dest_node)

                                    assert (maxflow == bandwidth)
                                    # max_flow_min_cost
                                    # need to decompose into (simple) flow paths to avoid cycles here...
                                    for (residual_path, pathflow) in getFlowPaths(edge_flows, s1, s2, bandwidth,
                                                                                  capacity='weight'):
                                        for (u, v) in pairwise(residual_path):
                                            if u != extra_dest_node and v != extra_dest_node:
                                                additional_flow[(v, u)] += pathflow
                                                if not directed_links:
                                                    additional_flow[(u, v)] += pathflow
                                                # print(str(residual_network[u][v]['weight']))
                                                residual_network[u][v]["weight"] -= pathflow
                                                # print(str(residual_network[u][v]['weight']))
                                                if residual_network[u][v]["weight"] == 0:
                                                    residual_network.remove_edge(u, v)
                                        if log:
                                            log.write("path %s -> %s %d =" % (vm1, vm2, pathflow))
                                            for u in residual_path:
                                                if u != extra_dest_node:
                                                    log.write(" " + u)
                                            log.write("\n")

                        if use_cover_for_refinement:
                            assert (len(residual_network.edges(extra_dest_node)) == 0)
                            sum_flow = 0
                            s1 = vm_assignments[vm1]
                            # print(vm1)
                            for vm2, bandwidth in refinement_maxflow_requirements[source][vm1]:

                                s2 = vm_assignments[vm2]
                                if s1 != s2:
                                    sum_flow += bandwidth
                                    if not residual_network.has_edge(s2, extra_dest_node):
                                        residual_network.add_edge(s2, extra_dest_node, weight=0, cost=0)
                                    residual_network[s2][extra_dest_node]["weight"] += bandwidth
                            if sum_flow == 0:
                                continue
                            if use_min_cost_max_flow:
                                maxflow, edge_flows = networkx.algorithms.flow.mincost.max_flow_min_cost(
                                    residual_network, s1, extra_dest_node, capacity="weight", weight="cost")
                            else:
                                maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(residual_network,
                                                                                                    s1, extra_dest_node,
                                                                                                    capacity="weight")

                            # printGraph(residual_network)
                            # printGraph(residual_network)

                            assert (maxflow == sum_flow)
                            # max_flow_min_cost
                            # need to decompose into (simple) flow paths to avoid cycles here...
                            vm1_flow_reqs = dict()
                            for vm2, bandwidth in refinement_maxflow_requirements[source][vm1]:
                                vm1_flow_reqs[vm2] = bandwidth

                            for (residual_path, pathflow) in getFlowPaths(edge_flows, s1, extra_dest_node, sum_flow,
                                                                          capacity='weight'):
                                server = residual_path[-2]

                                for (u, v) in pairwise(residual_path):
                                    if u != extra_dest_node and v != extra_dest_node:
                                        additional_flow[(v, u)] += pathflow
                                        if not directed_links:
                                            additional_flow[(u, v)] += pathflow
                                        if residual_network[u][v]["weight"] < pathflow:
                                            pass
                                        # print(str(residual_network[u][v]['weight']))
                                        assert (residual_network.has_edge(u, v))
                                        assert (residual_network[u][v]["weight"] >= pathflow)
                                        residual_network[u][v]["weight"] -= pathflow
                                        # print(str(residual_network[u][v]['weight']))
                                        if residual_network[u][v]["weight"] == 0:
                                            residual_network.remove_edge(u, v)

                                if log:
                                    # now assign the flow along this path to destination vms as needed.
                                    while pathflow > 0:
                                        found_vm = None
                                        used_flow = 0
                                        for vm2, bandwidth in refinement_maxflow_requirements[source][vm1]:
                                            required_bandwidth = vm1_flow_reqs[vm2]
                                            s2 = vm_assignments[vm2]
                                            if s2 == server and required_bandwidth > 0:
                                                found_vm = vm2
                                                used_flow = min(required_bandwidth, pathflow)
                                                assert (vm1_flow_reqs[vm2] >= used_flow)
                                                vm1_flow_reqs[vm2] -= used_flow
                                                break
                                        assert (pathflow >= used_flow)
                                        pathflow -= used_flow
                                        log.write("path %s -> %s %d =" % (vm1, found_vm, used_flow))
                                        for u in residual_path:
                                            if u != extra_dest_node:
                                                log.write(" " + u)
                                        log.write("\n")
                            for vm2, bandwidth in refinement_maxflow_requirements[source][vm1]:
                                s2 = vm_assignments[vm2]
                                if s1 != s2:
                                    if residual_network.has_edge(s2, extra_dest_node):
                                        residual_network.remove_edge(s2, extra_dest_node)
                            assert (len(residual_network.edges(extra_dest_node)) == 0)
                    directed_flow_net.remove_edge(extra_node, vm_assignments[source])
                    directed_flow_net.remove_node(extra_node)

                for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
                    ffor = directed_flow_net[s1][s2]["weight"] if directed_flow_net.has_edge(s1, s2) else 0
                    if not directed_links:
                        fback = directed_flow_net[s2][s1]["weight"] if directed_flow_net.has_edge(s2, s1) else 0
                    else:
                        fback = 0
                    # f = directed_flow_net[s1][s2]["weight"] if directed_flow_net.has_edge(s1,s2) else 0
                    f = max(ffor, fback) - min(ffor, fback)
                    f += additional_flow[(s1, s2)]
                    additional_flow[(s1, s2)] = 0
                    if not directed_links:
                        additional_flow[(s2, s1)] = 0
                    assert (f >= 0)
                    assert (f <= bandwidth)
                    # total_edge_flow[i]+=f
                    # print(str(f))
                    # total_edge_flow[i]+=f
                    sm, wt = sums[i]
                    assert (wt - f >= 0)
                    if f > 0:
                        sums[i] = (sm, wt - f)
                        changed[i] += f

                # Log the paths out if we haven't already done so.
                if (output_file or log) and not refine_this_flow:

                    for dest, f in flow_requirements:
                        flowreq = flow_req_dict[dest]
                        if vm_assignments[source] != vm_assignments[dest]:

                            for (path, flow) in decomposeIntoSimplePaths(directed_flow_net, vm_assignments[source],
                                                                         vm_assignments[dest], flowreq,
                                                                         capacity='weight'):
                                bandwidth_data.append((getOriginalVMName(source), getOriginalVMName(dest), flow,
                                                       list(path)))
                                if log:
                                    log.write("path %s -> %s %d =" % (source, dest, flow))
                                    for u in path:
                                        log.write(" " + u)
                                    log.write("\n")

        if log:
            log.flush()

        # for i,( s1, s2, bandwidth) in enumerate(instance["PN"]):
        #     if changed[i]!=0:
        #         print("Edge %d (%s->%s = %d)"%(i,s1,s2,bandwidth))
        #
        #         for j,b in enumerate(all_edges[i]):
        #             print("%d"%(edge_assigns[i][j]))
        #         print("<= %d <= %d"%(sums[i][0].value(),sums[i][1]))
        # printGraph(residual_network)
        vdc_allocation_map[n_solutions]["bw"] = dict(changed)
        for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
            if changed[i] != 0:
                f = changed[i]
                sm, wt = sums[i]

                assert (wt >= 0)
                assert (wt <= bandwidth)

                assumplit = (sm <= wt)
                bandwidth_assump_lits[i] = assumplit
                assumptions.append(assumplit)
                if residual_network:
                    expect = residual_network[s1][s2]['weight'] if residual_network.has_edge(s1, s2) else 0
                    if expect != wt:
                        pass

                    assert (expect == wt)

            else:
                assumplit = bandwidth_assump_lits[i]
                assumptions.append(assumplit)

        # Core and ram requirements also need to be reduced to account for assignments.
        vdc_allocation_map[n_solutions]["assigned"] = dict()
        # set the used_cores, etc
        for host in instance["Servers"]:
            # only update the servers that were assigned to; use existing assumption literals for the unchagned ones
            assigned_cores = server_assigned_cores[host]
            assigned_ram = server_assigned_ram[host]
            assigned_storage = server_assigned_storage[host]
            assigned_ext = server_assigned_ext[host]
            vdc_allocation_map[n_solutions]["assigned"][host] = (
            assigned_cores, assigned_ram, assigned_storage, assigned_ext)

            total_core_assignment[host] += assigned_cores
            total_ram_assignment[host] += assigned_ram
            total_storage_assignment[host] += assigned_storage

            for r in range(n_extended_resources):
                total_extended_resource_assignment[host][r] += assigned_ext[r]

            used_cores, used_ram, used_storage, used_ex = server_core_assignments[host][-1], \
                                                          server_ram_assignments[host][-1], \
                                                          server_storage_assignments[host][-1], \
                                                          server_extended_assignments[host][-1]
            for i, v in enumerate(used_cores):
                if (i + 1 == total_core_assignment[host]):
                    assumptions.append(v)
                else:
                    assumptions.append(~v)
            for i, v in enumerate(used_ram):
                if (i + 1 == total_ram_assignment[host]):
                    assumptions.append(v)
                else:
                    assumptions.append(~v)
            for i, v in enumerate(used_storage):
                if (i + 1 == total_storage_assignment[host]):
                    assumptions.append(v)
                else:
                    assumptions.append(~v)
            for r, res in enumerate(used_ex):
                for i, v in enumerate(res):
                    if (i < total_extended_resource_assignment[host][r]):
                        assumptions.append(v)
                    else:
                        assumptions.append(~v)

        # load the next VDC
        inst,vdc_name = selectVDC()
        if inst is None:
            break

        vm_instance, maxflow_requirements = loadVN(inst)
        used_vms = set(vm_instance["VMs"].keys())
        unused_vms = set(vms).difference(used_vms)
        for vm in used_vms:
            cores, ram, storage, aggregatebandwidth = vm_instance["VMs"][vm]
            assumptions.append(~unused_vm_edges[vm])  # force the vm to be assigned to a server
            assert (cores <= max_cores)
            assert (ram <= max_ram)
            assert (storage <= max_storage)

            for i in range(max_cores):
                v = vmcores[vm][i]
                if i + 1 == cores:
                    assumptions.append(v)
                else:
                    assumptions.append(~v)
            if not ignore_ram_constraints:
                for i in range(max_ram):
                    v = vmram[vm][i]
                    if i + 1 == ram:
                        assumptions.append(v)
                    else:
                        assumptions.append(~v)
            if not ignore_storage_constraints:
                for i in range(max_storage):
                    v = vmstorage[vm][i]
                    if i + 1 == storage:
                        assumptions.append(v)
                    else:
                        assumptions.append(~v)

        for vm in unused_vms:
            assumptions.append(unused_vm_edges[vm])  # prevent the vm from being assigned to a host
            for i in range(max_cores):
                v = vmcores[vm][i]
                assumptions.append(~v)
            if not ignore_ram_constraints:
                for i in range(max_ram):
                    v = vmram[vm][i]
                    assumptions.append(~v)
            if not ignore_storage_constraints:
                for i in range(max_storage):
                    v = vmstorage[vm][i]
                    assumptions.append(~v)
        unused_sources = set(vms)

        for source in vms:
            # for source, flow_requirements in maxflow_requirements.items():
            if source in maxflow_requirements:
                flow_requirements = maxflow_requirements[source]
                if len(flow_requirements) == 0:
                    continue;
                sum_flow = 0
                for dest, bandwidth in flow_requirements:
                    sum_flow += bandwidth

                (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf,
                 potential_dests, sum_bandwidth) = getFlowRequirement(source, True, True)

                assert (potential_dests is not None)

                non_dests = set(vms)
                non_dests.remove(source)
                for dest, bandwidth in flow_requirements:
                    assigned_bandwidth = potential_dests[dest]
                    assumptions.append(assigned_bandwidth == bandwidth)
                    non_dests.remove(dest)
                for dest in non_dests:
                    assigned_bandwidth = potential_dests[dest]
                    assumptions.append(assigned_bandwidth <= 0)
                # print("Setting sum flow for graph  for source " + str(source) + " to " + str(sum_flow))
                unused_sources.remove(source)
                assumptions.append(sum_bandwidth == sum_flow)

        for source in unused_sources:
            if not only_instantiate_networks_on_demand:
                # pass
                # disable this maxflow requirement

                (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf,
                 potential_dests, sum_bandwidth) = getFlowRequirement(source)
                if sum_bandwidth is not None:
                    assumptions.append(
                        sum_bandwidth <= 0)  # since maximum flow is always greater or equal to 0 in any graph,
                    # print("Setting sum flow for graph  for source " + str(source) + " to 0")
                # the maxflow>=sum_bandwidth constraint becomes a tautology is sum_bandwidth is <=0, and ideally can be optimized
                # away by the solver.
                if potential_dests is not None:
                    for vm, bandwidth in potential_dests.items():
                        assumptions.append(bandwidth <= 0)
                if internal_edge_bandwidths_limits is not None:
                    for bandwidth_zero in internal_edge_bandwidths_limits:
                        assumptions.append(bandwidth_zero)

        if min_cores:
            for host in instance["Servers"]:
                n_used = total_core_assignment[host]
                assumptions.append(used_server_cores[host] == n_used)

        if min_servers:
            for s in instance["Servers"]:
                used = used_servers[s]
                if s in used_hosts:
                    assumptions.append(used)
                else:
                    assumptions.append(Not(used))
        assumptions.extend(assume_false)
        if log:
            log.flush()
        if output_file:
            with open(output_file, 'w') as outfile:
                json.dump(json_data, outfile)

        # print("}")

        t = time.clock()
        elapsed_out += t - et
        if log_time_file:
            timefile.write("%d %f %f\n" % (n_solutions, t - prev_time, t - start_time))
            timefile.flush()
        prev_time = t

        print("Solving..")

    if (args.deallocate_prob == 0):
        break
    print("Datacenter is fully allocated; de-allocating %f percentage of instances" % (args.deallocate_prob))

    assumptions.clear();
    # pick a random, assigned vdc to remove
    n_remove = math.ceil(0.1 * len(allocated_vdcs))
    r_list = list(allocated_vdcs)
    random.shuffle(r_list)
    vdcs_to_remove = r_list[:n_remove]

    for to_remove in vdcs_to_remove:
        allocated_vdcs.remove(to_remove)
        print("*****Deallocating vdc %d*****" % (to_remove))
        n_solutions -= 1

        allocated_bw = vdc_allocation_map[to_remove]["bw"]

        for i, (s1, s2, bandwidth) in enumerate(instance["PN"]):
            if i in allocated_bw and allocated_bw[i] is not None and allocated_bw[i] != 0:
                f = allocated_bw[i]
                assert (f > 0);
                sm, wt = sums[i]
                assert (wt >= 0)
                assert (wt <= bandwidth)

                assumplit = (sm <= wt)
                bandwidth_assump_lits[i] = assumplit
                assumptions.append(assumplit)
                sums[i] = (sm, wt + f)

                assert (sums[i][1] <= bandwidth)

                if not residual_network.has_edge(s1, s2):
                    residual_network.add_edge(s1, s2, weight=0, cost=1)

                residual_network[s1][s2]['weight'] += f
                assert (residual_network[s1][s2]['weight'] <= bandwidth)
                if not directed_links:
                    assert (residual_network[s1][s2]['weight'] == residual_network[s2][s1]['weight'])

                if residual_network[s1][s2]['weight'] == 0:
                    residual_network.remove_edge(s1, s2)

            else:
                assumplit = bandwidth_assump_lits[i]
                assumptions.append(assumplit)

        for host in instance["Servers"]:
            (assigned_cores, assigned_ram, assigned_storage, assigned_ext) = vdc_allocation_map[to_remove]["assigned"][
                host]
            total_core_assignment[host] -= assigned_cores
            total_ram_assignment[host] -= assigned_ram
            total_storage_assignment[host] -= assigned_storage

            for r in range(n_extended_resources):
                total_extended_resource_assignment[host][r] -= assigned_ext[r]
        vdc_allocation_map[to_remove] = None

        if log_time_file:
            t = time.clock()
            timefile.write("%d %f %f\n" % (-1, t - prev_time, t - start_time))
            timefile.flush()
            prev_time = t

    # attempt to re-allocate the next instance
    for host in instance["Servers"]:
        # only update the servers that were assigned to; use existing assumption literals for the unchagned ones
        used_cores, used_ram, used_storage, used_ex = server_core_assignments[host][-1], server_ram_assignments[host][
            -1], server_storage_assignments[host][-1], server_extended_assignments[host][-1]
        for i, v in enumerate(used_cores):
            if (i + 1 == total_core_assignment[host]):
                assumptions.append(v)
            else:
                assumptions.append(~v)
        for i, v in enumerate(used_ram):
            if (i + 1 == total_ram_assignment[host]):
                assumptions.append(v)
            else:
                assumptions.append(~v)
        for i, v in enumerate(used_storage):
            if (i + 1 == total_storage_assignment[host]):
                assumptions.append(v)
            else:
                assumptions.append(~v)
        for r, res in enumerate(used_ex):
            for i, v in enumerate(res):
                if (i < total_extended_resource_assignment[host][r]):
                    assumptions.append(v)
                else:
                    assumptions.append(~v)

    vm_instance, maxflow_requirements = loadVN(inst)
    used_vms = set(vm_instance["VMs"].keys())
    unused_vms = set(vms).difference(used_vms)
    for vm in used_vms:
        cores, ram, storage, aggregatebandwidth = vm_instance["VMs"][vm]
        assumptions.append(~unused_vm_edges[vm])  # force the vm to be assigned to a server
        assert (cores <= max_cores)
        assert (ram <= max_ram)
        assert (storage <= max_storage)

        for i in range(max_cores):
            v = vmcores[vm][i]
            if i + 1 == cores:
                assumptions.append(v)
            else:
                assumptions.append(~v)
        if not ignore_ram_constraints:
            for i in range(max_ram):
                v = vmram[vm][i]
                if i + 1 == ram:
                    assumptions.append(v)
                else:
                    assumptions.append(~v)
        if not ignore_storage_constraints:
            for i in range(max_storage):
                v = vmstorage[vm][i]
                if i + 1 == storage:
                    assumptions.append(v)
                else:
                    assumptions.append(~v)

    for vm in unused_vms:
        assumptions.append(unused_vm_edges[vm])  # prevent the vm from being assigned to a host
        for i in range(max_cores):
            v = vmcores[vm][i]
            assumptions.append(~v)
        if not ignore_ram_constraints:
            for i in range(max_ram):
                v = vmram[vm][i]
                assumptions.append(~v)
        if not ignore_storage_constraints:
            for i in range(max_storage):
                v = vmstorage[vm][i]
                assumptions.append(~v)
    unused_sources = set(vms)

    for source in vms:
        # for source, flow_requirements in maxflow_requirements.items():
        if source in maxflow_requirements:
            flow_requirements = maxflow_requirements[source]
            if len(flow_requirements) == 0:
                continue;
            sum_flow = 0
            for dest, bandwidth in flow_requirements:
                sum_flow += bandwidth

            (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf,
             potential_dests, sum_bandwidth) = getFlowRequirement(source, True, True)

            assert (potential_dests is not None)

            non_dests = set(vms)
            non_dests.remove(source)
            for dest, bandwidth in flow_requirements:
                assigned_bandwidth = potential_dests[dest]
                assumptions.append(assigned_bandwidth == bandwidth)
                non_dests.remove(dest)
            for dest in non_dests:
                assigned_bandwidth = potential_dests[dest]
                assumptions.append(assigned_bandwidth <= 0)
            # print("Setting sum flow for graph  for source " + str(source) + " to " + str(sum_flow))
            unused_sources.remove(source)
            assumptions.append(sum_bandwidth == sum_flow)

    for source in unused_sources:
        if not only_instantiate_networks_on_demand:
            # pass
            # disable this maxflow requirement

            (physical_graph_copy, internal_edge_bandwidths, internal_edge_bandwidths_limits, copy_edges, mf,
             potential_dests, sum_bandwidth) = getFlowRequirement(source)
            if sum_bandwidth is not None:
                assumptions.append(
                    sum_bandwidth <= 0)  # since maximum flow is always greater or equal to 0 in any graph,
                # print("Setting sum flow for graph  for source " + str(source) + " to 0")
            # the maxflow>=sum_bandwidth constraint becomes a tautology is sum_bandwidth is <=0, and ideally can be optimized
            # away by the solver.
            if potential_dests is not None:
                for vm, bandwidth in potential_dests.items():
                    assumptions.append(bandwidth <= 0)
            if internal_edge_bandwidths_limits is not None:
                for bandwidth_zero in internal_edge_bandwidths_limits:
                    assumptions.append(bandwidth_zero)

    if min_cores:
        for host in instance["Servers"]:
            n_used = total_core_assignment[host]
            assumptions.append(used_server_cores[host] == n_used)

    if min_servers:
        for s in instance["Servers"]:
            used = used_servers[s]
            if s in used_hosts:
                assumptions.append(used)
            else:
                assumptions.append(Not(used))
    assumptions.extend(assume_false)
    if log:
        log.flush()
    if log_time_file:
        timefile.flush()

    # print("}")

    t = time.clock()

    prev_time = t


    # os.system("callgrind_control -i on")
total_time = time.clock() - start_time

# shutil.copy(filename,filename[:-3]+str(n_solutions+1) + ".gnf")



if log:
    log.close()
if output_file:
    with open(output_file, 'w') as outfile:
        json.dump(json_data, outfile)
if log_time_file:
    end_time = time.clock()
    timefile.write("done %d %f %f\n" % (n_solutions, end_time - prev_time, end_time - start_time))
    timefile.close()
print("Elapsed time: " + str(total_time) + " solving time: " + str(elapsed) + ", " + str(elapsed_out) + ", " + str(
    Monosat().elapsed_time) + ", " + str(PBManager().elapsed_time) + " (" + str(PBManager().import_time) + ")")
print("Solved %d" % (n_solutions))
print("Done")


