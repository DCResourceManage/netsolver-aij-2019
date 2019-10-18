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

#os.system("callgrind_control -i off")
bv=BitVector

if "PYTHONHASHSEED" not in os.environ or os.environ['PYTHONHASHSEED']!="1":
    print("Python 3.4+ randomizes dictionary hashes by default (using PYTHONHASHSEED as its random seed).\n Set PYTHONHASHSEED=1 in the runtime environment (e.g., '$export PYTHONHASHSEED=1') in order to reproduce experiments (currently, PYTHONHASHSEED is %s)."%(os.environ['PYTHONHASHSEED'] if 'PYTHONHASHSEED' in os.environ else "UNSET" ))
    sys.exit(1)
if __name__ == "__main__":
    from os import path
    import os
    import sys
    #Setup PYTHONPATH... if anyone knows a better way to do this without a shell script, I'm all ears..
    sys.path.append(os.path.abspath(os.path.join( path.dirname(__file__),os.pardir)))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def printGraph(G,key="weight"):
    print("graph{")
    for (s1,s2,data) in G.edges_iter(data=true):   
        print(s1 + " -- " + s2 + ("[xlabel=\"%d Mbps\"]"%(data[key]) if key is not None else "") )
    print("}")


def printDiGraph(G,key="weight"):
    print("digraph{")
    for (s1,s2,data) in G.edges_iter(data=true):   
        print(s1 + " -> " + s2 + "[xlabel=\"%d Mbps\"]"%(data[key]))
    print("}")
    
start_time=time.clock()

parser = argparse.ArgumentParser(description='VDCMapper')


parser.add_argument("--logfile",type=str,help="Output filename",default=None)

parser.add_argument("--timelog",type=str,help="Output filename",default=None)


parser.add_argument('--static_flow_caps',dest='static_flow_caps',help="",action='store_true')
parser.add_argument('--no-static_flow_caps',dest='static_flow_caps',help="",action='store_false')
parser.set_defaults(static_flow_caps=True)

parser.add_argument('--tree_addition',dest='tree_addition',help="Build multi-argument addition constraints in a tree, instead of linear chain",action='store_true')
parser.add_argument('--no-tree_addition',dest='tree_addition',help="",action='store_false')
parser.set_defaults(tree_addition=True)

parser.add_argument('--bitblast_addition',dest='bitblast_addition',help="",action='store_true')
parser.add_argument('--no-bitblast_addition',dest='bitblast_addition',help="",action='store_false')
parser.set_defaults(bitblast_addition=False)


parser.add_argument('--bitblast_addition_shadow',dest='bitblast_addition_shadow',help="",action='store_true')
parser.add_argument('--no-bitblast_addition_shadow',dest='bitblast_addition_shadow',help="",action='store_false')
parser.set_defaults(bitblast_addition_shadow=False)

parser.add_argument('--separate_reach_constraints',dest='separate_reach_constraints',help="",action='store_true')
parser.add_argument('--no-separate_reach_constraints',dest='separate_reach_constraints',help="",action='store_false')
parser.set_defaults(separate_reach_constraints=False)

parser.add_argument('--assert_all_physical_edges',dest='assert_all_physical_edges',help="",action='store_true')
parser.add_argument('--no-assert_all_physical_edges',dest='assert_all_physical_edges',help="",action='store_false')
parser.set_defaults(assert_all_physical_edges=True)

parser.add_argument('--ignore_ram_constraints',dest='ignore_ram_constraints',help="",action='store_true')
parser.add_argument('--no-ignore_ram_constraints',dest='ignore_ram_constraints',help="",action='store_false')
parser.set_defaults(ignore_ram_constraints=False)

parser.add_argument('--ignore_storage_constraints',dest='ignore_storage_constraints',help="",action='store_true')
parser.add_argument('--no-ignore_storage_constraints',dest='ignore_storage_constraints',help="",action='store_false')
parser.set_defaults(ignore_storage_constraints=False)

parser.add_argument('--remove_common_bandwidth_factor',dest='remove_common_bandwidth_factor',help="",action='store_true')
parser.add_argument('--no-remove_common_bandwidth_factor',dest='remove_common_bandwidth_factor',help="",action='store_false')
parser.set_defaults(remove_common_bandwidth_factor=True)

parser.add_argument('--force_virtual_switches',dest='force_virtual_switches',help="",action='store_true')
parser.add_argument('--no-force_virtual_switches',dest='force_virtual_switches',help="",action='store_false')
parser.set_defaults(force_virtual_switches=False)

parser.add_argument('--allow_virtual_switches_in_switches',dest='allow_virtual_switches_in_switches',help="",action='store_true')
parser.add_argument('--no-allow_virtual_switches_in_switches',dest='allow_virtual_switches_in_switches',help="",action='store_false')
parser.set_defaults(allow_virtual_switches_in_switches=True)

parser.add_argument('--intrinsic_edge_constraints',dest='intrinsic_edge_constraints',help="",action='store_true')
parser.add_argument('--no-intrinsic_edge_constraints',dest='intrinsic_edge_constraints',help="",action='store_false')
parser.set_defaults(intrinsic_edge_constraints=False)

parser.add_argument('--intrinsic_edge_sets',dest='intrinsic_edge_sets',help="",action='store_true')
parser.add_argument('--no-intrinsic_edge_sets',dest='intrinsic_edge_sets',help="",action='store_false')
parser.set_defaults(intrinsic_edge_sets=True)

parser.add_argument('--flush_pb_theory',dest='flush_pb_theory',help="",action='store_true')
parser.add_argument('--no-flush_pb_theory',dest='flush_pb_theory',help="",action='store_false')
parser.set_defaults(flush_pb_theory=True)

parser.add_argument('--refine_flow',dest='refine_flow',help="",action='store_true')
parser.add_argument('--no-refine_flow',dest='refine_flow',help="",action='store_false')
parser.set_defaults(refine_flow=True)

parser.add_argument('--remove_flow_cycles',dest='remove_flow_cycles',help="",action='store_true')
parser.add_argument('--no-remove_flow_cycles',dest='remove_flow_cycles',help="",action='store_false')
parser.set_defaults(remove_flow_cycles=False)

parser.add_argument('--remove_flow_cycles_python',dest='remove_flow_cycles_python',help="",action='store_true')
parser.add_argument('--no-remove_flow_cycles_python',dest='remove_flow_cycles_python',help="",action='store_false')
parser.set_defaults(remove_flow_cycles_python=False)


parser.add_argument('--remove_all_flow_cycles',dest='remove_all_flow_cycles',help="",action='store_true')
parser.add_argument('--no-remove_all_flow_cycles',dest='remove_all_flow_cycles',help="",action='store_false')
parser.set_defaults(remove_all_flow_cycles=False)


parser.add_argument('--use_min_cost_max_flow',dest='use_min_cost_max_flow',help="",action='store_true')
parser.add_argument('--no-use_min_cost_max_flow',dest='use_min_cost_max_flow',help="",action='store_false')
parser.set_defaults(use_min_cost_max_flow=False)

parser.add_argument('--use_cover_for_refinement',dest='use_cover_for_refinement',help="",action='store_true')
parser.add_argument('--no-use_cover_for_refinement',dest='use_cover_for_refinement',help="",action='store_false')
parser.set_defaults(use_cover_for_refinement=True)


parser.add_argument('--directed',dest ='directed_links',help='Assume data-center capacities are available separately in both directions',action='store_true')
parser.add_argument('--no-directed',dest ='directed_links',help='Assume data-center capacities are shared beween both directions',action='store_false')
parser.set_defaults(directed_links=False)

parser.add_argument('--debug',dest='debug',help="",action='store_true')
parser.add_argument('--no-debug',dest='debug',help="",action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--rnd-theory-freq',dest='rnd_theory_freq',type=float,help="",default=0.99)

parser.add_argument('--rnd-theory-order',dest='rnd_theory_order',help="",action='store_true')
parser.add_argument('--no-rnd-theory-order',dest='rnd_theory_order',help="",action='store_false')
parser.set_defaults(rnd_theory_order=False)

parser.add_argument('--theory-order-vsids',dest='theory_order_vsids',help="",action='store_true')
parser.add_argument('--no-theory-order-vsids',dest='theory_order_vsids',help="",action='store_false')
parser.set_defaults(theory_order_vsids=True)

parser.add_argument('--decide-opt-lits',dest='decide_opt_lits',help="",action='store_true')
parser.add_argument('--no-decide-opt-lits',dest='decide_opt_lits',help="",action='store_false')
parser.set_defaults(decide_opt_lits=True)

parser.add_argument('--binary-search',dest='binary_search',help="",action='store_true')
parser.add_argument('--no-binary-search',dest='binary_search',help="",action='store_false')
parser.set_defaults(binary_search=True)

parser.add_argument('--vsids-both',dest='vsids_both',help="",action='store_true')
parser.add_argument('--no-vsids-both',dest='vsids_both',help="",action='store_false')
parser.set_defaults(vsids_both=True)

parser.add_argument('--vsids-balance',dest='vsids_balance',type=float,help="",default=1)

parser.add_argument('--popcount',dest='popcount',type=str,help="",default="BV")

parser.add_argument('--optimization-conflict-limit',dest='opt_conflict_limit',type=int,help="",default=0)
parser.add_argument('--conflict-limit',dest='conflict_limit',type=int,help="",default=0)

parser.add_argument('--optimization-time-limit',dest='opt_time_limit',type=int,help="",default=-1)



parser.add_argument('--affinity',dest='affinity',help="",action='store_true')
parser.add_argument('--no-affinity',dest='affinity',help="",action='store_false')
parser.set_defaults(affinity=True)

parser.add_argument('--soft-affinity',dest='affinity_soft',help="",action='store_true')
parser.add_argument('--no-soft-affinity',dest='affinity_soft',help="",action='store_false')
parser.set_defaults(affinity_soft=True)

parser.add_argument('--anti-affinity',dest='anti_affinity',help="",action='store_true')
parser.add_argument('--no-anti-affinity',dest='anti_affinity',help="",action='store_false')
parser.set_defaults(anti_affinity=True)

parser.add_argument('--min-servers',dest='min_servers',help="",action='store_true')
parser.add_argument('--no-min-servers',dest='min_servers',help="",action='store_false')
parser.set_defaults(min_servers=False)

parser.add_argument('--min-cores',dest='min_cores',help="",action='store_true')
parser.add_argument('--no-min-cores',dest='min_cores',help="",action='store_false')
parser.set_defaults(min_cores=False)

parser.add_argument('PN', metavar='PN', type=str,  
                   help='Physical data center to allocate the virtual data center to')


parser.add_argument('VDC', metavar='VDC', type=str,
                   help='Virtual data center to allocate (as many copies as possible will be allocated)')



args = parser.parse_args()   

print("Settings: " + str(args))

log_file=args.logfile

log_time_file=args.timelog

static_flow_caps= args.static_flow_caps
tree_addition=args.tree_addition
bitblast_addition=args.bitblast_addition
bitblast_addition_shadow=args.bitblast_addition_shadow
separate_reach_constraints=args.separate_reach_constraints
assert_all_physical_edges=args.assert_all_physical_edges
#remove_redundant_ram_cpu_constraints=True
ignore_ram_constraints=args.ignore_ram_constraints
ignore_storage_constraints=args.ignore_storage_constraints

remove_common_bandwidth_factor=args.remove_common_bandwidth_factor
force_virtual_switches=args.force_virtual_switches
allow_virtual_switches_in_switches=args.allow_virtual_switches_in_switches
intrinsic_edge_constraints=args.intrinsic_edge_constraints
intrinsic_edge_sets=args.intrinsic_edge_sets
flush_pb_theory=args.flush_pb_theory
remove_flow_cycles=args.remove_flow_cycles
remove_all_flow_cycles=args.remove_all_flow_cycles
refine_flow=args.refine_flow
use_min_cost_max_flow= args.use_min_cost_max_flow
use_cover_for_refinement=args.use_cover_for_refinement
remove_flow_cycles_python=args.remove_flow_cycles_python
directed_links=args.directed_links
anti_affinity = args.anti_affinity
min_servers= args.min_servers
decide_opt_lits=args.decide_opt_lits
min_cores = args.min_cores
vsids_both = args.vsids_both
vsids_balance=args.vsids_balance
popcount=args.popcount
if not force_virtual_switches:
    refine_flow=False

theory_order_vsids = args.theory_order_vsids
debug = args.debug
rnd_theory_freq = args.rnd_theory_freq
print("Setting random decision freq to %f"%(rnd_theory_freq))
monosat_settings="-opt-conflict-limit=%d -opt-time-limit=%d -verb=0 %s %s %s -rnd-theory-freq=%f -vsids-balance=%f -verb-time=0 -no-decide-bv-intrinsic  -decide-bv-bitwise  -decide-graph-bv -decide-theories -no-decide-graph-rnd   -lazy-maxflow-decisions -no-conflict-min-cut -conflict-min-cut-maxflow -reach-underapprox-cnf -check-solution "%(args.opt_conflict_limit,args.opt_time_limit, "-theory-order-vsids" if theory_order_vsids else "-no-theory-order-vsids", "-decide-opt-lits" if decide_opt_lits else "-no-decide-opt-lits","-vsids-both" if vsids_both else "-no-vsids-both",  rnd_theory_freq,vsids_balance)
print("Monosat Settings: "+monosat_settings)
Monosat().init(monosat_settings)

print("begin encode");

vn_file = args.VDC
pn_file=args.PN

instance= dict()
print("Loading VN...")
vn= json.load(open(vn_file))
print("Loading PN...")
pn = json.load(open(pn_file))
instance["PN"]=sorted(pn["PN"]) #for reproducibility
instance["Servers"]=pn["Servers"]
instance["VN"]=sorted(vn["VN"]) #for reproducibility
instance["VMs"]=vn["VMs"]

instance_name=os.path.basename(vn_file) + "_to_"  + os.path.basename(pn_file)
n_removed_cycles=0
use_cover=True
if directed_links:
    use_cover_for_refinement=False
    use_cover=False
    
    #double up the VDC (since unlike the datacenter, it should be undirected), to match our secondnet experiments
    
    #for vm1,vm2,bw in list(instance["VN"]):
    #    instance["VN"].append([vm2,vm1,bw])

def _getFlowPaths(edge_flows,source,sink,maximum_bandwidth, capacity='capacity'):
    if source==sink:
        yield ([source,sink],maximum_bandwidth)
        return
    flow_graph = nx.DiGraph()
    for s,flows in edge_flows.items():
        for t,flow in flows.items():
            if flow>0:           
                if not flow_graph.has_edge(s,t):
                    flow_graph.add_edge(s,t,weight=flow)
    
    #yield from 
    for item in decomposeIntoSimplePaths(flow_graph,source,sink,maximum_bandwidth,capacity): yield item

def remove_cycle(g,start,end,stack,weight='weight'):
    global n_removed_cycles
    n_removed_cycles+=1    
    max_weight = g[start][end][weight]
    prev = end
    #assert(start in stack)
    #assert(end in stack)
    #assert(stack[-1]==start)
    for p,_ in reversed(stack):
        w =  g[p][prev][weight]
        prev = p
        max_weight = min(max_weight,w)    
        if p == end:
            break
    prev = end
    earliest = end
    assert( max_weight>0)
    for p,_ in reversed(stack):        
        assert(g[p][prev][weight]>=max_weight)
        g[p][prev][weight]-=max_weight 
        if g[p][prev][weight]==0:
            #g.remove_edge(p,prev)
            earliest = p
        prev = p
        if p == end:
            break    
    return earliest

def remove_cycles(g,n,done_set,weight='weight'):
    stack=[]
    seen=set()
    seen.add(n) 
    stack.append((n, g.edges_iter([n],data=True)))
    
    while(len(stack)>0):
        n,edges = stack[-1]
        try:
            (n,c,data) = next(edges)
            if data[weight]>0:
                if c in seen:
                    #cycle
                    #it is safe to not consider c again here, because either the edge (n,c) is removed, or an earlier edge is removed, in which case we will backtrack.
                    backtrack_to = remove_cycle(g,n,c, stack,weight)
                    if backtrack_to!=stack[-1][0]:
                        while(stack[-1][0]!=backtrack_to):
                            seen.remove(stack[-1][0])
                            stack.pop() 
                        continue                   
                else:
                    seen.add(c)          
                    stack.append((c,g.edges_iter([c],data=True)))
                    
        except StopIteration:
            stack.pop()
            seen.remove(n)
            done_set.add(n)

def remove_all_cycles(g, weight='weight'):
    print("Removing cycles...")
    original_removed=n_removed_cycles 
    done_set=set()
    
    for n in g.nodes():
        if n not in done_set:
            remove_cycles(g,n,done_set,weight)
       
    if n_removed_cycles>original_removed:
        #remove any 0-weight edges
        remove=[]
        for (u,v,data) in g.edges(data=True):
            if data[weight]==0:
                remove.append((u,v))
        g.remove_edges_from(remove)
    #printDiGraph(g)
    print("Done removing cycles...")
    if debug:
        assert(is_directed_acyclic_graph(g))
        print ("Removed %d (of %d) cycles"%(n_removed_cycles-original_removed,n_removed_cycles))
#Decompose the edge flow into (acyclic) flows
#iti = 0
def getFlowPaths(edge_flows,source,sink,maximum_bandwidth, capacity='capacity'):
    #global iti
    if source==sink:
        yield ([source,sink],maximum_bandwidth)
        return
    flow_graph = nx.DiGraph()
    for s,flows in edge_flows.items():
        for t,flow in flows.items():
            if flow>0:           
                if not flow_graph.has_edge(s,t):
                    flow_graph.add_edge(s,t,{capacity:flow})
    if source not in flow_graph.nodes() or sink not in flow_graph.nodes():
        
        return
    
    #while there is a cycle, remove it
    #There is probably a better way to do this!
    #Do a simple dfs cycle detection and removal
    #printDiGraph(flow_graph)
    #iti +=1
    #print(iti)
    #if iti==48:
    #    pass
    if remove_all_flow_cycles:
        if debug:
            expected_flow, _ = networkx.algorithms.flow.maxflow.maximum_flow(flow_graph,source,sink,capacity)
        remove_all_cycles(flow_graph,capacity)
        #printDiGraph(flow_graph)
        if debug:
            remaining_flow, _ = networkx.algorithms.flow.maxflow.maximum_flow(flow_graph,source,sink,capacity)
            assert(expected_flow==remaining_flow)
    
    #yield from 
    for item in decomposeIntoSimplePaths(flow_graph,source,sink,maximum_bandwidth,capacity): yield item

#decompose the graph into simple paths from source to sink, ignoring any cycles.
#note: the 'capacity' field will be altered by this process!
def decomposeIntoSimplePaths(graph,source,sink,maximum_bandwidth, capacity='capacity'):
    if source==sink:
        yield ([source,sink],maximum_bandwidth)
        return 
    if debug:
        original_graph = graph.copy()               
    done =False
    total_flow=0
    while(not done and total_flow<maximum_bandwidth):
        try:
            residual_path = nx.shortest_path(graph,source,sink)
            pathflow = min((graph[u][v][capacity] for (u,v) in pairwise(residual_path)))
            pathflow = min(maximum_bandwidth-total_flow,pathflow)   
            total_flow+=pathflow
            assert(pathflow>0) 
        except networkx.exception.NetworkXNoPath:
            #no s1 -- s2 paths exist in the residual network
            if total_flow!=maximum_bandwidth:
                if debug:
                    printDiGraph(original_graph,capacity)
                print("Total flow along flow paths from %s to %s  was %d, but expected %d"%(source,sink,total_flow,maximum_bandwidth), file=sys.stderr)
                sys.exit(3)
            done=True
            break
            #done
             
        for (u,v) in pairwise(residual_path):
            graph[u][v][capacity]-=pathflow
            if graph[u][v][capacity]==0:
                graph.remove_edge(u,v)
        yield((residual_path,pathflow))
                
                
#need to replace dicts and sets with sorted dicts/sets, to make experiments reproducible in Python 3.3+
#pad the servers with extra info if they dont have it (only some of our instances list combined bandwidth)
sorted_servers = OrderedDict()#for reproducibility
for servername in sorted(instance["Servers"]):
    data = instance["Servers"][servername]
    if len(data)==4:
        cores,ram,storage,bandwidth=data
        sorted_servers[servername] = [cores,ram,storage,bandwidth]
    else:
        assert(len(data)==3)
        cores,ram,storage= data
        sorted_servers[servername] = [cores,ram,storage,0]#0 to fill in for missing storage capacity.
instance["Servers"]=sorted_servers

sorted_vms = OrderedDict()#for reproducibility
for vm in sorted(instance["VMs"]):
    data =instance["VMs"][vm]
    if len(data)==4:
        sorted_vms[vm] = instance["VMs"][vm]
    else:
        assert(len(data)==3)
        cores,ram,storage= data
        sorted_vms[vm] =  [cores,ram,storage,0]
instance["VMs"]=sorted_vms


if ignore_ram_constraints:
    print("Warning - ignoring RAM constraints")
print(instance_name)
common_bw_factor=1
if remove_common_bandwidth_factor:
    bandwidths = []
    for u,v,bandwidth in instance["VN"]:
        bandwidths.append(int(bandwidth))
    for s1, s2, bandwidth in instance["PN"]:
        bandwidths.append(int(bandwidth))
    common_bw_factor= reduce(gcd,bandwidths)
    
    if common_bw_factor>1:
        newvn = []
        for u,v,bandwidth in instance["VN"]:
            newvn.append((u,v,bandwidth//common_bw_factor))
        instance["VN"]=newvn

        newpn = []
        for s1, s2, bandwidth in instance["PN"]:
            newpn.append((s1,s2,bandwidth//common_bw_factor))
        instance["PN"]=newpn        
        print("Removed common factor of %d from bandwidths"%(common_bw_factor))
#enforce bandwidth constraints
#to do this, first split the virtual network into source and destination nodes.
    
     
#using networkx to find the cover (so first need to construct the networkx virtual graph):
virt_network = nx.Graph() if not directed_links else nx.DiGraph()
for u,v,bandwidth in instance["VN"]:
    virt_network.add_edge(u,v)
#printGraph(virt_network,None)
virtual_switches=set()
maxflow_requirements= dict()
for n in virt_network:
    maxflow_requirements[n]=[]

original_vn=list(instance["VN"])

refinement_maxflow_requirements=dict()
if not force_virtual_switches:
    #find an approximate vertex cover
    if use_cover:
        cover= min_weighted_vertex_cover(virt_network);
        
        #for each node in the vertex cover, find the set of all edges incident to that node in the virtual network that are not yet covered; 
        #those form a single maxflow query.
        
        
        covered_edges = set()
        for n in cover:
            refinement_maxflow_requirements[n]=dict()
            refinement_maxflow_requirements[n][n]=[]
            for u,v,bandwidth in instance["VN"]:
          
                if (u,v) not in covered_edges and (u == n or v==n):
                    covered_edges.add((u,v))      
                      
                    assert(u==n or v==n)
                    if u==n:
                        dest=v
                    else:
                        dest=u
                    maxflow_requirements[n].append((dest, bandwidth))
                    refinement_maxflow_requirements[n][n].append((dest, bandwidth))
    else:
        for n in virt_network:
            refinement_maxflow_requirements[n]=dict()
            refinement_maxflow_requirements[n][n]=[]
            for u,v,bandwidth in instance["VN"]:    
                if u==n:
                    maxflow_requirements[n].append((v, bandwidth))
                    refinement_maxflow_requirements[n][n].append((v, bandwidth))
            
        
    #if use_cover_for_refinement:
    #    refinement_maxflow_requirements = maxflow_requirements.copy()
  
else:
    components = networkx.algorithms.components.connected_components(virt_network)
    #original_vn=list(instance["VN"])
    if use_cover_for_refinement:
        cover= min_weighted_vertex_cover(virt_network);
    for component in components:
        #introduce a switch, IF the component is not already covered by a single node.
        covering_node=None
        component_nodes = set(component)
        intersect = component_nodes
        for edge in virt_network.edges(component):
            intersect = intersect & (set(edge))
        if len( intersect)>0:
            #intersect can have at most two nodes
            covering_node = next(iter(intersect))
        if covering_node is not None:
            for u,v,bandwidth in instance["VN"]:
                if u==covering_node:
                    maxflow_requirements[covering_node].append((v, bandwidth))
                elif v==covering_node:
                    maxflow_requirements[covering_node].append((u, bandwidth))
            
        elif covering_node is None:
            #introduce a virtual switch, and modify the VN to run all the bandwidth in this component through it.
            sw = "virtual_switch%d"%(len(virtual_switches))
            virtual_switches.add(sw)
            covering_node=sw
            maxflow_requirements[sw]=[]
            sum_bw =  OrderedDict()
            for n in component:
                sum_bw[n]=0
            newvn=[]
            for u,v,bandwidth in instance["VN"]:
                
                if u in component_nodes:
                    assert(v in component_nodes)
                    sum_bw[u]+=bandwidth
                    sum_bw[v]+=bandwidth                    
                else:
                    assert(v not in component_nodes)
                    newvn.append((u,v,bandwidth))
            total_bandwidth=0        
            for u, sum_bandwidth in sum_bw.items():
                newvn.append((u,sw,sum_bandwidth))
                maxflow_requirements[sw].append((u, sum_bandwidth))
                total_bandwidth+=sum_bandwidth
                
            instance["VMs"][sw]= [0, 0,0, total_bandwidth] #A virtual switch is a vm with neither ram nor core requirements (and hence doesn't need to be allocated to any particular server).
            instance["VN"]=newvn
            
        
        refinement_maxflow_requirements[covering_node]=dict()
        for n in component:
            refinement_maxflow_requirements[covering_node][n]=[]
        if use_cover_for_refinement:            
            covered_edges = set()
            for n in cover:
                if n in component:
                    #this can be improved
                    for u,v,bandwidth in original_vn:
                        
                        if (u,v) not in covered_edges and (u == n or v==n):
                            covered_edges.add((u,v))      
                            assert(u in component)
                            assert(v in component)
                            assert(u==n or v==n)
                            if u==n:
                                dest=v
                            else:
                                dest=u
                            refinement_maxflow_requirements[covering_node][n].append((dest, bandwidth))
        else:
            for u,v,bandwidth in original_vn:
                if u in component or v in component:
                    assert(v in component)
                    assert(u in component)
                    refinement_maxflow_requirements[covering_node][u].append((v, bandwidth))

    if len(virtual_switches)>0:
        print("Added %d virtual switches"%(len(virtual_switches)))

if len(maxflow_requirements)==0:
    print("Warning: no flow requirements!")
    sys.exit(1) 
                       
#for each maxflow requirement, create a copy of the physical network (slightly altered to convert multiple destinations into a single destination)
bvwidth = 1
for u,v,bandwidth in instance["VN"]:
    if (bandwidth>= 1<<bvwidth):
        bvwidth = math.ceil(math.log(bandwidth+1)/math.log(2))
for s1, s2, bandwidth in instance["PN"]:
    if (bandwidth>= 1<<bvwidth):
        bvwidth = math.ceil(math.log(bandwidth+1)/math.log(2))      
for source in instance["VMs"]:
#for source, flow_requirements in maxflow_requirements.items():
    sum_bandwidth = 0
    if source in maxflow_requirements:
        flow_requirements = maxflow_requirements[source]
        if  len(flow_requirements)==0:
            continue;
        sum_bandwidth = 0
        for dest, bandwidth in flow_requirements:
            sum_bandwidth+=bandwidth  
    if (sum_bandwidth>= 1<<bvwidth):
        bvwidth = math.ceil(math.log(sum_bandwidth+1)/math.log(2))           
print("Bitvector width is %d"%(bvwidth))   

monosat.bvtheory.BVManager().bitblast_addition=bitblast_addition
monosat.bvtheory.BVManager().bitblast_addition_shadow=bitblast_addition_shadow       
physical_graph = Graph()

#we have a set of physical servers, connected by a fixed topology of switches, with fixed bandwidth
physical_servers = [] #(node,servername,[cores,ram,bandwidth])
hostdict = dict()
switches=[]
switchdict = dict()
servercores = dict()
serverram=dict()
serverstorage=dict()
serverbandwidth=dict()

vmcores = dict()
vmram=dict()
vmstorage=dict()
vmbandwidth=dict()#aggregate bandwidth requirements of the vm, equal to the sum of all its bandwidth requirements in the virtual network

physical_nodes = dict()
for servername in instance["Servers"]:
    #List of servers that can host virtual machines
    data = instance["Servers"][servername]    
    data = list(map(int,data))
    cores,ram,storage,bandwidth= data
    servercores[servername]=cores 
    serverram[servername]=ram
    serverstorage[servername]=storage
    serverbandwidth[servername]=bandwidth
    
    node = physical_graph.addNode(servername)
    physical_nodes[servername]=node
    hostdict[servername]=len(physical_servers)
    physical_servers.append((node,servername,data))
        
#print("graph{\nrankdir = BT")
#for s in instance["Servers"]:
#    print(s + " [label = \"c%d m%d d%d \"]"%(servercores[s],serverram[s],serverstorage[s]))
switches=[]
pn_edges=[]
for i,(s1, s2, bandwidth) in enumerate(instance["PN"]):
    #each connection is of the form ["s0", "tor0", 1000]
    bandwidth = int(bandwidth)
    if s1 in physical_nodes:
        n1 = physical_nodes[s1]
    else:
        n1 =  physical_graph.addNode(s1)
        physical_nodes[s1]=n1
        switches.append(s1)

    if s2 in physical_nodes:
        n2 = physical_nodes[s2]
    else:
        n2 =  physical_graph.addNode(s2)
        physical_nodes[s2]=n2
        switches.append(s2)
    if not directed_links:
        e = physical_graph.addUndirectedEdge(n1,n2, bandwidth)
    else:
        e = physical_graph.addEdge(n1,n2, bandwidth)
    pn_edges.append((e,(s1,s2,bandwidth)))
    if assert_all_physical_edges:
        Assert(e)
#    print(s1 + " -- " + s2 + "[xlabel=\"%d Mbps\"]"%(bandwidth))
#print("}")




"""print("graph{")
for vm in instance["VMs"]:
    print(vm)
for u,v,bandwidth in instance["VN"]:
    print(u + " -- " + v + "[xlabel=\"%s Mbps\"]"%(bandwidth))
print("}")
"""
"""if original_vn:
    print("Original VN:")
    print("graph{")
    for u,v,bandwidth in original_vn:
        print(u + " -- " + v + "[xlabel=\"%s Mbps\"]"%(bandwidth))
    print("}")"""

virt_nodes_out = dict()
virt_nodes_in = dict()
max_virtbandwidth =dict()

for  n in instance["VMs"]:
    cores,ram,storage,aggregatebandwidth =  instance["VMs"][n]
    v = physical_graph.addNode("virt_out_"+str(n))
    virt_nodes_out[n]=v
    v2 = physical_graph.addNode("virt_in_"+str(n))
    virt_nodes_in[n]=v2
    
    #count up all the bandwidth in or out of this virtual machine in the virtual network:
    maxbandwidth =0
    #speed this up:
    for vm1, vm2,bandwidth in instance["VN"]:
        if vm1 == n or vm2 == n:
            maxbandwidth+=int(bandwidth)
    #if aggregatebandwidth != maxbandwidth:
    #   print("Warning: specified bandwidth (%d) for vm "%(aggregatebandwidth) + str(n) + " doesn't match the actual aggregate bandwidth of its connections (%d); using the aggregate bandwidth value"%(maxbandwidth))
    #maxbandwidth = max(maxbandwidth,aggregatebandwidth)
    max_virtbandwidth[n]=maxbandwidth
    
    vmcores[n]=cores 
    vmram[n]=ram 
    vmstorage[n]=storage
    vmbandwidth[n]=maxbandwidth 
    

vm_server_assignments=dict()
server_vm_assignments=dict()
switch_vm_assignments=dict()

for host in instance["Servers"]:
    server_vm_assignments[host] =dict()
for switch in switches:
    switch_vm_assignments[switch] =dict()
all_edge_sets=[]
edge_sets=dict()
for vm in instance["VMs"]:
    edge_sets[vm]=[]
    start_vm =  time.clock()
    server_edges = []
    virtnode_out = virt_nodes_out[vm]
    virtnode_in = virt_nodes_in[vm]
    maxbandwidth = max_virtbandwidth[vm]
    
    for host in instance["Servers"]:     
        e = physical_graph.addEdge(virtnode_out, physical_nodes[host],maxbandwidth)
        #e.setSymbol("virt_"+str(vm) + "to_host_" + str(host))

        server_edges.append((e,host))
        server_vm_assignments[host][vm]=e        
        e2 = physical_graph.addEdge(physical_nodes[host],virtnode_in,maxbandwidth)
        AssertEq(e,e2)

    #this is a virtual switch, and so can be allocated to switches in the network
    if vmcores[vm]==0 and vmram[vm]==0  and vmstorage[vm]==0 and allow_virtual_switches_in_switches:
        
        for host in switches:     
            e = physical_graph.addEdge(virtnode_out, physical_nodes[host],maxbandwidth)
            #e.setSymbol("virt_"+str(vm) + "to_switch_" + str(host))

            server_edges.append((e,host))
            switch_vm_assignments[host][vm]=e 
            #server_vm_assignments[host][vm]=e        
            e2 = physical_graph.addEdge(physical_nodes[host],virtnode_in,maxbandwidth)
            AssertEq(e,e2)
    
    #Each virtual machine is assigned to exactly one physical server
    edge_vars = [x[0] for x in server_edges]
    vm_server_assignments[vm] =server_edges

    if (intrinsic_edge_sets):       
        physical_graph.newEdgeSet(edge_vars)
        all_edge_sets.append(edge_vars)
        edge_sets[vm].append(edge_vars)
    elif (intrinsic_edge_constraints):
        AssertExactlyOne(edge_vars);
    else:
        AssertEqualPB(edge_vars,1);
        if flush_pb_theory: #In large physical networks, these PB constraints can become excessively expensive to encode (because minisat+ is trying very hard to optimize them); forcing them into clauses immediately keeps minisat+ from trying those optimizations  (which are probably not very helpful here anyhow.)      
            PBManager().flush()
   

print(Monosat().nVars())
print("Enforcing server core/ram counts")

server_core_assignments=dict()
server_ram_assignments=dict()
server_storage_assignments=dict()
#Now, ensure that no server is assigned more virtual machines than it has the capacity to host.
#For now, we'll ignore ram, and model server capacity as just the number of cores.
#In the fmcad paper, each server has a capacity of 4 virtual machines.
for host in instance["Servers"]:
    cores, ram,storage, bandwidth = servercores[host],serverram[host],serverstorage[host],serverbandwidth[host]
    vm_assigns = server_vm_assignments[host]    
    edges = []
    core_weights=[]
    ram_weights=[]
    storage_weights=[]
    bandwidth_weights=[]
    for vmname in instance["VMs"]:
        if vmname in vm_assigns:
            edge = vm_assigns[vmname]

            nvmcores, nvmram,nvstorage, nvmbandwidth = vmcores[vmname],vmram[vmname],vmstorage[vmname],vmbandwidth[vmname]        
            edges.append(edge)
            core_weights.append(nvmcores) 
            ram_weights.append(nvmram)
            storage_weights.append(nvstorage)
            bandwidth_weights.append(nvmbandwidth)   

    server_core_assignments[host]=(edges,cores,core_weights)
    server_ram_assignments[host]=(edges,ram,ram_weights)
    server_storage_assignments[host]=(edges,storage,storage_weights)
    #sometimes, one of these two constraints is implied by the other one. Should probably detect that and remove the other constraint.
    AssertLessEqPB(edges,cores,core_weights);#Weighting the pseudo-boolean constraints by the ram and core requirements of each virtual machine...
    if not ignore_ram_constraints:
        AssertLessEqPB(edges,ram,ram_weights);
    if not ignore_storage_constraints:
        AssertLessEqPB(edges,storage,storage_weights);
    #Note: need to be careful here, because two vms in the same server have free bandwidth between each other, which may impact the above bandwidth requirement calculation
    #As a result, this constraint is not valid:
    #AssertLessEqPB(edges,bandwidth,bandwidth_weights);

#We will then assert that 
#a) all physical network copies have the same edge assignment, and
#b) the sum of the bandwidth of all physical network copy edges is <= the allowed bandwidth for that edge 
flow_requirement_lits=dict()
physical_graph_copies=[]
all_edges=[ [] for i in instance["PN"]]

for source in instance["VMs"]:
#for source, flow_requirements in maxflow_requirements.items():
    if source in maxflow_requirements:
        flow_requirements = maxflow_requirements[source]
        if  len(flow_requirements)==0:
            continue;
        
        sum_flow = 0
        for dest, bandwidth in flow_requirements:    
            sum_flow+=bandwidth
        
        print("maxflow for " + source)   
        physical_graph_copy = Graph()
        for n in range(physical_graph.nodes):
            physical_graph_copy.addNode()
        #add all the edges of the physical network, with bitvector weights.
        #and add the edges for the _relevant_ virtual machines for this flow query.
        copy_edges=dict()
        for i,( s1, s2, bandwidth) in enumerate(instance["PN"]):
            bandwidth = bandwidth
            
                
            u = physical_nodes[s1]    
            v = physical_nodes[s2]
            
            bv2 = bv(bvwidth)
            bv2.setSymbol("bw_" + str(u) + "to" + str(v) + "_source_"+str(source))
            #can't do this because on the next round, we will be reducing the flow on each edge...
            #if static_flow_caps and n_flow_requirements==1:
            #    Assert(bv2==bandwidth)#If only one physical network is instantiated (because there is only one flow requirement), then we can set all edge capacities to their maximum values.
            #else:
            if static_flow_caps and sum_flow<bandwidth:
                Assert(bv2<=sum_flow)#we can't possibly assign more flow than sum_flow, so cap the bv at that level.
            else:
                Assert(bv2<=bandwidth)
                
            all_edges[i].append(bv2)        
            e2,(s1_check,s2_check,bandwidth_check) = pn_edges[i]
            assert(s1==s1_check)#sanity checks
            assert(s2==s2_check)
            assert(bandwidth==bandwidth_check)
            
            if not directed_links:            
                eforward = physical_graph_copy.addEdge(u,v,bv2)
                eback = physical_graph_copy.addEdge(v,u,bv2)
                AssertEq(eforward,eback)
            else:
                eback=None
                eforward = physical_graph_copy.addEdge(u,v,bv2)
            #e = physical_graph_copy.addUndirectedEdge(u,v,bv2)
            AssertEq(eforward,e2)#ensure that this edge in the physical network copy is enabled only if it is enabled in the master physical network.
            
            copy_edges[i]=(eforward,eback)
            #This edge can only have non-zero weight if it is enabled.
            if not assert_all_physical_edges:
                AssertImplies(bv2>0, eforward)

           
            
        sum_bandwidth = 0
        for dest, bandwidth in flow_requirements:
            sum_bandwidth+=bandwidth
        edgelitmap = dict()    
        #add edges for the virtual machine source   
        for var, host in vm_server_assignments[source]:
            assert(sum_bandwidth<=max_virtbandwidth[source])
            h = physical_nodes[host]
            vm = virt_nodes_out[source]            
            e = physical_graph_copy.addEdge(vm,h,bv(bvwidth,sum_bandwidth))
            AssertEq(e,var);
            edgelitmap[var]=e

        if (intrinsic_edge_sets):
            for edgeset in edge_sets[source]:
                copyedgeset = [edgelitmap[e] for e in edgeset]
                physical_graph_copy.newEdgeSet(copyedgeset)
        
        for dest, bandwidth in flow_requirements:
            vm = virt_nodes_in[dest] 
            assert(bandwidth<=max_virtbandwidth[dest])
            for var, host in vm_server_assignments[dest]:
                h = physical_nodes[host]                           
                e = physical_graph_copy.addEdge(h,vm,bv(bvwidth,bandwidth))
                AssertEq(e,var);
        dests=[]
        if len(flow_requirements)>1:
            #if there are multiple flow requirements for this source, collect them all into one, new sink node
            sum_dest = physical_graph_copy.addNode()
            for dest, bandwidth in flow_requirements:       
                dests.append(dest)
                virt_dest=  virt_nodes_in[dest]            
                Assert(physical_graph_copy.addEdge(virt_dest,sum_dest,bv(bvwidth,bandwidth)))
        else:
            dests.append(flow_requirements[0][0])
            sum_dest = virt_nodes_in[flow_requirements[0][0]]
           
        virt_source=  virt_nodes_out[source]  
        mf =physical_graph_copy.maxFlowGreaterOrEqualTo(virt_source, sum_dest,sum_bandwidth)
        flow_requirement_lits[source]=(physical_graph_copy,copy_edges,mf,dests)
        Assert(mf)              
                
                
print("enforcing sums")            
sums=dict()
#ensure the sum of all edges in the network copies add up to less or equal to the actual bandwidth of that edge 
for i,(s1, s2, wt) in enumerate( instance["PN"]):

    n1 = physical_nodes[s1]
    n2 = physical_nodes[s2]
    assert(len(all_edges[i])>0);
    if not tree_addition:
        sm = bv(bvwidth,0)    
        for bv2 in all_edges[i]:
            sm += bv2
        
    else:
        elements = list(all_edges[i])
        while(len(elements)>1):
            if (len(elements)%2 == 1):
                elements.append(bv(bvwidth,0))
    
            next_elements =[]
            for a,b in zip(elements[::2],elements[1::2]):
                next_elements.append(a+b)
            elements= next_elements
        assert(len(elements)==1);
        sm = elements[0]
    sums[i]=(sm,wt)
    Assert(sm<=wt)
    
if anti_affinity and "antiaffinity" in vn:
    print("Enforcing anti-affinity constraints...")
    anti_affinity_sets = vn["antiaffinity"]
    for anti_affinity_set in anti_affinity_sets:
        anti_affinity_set = set(anti_affinity_set)
        if (len(anti_affinity_set)<2):
            continue
        #An anti-affinity set is a collection of 2 or more (unique) vms, which 
        #Canot be placed on the same server (or rack, for rack_antiaffinity)
        
        #For each vm in the anti_affinity set, and each server assignment link s,
        #Use PB constraints to assert that at most one of (link(vm,s)) in the anti
        #affinity set is selected
        for s in instance["Servers"]:
            links=[]            
            for vm in anti_affinity_set:
                e= server_vm_assignments[s][vm]
                links.append(e)
            AssertLessEqPB(links,1)

if args.affinity and not args.affinity_soft and "affinity" in vn:
    print("Enforcing hard affinity constraints...")
    affinity_sets = vn["affinity"]
    for affinity_set in affinity_sets:
        affinity_set = set(affinity_set)
        if (len(affinity_set)<2):
            continue
        #A hard affinity set is a collection of 2 or more (unique) vms, which 
        #must be placed on the same server
        
        #For each vm in the affinity set, and each server assignment link s,
        #Use PB constraints to assert that at most one of (link(vm,s)) in the anti
        #affinity set is selected
        for s in instance["Servers"]:
            links=[]            
            for vm in affinity_set:
                e= server_vm_assignments[s][vm]
                links.append(e)
            AssertEq(links)
            
min_bvs=[]
assumptions=[]

if args.affinity and args.affinity_soft and "affinity" in vn:
    print("Enforcing soft affinity constraints...")
    affinity_sets = vn["affinity"]
    for affinity_set in affinity_sets:
        affinity_set = set(affinity_set)
        if (len(affinity_set)<2):
            continue
        #A soft affinity set is a collection of 2 or more (unique) vms, which 
        #should be placed on the same server, if possible        
        
        #For each vm in the affinity set, and each server assignment link s,
        #Use PB constraints to assert that at most one of (link(vm,s)) in the anti
        #affinity set is selected
        servers_with_vms_from_set=[]

        for s in instance["Servers"]:
            links=[]            
            for vm in affinity_set:
                e= server_vm_assignments[s][vm]
                links.append(e)
            servers_with_vms_from_set.append(Or(links))
            
        if popcount=="BV":
            has_affinity_count = PopCount(servers_with_vms_from_set,method="BV",return_bv=True)
        else:
            n_used = PopCount(servers_with_vms_from_set,method=popcount)            
            has_affinity_count = BitVector(n_used);  

        Assert(has_affinity_count>=1)    
        min_bvs.append(has_affinity_count)
        
if min_cores:
    print("Adding constraints to minimize the most utilized server in the current round (in terms of cores used)...")
    #Note: This is a local optimization, for the current round only, that attempts to minimize the most utilized CPU used in
    #This current allocation round. The utilization of CPUs that are not assigned to 
    #in the current round is intentionally ignored.
    max_cores=0
    for host in instance["Servers"]:
        max_cores= max(max_cores,servercores[host])
    cpu_width = int( math.ceil(math.log(max_cores+1)/math.log(2)))
    used_server_cores=dict()
    for host in instance["Servers"]:
        used = BitVector(cpu_width)
        used_server_cores[host]=used
        assumptions.append(used==0)
    total_core_usage=dict()
    for host in instance["Servers"]:
        cores = servercores[host]
        vm_assigns = server_vm_assignments[host]    
        core_weights=[]
        edges=[]
        #for vmname,edge in vm_assigns.items(): 
        for vmname in instance["VMs"]:
            if vmname in vm_assigns:
                edge = vm_assigns[vmname]           
                nvmcores = vmcores[vmname]
                edges.append(edge)
                core_weights.append(nvmcores)
        any_used_this_round = Or(edges)
        total = used_server_cores[host]
        for vm_edge,vm_required_cores in zip(edges,core_weights):
            total = If(vm_edge,total+vm_required_cores,total)
        
        total_core_usage[host]=If(any_used_this_round,total,0)
        Assert(total_core_usage[host]<=cores)
    

    
    min_bv = Max(list(total_core_usage.values()))  
    min_bvs.append(min_bv)
    
if min_servers:
    print("Adding constraints to minimize the number of utilized servers...")
    

    #Important: We can statically derive a minimum number of possible servers
    #especially if anti-affinity constraints are used. This can save the solver
    #a _lot_ of useless effort.
    
    used_servers=dict()
    for host in instance["Servers"]:
        used = Var()
        used_servers[host]=used
        assumptions.append(Not(used))
        
    #minimize the number of new servers used.
    new_server_assignments=[] 
    any_servers_used=false()
    all_server_edges=[]
    for s in instance["Servers"]:
        
        server_edges = []
        for vm in instance["VMs"]:
            e= server_vm_assignments[s][vm]
            server_edges.append(e)
            all_server_edges.append(e)
            #If any vm is assigned to this server in the current assignment, then the server is used
            #(Below, we also need to set any server that has been assigned to in previous rounds to 'used')
            #AssertImplies(e,used_servers[s])
        any_used = Or(server_edges)
        #any_servers_used |=any_used
        new_server_assignments.append(And(Not(used_servers[s]), any_used ))  

    #
    if popcount=="BV":

        min_bv = PopCount(new_server_assignments,method="BV",return_bv=True)
    else:
        n_used = PopCount(new_server_assignments,method=popcount)
        min_bv = BitVector(n_used);   
        
    Assert(min_bv>=1) #At least one server must be used.
    min_bvs.append(min_bv)

print("solving...")

total_core_assignment=dict()
total_ram_assignment=dict()
total_storage_assignment=dict()
for host in instance["Servers"]:
    total_core_assignment[host]=0
    total_ram_assignment[host]=0
    total_storage_assignment[host]=0
    


residual_network=nx.Graph() if not directed_links else nx.DiGraph()

for u,v,bandwidth in instance["PN"]:
    residual_network.add_edge(u,v,weight=bandwidth,cost=1)
      
n_solutions=0

elapsed=0
elapsed_out=0
used_hosts=set()
print("Done Encoding..")
t = time.clock()
log=None
if log_file is not None:
    log=open(log_file,"w")
    
if log_time_file is not None:
    timefile = open(log_time_file,"w")    
    timefile.write("# instance " + instance_name + "\n")
    timefile.write("# " + "Settings: " + str(args)+"\n")
    prev_time=t
    timefile.write("init %f %f\n"%(t-start_time, t-start_time))
#os.system("callgrind_control -i on")
while   Solve(assumptions, bvs_to_minimize=min_bvs,conflict_limit = args.conflict_limit):
    #os.system("callgrind_control -i off")
    et = time.clock()
    elapsed+= et-t
    assumptions=[]
    n_solutions+=1
    print("Solved %d"%(n_solutions))
    found_optimal = FoundOptimal()

    if log:       
        log.write("# vdc allocation %d\n"%(n_solutions)) 
        if(found_optimal):
            log.write("# proved optimal allocation\n")
        else:
            log.write("# failed to prove optimal allocation\n")   
        log.write("vdc\n")
    
    vm_assignments=dict()
    server_assigned_cores=dict()
    server_assigned_ram=dict()
    server_assigned_storage=dict()
    for host in instance["Servers"]:  
        assigned_cores=0
        assigned_ram=0
        assigned_storage=0
        for vm in instance["VMs"]: 
            e = server_vm_assignments[host][vm]

            if e.value():
                #vm was assigned to host
                #so, we need to reduce the number of available cores and ram on the host.
                vm_assignments[vm]=host
                cores = vmcores[vm]
                ram = vmram[vm]
                storage=vmstorage[vm]
                assigned_cores+=cores
                assigned_ram+=ram
                assigned_storage+=storage
                if log:
                    log.write("%s -> %s\n"%(vm,host))
                print("vm " + vm + " assigned to " + host)               
                used_hosts.add(host)
   
        server_assigned_cores[host]=assigned_cores 
        server_assigned_ram[host]=assigned_ram       
        server_assigned_storage[host]=assigned_storage  
    for virtual_switch in virtual_switches:  
        if virtual_switch not in  vm_assignments:
            #the virtual switch was attached to a switch, not a server
            for switch in switches:
                e = switch_vm_assignments[switch][virtual_switch]
                if e.value():
                    vm_assignments[virtual_switch]=switch
                    print("vswitch " + vm + " assigned to " + host)
    for vm in instance["VMs"]:  
        if vm not in vm_assignments:
            print("Error: unassigned vm " + vm)
        assert(vm in vm_assignments)

    changed=dict()
    additional_flow=dict()
    for i,( s1, s2, bandwidth) in enumerate(instance["PN"]):
        changed[i]=0
        additional_flow[(s1,s2)]=0
        additional_flow[(s2,s1)]=0
    directed_flow_nets=dict()  
    for source in instance["VMs"]:
    #for source, flow_requirements in maxflow_requirements.items():
        
        if source in maxflow_requirements:
            refine_this_flow = refine_flow and source in virtual_switches
            flow_requirements = maxflow_requirements[source]
            if  len(flow_requirements)==0:
                continue;    
            (physical_graph_copy,copy_edges,mf,dests) = flow_requirement_lits[source]
            flow=physical_graph_copy.getMaxFlow(mf)
            source_dest_flow=flow
            assert(flow>0)
   
            flow_req_dict = dict()
            for (dest, bandwidth) in flow_requirements:
                flow_req_dict[dest]=bandwidth

            directed_flow_net=nx.DiGraph()
            for a,b,bw in instance["PN"]:
                directed_flow_net.add_node(a)
                directed_flow_net.add_node(b)
            
           
           
            for i,( s1, s2, bandwidth) in enumerate(instance["PN"]):
                efor= copy_edges[i][0]
                eback = copy_edges[i][1]
                ffor = physical_graph_copy.getEdgeFlow(mf,efor,remove_flow_cycles)
                fback = physical_graph_copy.getEdgeFlow(mf,eback,remove_flow_cycles) if eback is not None else 0 
            
                if ffor>0:
                    if not directed_flow_net.has_edge(s1,s2):
                        directed_flow_net.add_edge(s1,s2,weight=0)
                    directed_flow_net[s1][s2]['weight']+=ffor
                if fback>0:
                    if not directed_flow_net.has_edge(s2,s1):
                        directed_flow_net.add_edge(s2,s1,weight=0)
                    directed_flow_net[s2][s1]['weight']+=fback
                #edge flows in PN can be 0, or less than the total flow, if some of the vms were assigned to the same server.
                assert(ffor>=0)
                assert(fback>=0)
                f = max(ffor,fback)-min(ffor,fback)
                assert(f>=0)
                assert(f<=bandwidth)   
                
                if f>0:
                    assert(residual_network[s1][s2]['weight']>=f)
                    residual_network[s1][s2]['weight']-=f                    
                    if residual_network[s1][s2]['weight']==0:
                        residual_network.remove_edge(s1,s2)
            #printDiGraph(directed_flow_net)    
            directed_flow_nets[source]=directed_flow_net
            if remove_flow_cycles and debug:
                if(not is_directed_acyclic_graph(directed_flow_net)):
                    printDiGraph(directed_flow_net)
                    assert(False)
    
     
    for source in instance["VMs"]:    
        if source in maxflow_requirements:
            refine_this_flow = refine_flow and source in virtual_switches
            flow_requirements = maxflow_requirements[source]
            if  len(flow_requirements)==0:
                continue;    
            (physical_graph_copy,copy_edges,mf,dests) = flow_requirement_lits[source]
            flow=physical_graph_copy.getMaxFlow(mf)
            source_dest_flow=flow
            assert(flow>0)
      
            #this can be improved
            flow_req_dict = dict()
            for (dest, bandwidth) in flow_requirements:
                flow_req_dict[dest]=bandwidth
                
            directed_flow_net=directed_flow_nets[source]
                                      
            #before any refinement can happen, ALL flows from all flow sources must be removed from the residual network.     
            if refine_this_flow:             
                assert(force_virtual_switches)
                assert(source in virtual_switches)   
                #Starting with the _RESIDUAL_ network
                #find shortest paths between connections.
                #while such a path exists, move flow from its old path (which went through source) to the new one.
                #this is always safe to do, and is guaranteed not to remove the connections that the solver found (as we always operate on the residual network). 
                #this can be short-cut for vm's assigned to the same server.
                print("Refinining...")
                #printGraph(flow_network)

                    
                #Should only consider the flows req that this maxflow is covering... especially for virtual switches.
                #generalize this later...
                extra_node = "extra_node"
                extra_dest_node = "extra_dest_node"
                extra_source_node = "extra_source_node"
                directed_flow_net.add_node(extra_source_node)
                directed_flow_net.add_node(extra_node)
                directed_flow_net.add_node(extra_dest_node)
                
                outer_edge = directed_flow_net.add_edge(extra_node,vm_assignments[source],weight=0 )
                for vm1 in refinement_maxflow_requirements[source]:
                    for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]:
                        s1 = vm_assignments[vm1]
                        s2 = vm_assignments[vm2]
                        sVirt = vm_assignments[source]         
                        found_path=[]
                        if s1==s2:               
                            if sVirt!=s1:
                                directed_flow_net[extra_node][sVirt]["weight"]=bandwidth
                                #(that flow had to pass through the virtual switch)
                                maxflow, edge_flows =  networkx.algorithms.flow.maxflow.maximum_flow(directed_flow_net,extra_node,s1,capacity="weight")
                                #note: this _requires_ networkx 1.9+; the old ford_fulkerson code in networkx 1.8 had a subtly different interface that will break the code below!
                                assert(maxflow==bandwidth)
                                #remove that flow from the returned flow-net                       
                                for (residual_path,pathflow) in getFlowPaths(edge_flows,sVirt,s1,bandwidth, capacity='weight'):
                                    for (u,v) in pairwise(residual_path): 
                                        if u != extra_node and v != extra_node:
                                            assert(pathflow>0)
                                            assert(directed_flow_net[u][v]['weight']>=pathflow)
                                            directed_flow_net[u][v]['weight']-=pathflow 
                                            if directed_flow_net[u][v]['weight']==0:
                                                directed_flow_net.remove_edge(u,v)
                                            if not residual_network.has_edge(u,v):
                                                residual_network.add_edge(u,v,weight=0,cost=1)        
                                            residual_network[u][v]['weight']+=pathflow
                                           
                            if sVirt!=s2:
                                directed_flow_net[extra_node][sVirt]["weight"]=bandwidth
                                #(that flow had to pass through the virtual switch)                                
                                maxflow, edge_flows =  networkx.algorithms.flow.maxflow.maximum_flow(directed_flow_net,extra_node,s2,capacity="weight")
                                #note: this _requires_ networkx 1.9+; the old ford_fulkerson code in networkx 1.8 had a subtly different interface that will break the code below!
                                assert(maxflow==bandwidth)
                                #remove that flow from the returned flow-net
                                #print(edge_flows)
                                
                                #need to decompose into (simple) flow paths to avoid cycles here...                        
                                for (residual_path,pathflow) in getFlowPaths(edge_flows,sVirt,s2,bandwidth, capacity='weight'):
                                    for (u,v) in pairwise(residual_path):    
                                        if u != extra_node and v != extra_node:     
                                            assert(pathflow>0)
                                            assert(directed_flow_net[u][v]['weight']>=pathflow)
                                            directed_flow_net[u][v]['weight']-=pathflow 
                                            if directed_flow_net[u][v]['weight']==0:
                                                directed_flow_net.remove_edge(u,v)                                                
                                            if not residual_network.has_edge(u,v):
                                                residual_network.add_edge(u,v,weight=0,cost=1)        
                                            residual_network[u][v]['weight']+=pathflow
                                        
                
                print("Refinining after removing pairs...")
                #printGraph(directed_flow_net)                        
                for vm1 in refinement_maxflow_requirements[source]:
                    for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]:
                    
                        s1 = vm_assignments[vm1]
                        s2 = vm_assignments[vm2]
                        sVirt = vm_assignments[source]         
                        
                        if s1 != s2: # and s1 != sVirt and s2 != sVirt:
                            #printGraph(directed_flow_net)
                            directed_flow_net.add_edge(s1,extra_dest_node,weight=bandwidth)
                            directed_flow_net.add_edge(s2,extra_dest_node,weight=bandwidth)
                            maxflow, edge_flows =  networkx.algorithms.flow.maxflow.maximum_flow(directed_flow_net,sVirt,extra_dest_node,capacity="weight")
                            #note: this _requires_ networkx 1.9+; the old ford_fulkerson code in networkx 1.8 had a subtly different interface that will break the code below!
                            if(maxflow!=bandwidth*2):                                
                                printDiGraph(directed_flow_net)
                                print("%s %s"%(s1, s2));
                            assert(maxflow==bandwidth*2)
                            #remove that flow from the returned flow-net
                            #print(edge_flows)
    
                            for (residual_path,flow) in getFlowPaths(edge_flows,sVirt,extra_dest_node,bandwidth*2, capacity='weight'):
                                for (s,t) in pairwise(residual_path):
                                    if s != extra_dest_node and t != extra_dest_node:                            
                                        assert(flow>0)
                                        assert(directed_flow_net[s][t]['weight']>=flow)
                                        directed_flow_net[s][t]['weight']-=flow 
                                        if directed_flow_net[s][t]['weight']==0:
                                            directed_flow_net.remove_edge(s,t) 
                                        #if not directed_flow_net.has_edge(t,s):
                                        #    directed_flow_net.add_edge(t,s,weight=0)
                                        #directed_flow_net[t][s]['weight']+=flow   
                                        
                                        if not residual_network.has_edge(s,t):
                                            residual_network.add_edge(s,t,weight=0,cost=1)          
                                        #print(str(residual_network[s][t]['weight']))
                                        residual_network[s][t]['weight']+=flow
                            
                            #Now that we have removed the current flow from the flow graph and added it to the residual graph, a valid s1 s2 flow of bandwidth is guaranteed to exist in bandwidth.
                            #use min_cost_max_flow to find the shortest such flow (which, in the worst case, is the same one we just added to the residual graph)
                            directed_flow_net.remove_edge(s1,extra_dest_node)
                            directed_flow_net.remove_edge(s2,extra_dest_node)

                            if not use_cover_for_refinement:
                    
                                s1 = vm_assignments[vm1]
                                s2 = vm_assignments[vm2]
                                sVirt = vm_assignments[source]                                
                                assert(s1!=s2)
                                residual_network.add_edge(s2,extra_dest_node,weight=bandwidth,cost=0)
                                if use_min_cost_max_flow:
                                    maxflow, edge_flows = networkx.algorithms.flow.mincost.max_flow_min_cost(residual_network,s1,extra_dest_node,capacity="weight", weight="cost")
                                else:
                                    maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(residual_network,s1,extra_dest_node,capacity="weight")
                                
                                residual_network.remove_edge(s2,extra_dest_node)
 
                                assert(maxflow==bandwidth)
                                #max_flow_min_cost
                                #need to decompose into (simple) flow paths to avoid cycles here...                        
                                for (residual_path,pathflow) in getFlowPaths(edge_flows,s1,s2,bandwidth, capacity='weight'):              
                                    for (u,v) in pairwise(residual_path):   
                                        if u !=  extra_dest_node and v != extra_dest_node:      
                                            additional_flow[(v,u)]+=pathflow
                                            if not directed_links:
                                                additional_flow[(u,v)]+=pathflow
                                            #print(str(residual_network[u][v]['weight']))
                                            residual_network[u][v]["weight"]-=pathflow
                                            #print(str(residual_network[u][v]['weight']))
                                            if residual_network[u][v]["weight"]==0:
                                                residual_network.remove_edge(u,v)    
                                    if log:
                                        log.write("path %s -> %s %d ="%(vm1, vm2, pathflow*common_bw_factor))
                                        for u in residual_path:
                                            if u !=  extra_dest_node:
                                                log.write(" " + u)
                                        log.write("\n")
           
                    if use_cover_for_refinement:   
                        assert(len(residual_network.edges(extra_dest_node))==0) 
                        sum_flow=0  
                        s1 = vm_assignments[vm1]    
                        #print(vm1)          
                        for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]:          
                            
                            s2 = vm_assignments[vm2]
                            if s1!=s2:                                                         
                                sum_flow+=bandwidth
                                if not residual_network.has_edge(s2,extra_dest_node):
                                    residual_network.add_edge(s2,extra_dest_node,weight=0,cost=0)
                                residual_network[s2][extra_dest_node]["weight"]+=bandwidth
                        if sum_flow==0:
                            continue
                        if use_min_cost_max_flow:
                            maxflow, edge_flows = networkx.algorithms.flow.mincost.max_flow_min_cost(residual_network,s1,extra_dest_node,capacity="weight", weight="cost")
                        else:
                            maxflow, edge_flows = networkx.algorithms.flow.maxflow.maximum_flow(residual_network,s1,extra_dest_node,capacity="weight")
                        
                        #printGraph(residual_network)
                        #printGraph(residual_network)

                        assert(maxflow==sum_flow)
                        #max_flow_min_cost
                        #need to decompose into (simple) flow paths to avoid cycles here...
                        vm1_flow_reqs=dict()
                        for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]: 
                            vm1_flow_reqs[vm2]=bandwidth
                            
                        for (residual_path,pathflow) in getFlowPaths(edge_flows,s1,extra_dest_node,sum_flow, capacity='weight'):  
                            server =residual_path[-2]
                            
                            for (u,v) in pairwise(residual_path):   
                                if u !=  extra_dest_node and v != extra_dest_node:      
                                    additional_flow[(v,u)]+=pathflow
                                    if not directed_links:
                                        additional_flow[(u,v)]+=pathflow
                                    if residual_network[u][v]["weight"]<pathflow:
                                        pass
                                    #print(str(residual_network[u][v]['weight']))
                                    assert(residual_network.has_edge(u,v))
                                    assert(residual_network[u][v]["weight"]>=pathflow)
                                    residual_network[u][v]["weight"]-=pathflow
                                    #print(str(residual_network[u][v]['weight']))
                                    if residual_network[u][v]["weight"]==0:
                                        residual_network.remove_edge(u,v)    
                                            
                            if log:
                                #now assign the flow along this path to destination vms as needed.
                                while pathflow>0:
                                    found_vm=None
                                    used_flow = 0
                                    for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]: 
                                        required_bandwidth=vm1_flow_reqs[vm2]
                                        s2 = vm_assignments[vm2]   
                                        if s2==server and required_bandwidth>0:
                                            found_vm=vm2
                                            used_flow = min(required_bandwidth,pathflow)
                                            assert(vm1_flow_reqs[vm2]>=used_flow)
                                            vm1_flow_reqs[vm2]-=used_flow
                                            break
                                    assert(pathflow>=used_flow)
                                    pathflow-=used_flow
                                    log.write("path %s -> %s %d ="%(vm1, found_vm, used_flow*common_bw_factor))
                                    for u in residual_path:
                                        if u !=  extra_dest_node:
                                            log.write(" " + u)
                                    log.write("\n")      
                        for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]:                                      
                            s2 = vm_assignments[vm2]
                            if s1!=s2:         
                                if residual_network.has_edge(s2,extra_dest_node):                                    
                                    residual_network.remove_edge(s2,extra_dest_node)                                   
                        assert(len(residual_network.edges(extra_dest_node))==0) 
                directed_flow_net.remove_edge(extra_node,vm_assignments[source])   
                directed_flow_net.remove_node(extra_node)

               
            for i,( s1, s2, bandwidth) in enumerate(instance["PN"]):
                ffor = directed_flow_net[s1][s2]["weight"] if directed_flow_net.has_edge(s1,s2) else 0
                if not directed_links:
                    fback = directed_flow_net[s2][s1]["weight"]  if directed_flow_net.has_edge(s2,s1) else 0
                else:
                    fback=0  
                #f = directed_flow_net[s1][s2]["weight"] if directed_flow_net.has_edge(s1,s2) else 0
                f = max(ffor,fback)-min(ffor,fback)
                f+= additional_flow[(s1,s2)]
                additional_flow[(s1,s2)]=0
                if not directed_links:
                    additional_flow[(s2,s1)]=0
                assert(f>=0)
                assert(f<=bandwidth)
                #total_edge_flow[i]+=f
                #print(str(f))
                #total_edge_flow[i]+=f
                sm,wt = sums[i]
                assert(wt-f>=0)
                if f>0: 
                    sums[i]=(sm,wt-f)
                    changed[i]+=f     
            
                
            #Log the paths out if we haven't already done so.
            if log and not refine_this_flow:
                if not source in virtual_switches:
                    for dest in dests:
                        flowreq= flow_req_dict[dest]
                        if vm_assignments[source] !=vm_assignments[dest]:
                            
                            for (path,flow) in decomposeIntoSimplePaths(directed_flow_net,vm_assignments[source] ,vm_assignments[dest],flowreq,capacity='weight' ):
                                if log:
                                    log.write("path %s -> %s %d ="%(source, dest, flow*common_bw_factor))
                                    for u in path:                            
                                        log.write(" " + u)
                                    log.write("\n")              
                else:
                    extra_dest_node = "extra_dest_node"                       
                    directed_flow_net.add_node(extra_dest_node)
                    for vm1 in refinement_maxflow_requirements[source]:
                        for vm2,bandwidth in refinement_maxflow_requirements[source][vm1]:
                        
                          
                            s1 = vm_assignments[vm1]
                            s2 = vm_assignments[vm2]
                            sVirt = vm_assignments[source]         
                            
                            if s1 != s2: # and s1 != sVirt and s2 != sVirt:
        
                                directed_flow_net.add_edge(s1,extra_dest_node,weight=bandwidth)
                                directed_flow_net.add_edge(s2,extra_dest_node,weight=bandwidth)
                                maxflow, edge_flows =  networkx.algorithms.flow.maxflow.maximum_flow(directed_flow_net,sVirt,extra_dest_node,capacity="weight")
    
                                assert(maxflow==bandwidth*2)
                                #remove that flow from the returned flow-net
                                #print(edge_flows)
                                s1paths=[]
                                s2paths=[]
                                for (residual_path,flow) in getFlowPaths(edge_flows,sVirt,extra_dest_node,bandwidth*2, capacity='weight'):
                                    assert(residual_path[-1]==extra_dest_node)
                                    assert(residual_path[-2]==s1 or residual_path[-2]==s2)
                                    for (s,t) in pairwise(residual_path):                       
                                        assert(flow>0)
                                        assert(directed_flow_net[s][t]['weight']>=flow)
                                        directed_flow_net[s][t]['weight']-=flow 
                                        if directed_flow_net[s][t]['weight']==0:
                                            directed_flow_net.remove_edge(s,t) 
                                    if residual_path[-2]==s1:
                                        s1paths.append((flow,list(residual_path[:-1])))
                                    if residual_path[-2]==s2:
                                        s2paths.append((flow,list(residual_path[:-1])))
    
                                
                                #these form a set of paths from sVirt to s1 and s2. now pair them up as needed.
                                for i,(flow,path) in enumerate(s1paths):
                                    for j,(flow2,path2) in enumerate(s2paths):
                                        f = min(flow2,flow)
                                        if f>0:
                                            flow -=f
                                            flow2-=f
                                            s1paths[i]=(flow,path)
                                            s2paths[j]=(flow2,path2)
                                            
                                            assert(path[0]==sVirt)
                                            assert(path2[0]==sVirt)
                                            pathr = list(path)
                                            pathr.reverse()
                                            residual_path = pathr[:-1] + path2
                                            if log:
                                                log.write("path %s -> %s %d ="%(vm1, vm2, f *common_bw_factor))
                                                for u in residual_path:                                             
                                                    log.write(" " + u)
                                                log.write("\n")      
    
    if log:
        log.flush()                                            
        
    #printGraph(residual_network)            
    for i,( s1, s2, bandwidth) in enumerate(instance["PN"]):
        if changed[i]!=0:
            f = changed[i]
            sm,wt = sums[i]
            assert(wt>=0)
            assert(wt<=bandwidth)
            Assert(sm<=wt)  
            if residual_network:    
                expect = residual_network[s1][s2]['weight'] if  residual_network.has_edge(s1,s2) else 0
                if expect!=wt:
                    pass
            
                assert(expect==wt )
    
    #Core and ram requirements also need to be reduced to account for assignments.
    for host in instance["Servers"]:  
        assigned_cores=server_assigned_cores[host] 
        assigned_ram=server_assigned_ram[host]      
        assigned_storage=server_assigned_storage[host]  
        if assigned_cores>0:
            total_core_assignment[host]+=assigned_cores
            total_assigned_cores=total_core_assignment[host]            
            edges,cores,core_weights=server_core_assignments[host]
            assert(total_assigned_cores<=cores)
          
            AssertLessEqPB(edges,cores-total_assigned_cores,core_weights);
            
        if not ignore_ram_constraints and assigned_ram>0:

            total_ram_assignment[host]+=assigned_ram
            total_assigned_ram=total_ram_assignment[host]            
            edges,ram,ran_weights=server_ram_assignments[host]
            assert(total_assigned_ram<=ram)
            AssertLessEqPB(edges,ram-total_assigned_ram,ram_weights);

        if not ignore_storage_constraints and assigned_storage>0:

            total_storage_assignment[host]+=assigned_storage
            total_assigned_storage=total_storage_assignment[host]            
            edges,storage,storage_weights=server_storage_assignments[host]
            assert(total_assigned_storage<=storage)
            AssertLessEqPB(edges,storage-total_assigned_storage,storage_weights); 
    
    if min_cores:
        for host in instance["Servers"]:
            n_used = total_core_assignment[host]
            assumptions.append(used_server_cores[host]==n_used)
    
    if min_servers:
        for s in instance["Servers"]:
            used = used_servers[s]     
            if s in used_hosts:
                assumptions.append(used)
            else:
                assumptions.append(Not(used))

    if log:
        log.flush()
    #for (v,w,var,wt) in physical_graph.getAllEdges():
    #    #if  var in model:
    #    if var.value():
            
                
    #        #G.add_edge(physical_graph.names[v],physical_graph.names[w],weight=wt)
            

    """ print("graph{")
    for v,w,data in G.edges_iter(data=True):
        if v.startswith("virt_out_") or w.startswith("virt_out_"): 
            continue #don't print these
        if  v.startswith("virt_in_"):
            v = v[8:]
        if  w.startswith("virt_in_"):
            w = w[8:]
        #print("edge " + str(v) + "->" + str(w) + ":" + str(data))
        if 'weight' in data:
            print("%s--%s [label=%d]"%(v,w,data["weight"]))
        else:
            print("%s--%s"%(v,w))
        """
    
    
    #print("}")
    
    t = time.clock()
    elapsed_out+= t-et
    if log_time_file:
        timefile.write("%d %f %f\n"%( n_solutions,t-prev_time, t-start_time))
        timefile.flush()
    prev_time=t
    print("Solving..")
    
    #os.system("callgrind_control -i on")
total_time = time.clock()-start_time


if log:
    log.close()

if log_time_file:

    end_time= time.clock()
    timefile.write("done %d %f %f\n"%(n_solutions,end_time-prev_time, end_time-start_time))
    timefile.close()
print("Elapsed time: " + str(total_time) + " solving time: " + str(elapsed) + ", " + str(elapsed_out) + ", " + str(Monosat().elapsed_time) + ", " + str(PBManager().elapsed_time) + " (" + str(PBManager().import_time) +")")
print("Solved %d"%(n_solutions))
print("Done")

