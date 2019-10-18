#!/usr/bin/python3

import argparse
import datetime
import json
import statistics
import time
from gurobipy import *

trueList = ['true', 't', '1']

parser = argparse.ArgumentParser()
parser.add_argument("PhysicalNetwork", help="Path to the physical network file")
parser.add_argument("VirtualNetworkSequenceFile", help="Path to the virtual network sequence file")
parser.add_argument("IsUndirected", help="Flag to indicate whether physical network is undirected or not")
parser.add_argument("IsMultiThreaded", help="Flag to indicate whether Gurobi ILP solver can spawn multiple threads for solving")
parser.add_argument("OutputFile", help="File to stream timestamp output after each allocation is made")
args = parser.parse_args()

# Physical/virtual network file loading (custom simple files here for now)
start = datetime.datetime.now()
pn = json.load(open(args.PhysicalNetwork))
#vn = json.load(open(args.VirtualNetwork))
out = open(args.OutputFile, "w")
isUndirected = args.IsUndirected.lower() in trueList
isMultiThreaded = args.IsMultiThreaded.lower() in trueList

physicalServerResources = [{}, {}, {}]
for i in pn["Servers"].keys():
    physicalServerResources[0][i] = pn["Servers"][i][0]
    physicalServerResources[1][i] = pn["Servers"][i][1]
    physicalServerResources[2][i] = pn["Servers"][i][2]

# Build physical network from JSON input file
# arcs, capacity, cost, nodes variables from Gurobi netflow.py example
arcs = tuplelist()
indexArcs = tuplelist()
capacity = tupledict()
nodes = set()

if isUndirected:
    for i in pn["PN"]:
        arcs.append((i[0], i[1]))
        arcs.append((i[1], i[0]))
        indexArcs.append((i[0], i[1]))
        capacity[(i[0], i[1])] = i[2]
        nodes.add(i[0])
        nodes.add(i[1])

else:
    for i in pn["PN"]:
        arcs.append((i[0], i[1]))
        capacity[(i[0], i[1])] = i[2]
        nodes.add(i[0])
        nodes.add(i[1])

nodes = list(nodes)

counter = 0
allocationTimes = []

init = datetime.datetime.now()
delta = init - start
print("# instance {}".format(args.PhysicalNetwork), file=out)
print("# Settings: PN={}, VN={}, multi-thread={}".format(args.PhysicalNetwork, args.VirtualNetworkSequenceFile, isMultiThreaded), file=out)
print("init {} {}".format(delta.total_seconds(), delta.total_seconds()), file=out, flush=True)

with open(args.VirtualNetworkSequenceFile, "r") as ins:
    for line in ins:
        vn = json.load(open(line.rstrip('\r\n')))

        virtualPhysicalServerPairs = tuplelist([(i, j) for i in vn["VMs"].keys() for j in pn["Servers"].keys()])
        
        # Build the virtual server requirements here
        virtualServerCpuRequirements = tupledict()
        virtualServerRamRequirements = tupledict()
        virtualServerThirdRequirements = tupledict()

        for i in vn["VMs"].keys():
            for j in virtualPhysicalServerPairs.select(i, '*'):
                virtualServerCpuRequirements[j] = vn["VMs"][i][0]
                virtualServerRamRequirements[j] = vn["VMs"][i][1]
                virtualServerThirdRequirements[j] = vn["VMs"][i][2]

        # Build commodities (virtual source-destination pair tuples) and bandwidth requirements here
        commodities = tuplelist()
        bandwidthReqs = tupledict()
        for i in vn["VN"]:
            commodities.append((i[0], i[1]))
            bandwidthReqs[(i[0], i[1])] = i[2]

        m = Model('netsolver-gurobi')
        m.Params.OutputFlag = 0
        if isMultiThreaded:
            m.Params.Threads = 0
        else:
            m.Params.Threads = 1
        place = m.addVars(virtualPhysicalServerPairs, name="place", vtype=GRB.BINARY)

        # Add constraints to ensure that each virtual machine is mapped to a single physical server
        equal = m.addConstrs((place.sum(i, '*') == 1 for i in vn["VMs"].keys()), "equal")

        # Add constraints to ensure that virtual machine requirements are met by mappings to physical servers
        for i in pn["Servers"].keys():
            m.addConstr(virtualServerCpuRequirements.prod(place, '*', i) <= physicalServerResources[0][i], i + "-CpuConstraint")
            m.addConstr(virtualServerRamRequirements.prod(place, '*', i) <= physicalServerResources[1][i], i + "-RamConstraint")
            m.addConstr(virtualServerThirdRequirements.prod(place, '*', i) <= physicalServerResources[2][i], i + "-ThirdReqConstraint")

        # Create inflow dictionary for each source-dest pair (each commodity) here
        # - means more inflow than outflow (sink), + means more outflow than inflow (source)
        inflow = tupledict()
        for i in commodities:
            for j in nodes:
                if j not in pn["Servers"].keys():   # If j is a switch
                    inflow[(i, j)] = 0
                else:
                    inflow[(i, j)] = (bandwidthReqs[i] * place[(i[0], j)]) - (bandwidthReqs[i] * place[(i[1], j)])

        flow = m.addVars(commodities, arcs, name="flow", vtype=GRB.INTEGER, lb=0)
        m.setObjective(1)

        if isUndirected:
            cap = m.addConstrs((flow.sum('*', '*', i, j) + flow.sum('*', '*', j, i) <= capacity[i, j] for i, j in indexArcs), "cap")

        else:
            cap = m.addConstrs((flow.sum('*', '*', i, j) <= capacity[i, j] for i, j in arcs), "cap")

        m.addConstrs((flow.sum(i[0], i[1], '*', j) + inflow[(i, j)] == flow.sum(i[0], i[1], j, '*') for i in commodities for j in nodes), "node")

        m.optimize()
        sys.stdout.flush()
        
        if m.status == GRB.Status.OPTIMAL:
            currEnd = datetime.datetime.now()
            allocationTimes.append((currEnd - init).total_seconds())
            counter = counter + 1
            #print('Allocation %g\n' % (counter))
            print("{} {} {}".format(counter, (currEnd - init).total_seconds(), (currEnd - start).total_seconds()), file=out, flush=True)

            init = datetime.datetime.now()
            #print('%g seconds since script started\n' % (init - start))

            placeSolution = m.getAttr('x', place)
            flowSolution = m.getAttr('x', flow)

            for i in vn["VMs"].keys():
                for j in pn["Servers"].keys():
                    if placeSolution[i, j] == 1:
                        #print('%s -> %s\n' % (i, j))
                        physicalServerResources[0][j] = physicalServerResources[0][j] - virtualServerCpuRequirements[i, j]
                        physicalServerResources[1][j] = physicalServerResources[1][j] - virtualServerRamRequirements[i, j]
                        physicalServerResources[2][j] = physicalServerResources[2][j] - virtualServerThirdRequirements[i, j]
                        break

            if isUndirected:
                for i in commodities:
                    #print('For [%s, %s]:\n' % (i[0], i[1]))
                    for j, k in indexArcs:
                        capacity[j, k] = capacity[j, k] - flowSolution[i[0], i[1], j, k] - flowSolution[i[0], i[1], k, j]
                        #print('%s -> %s    -    %g\n' % (j, k, flowSolution[i[0], i[1], j, k]))
                        #print('%s -> $s    -    %g\n' % (k, j, flowSolution[i[0], i[1], k, j]))

            else:
                for i in commodities:
                    for j, k in arcs:
                        capacity[j, k] = capacity[j, k] - flowSolution[i[0], i[1], j, k]
                        #print('%s -> %s    -    %g\n' % (j, k, flowSolution[i[0], i[1], j, k]))

            sys.stdout.flush()
        else:
            break

end = datetime.datetime.now()
print("done {} {} {}".format(counter, (end - init).total_seconds(), (end - start).total_seconds()))
print((end - start).total_seconds())
print(counter)
print('Median time: %g s' % (statistics.median(allocationTimes)))
