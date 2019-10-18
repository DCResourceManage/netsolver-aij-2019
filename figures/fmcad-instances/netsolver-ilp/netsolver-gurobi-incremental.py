#!/usr/bin/python3

import argparse
import datetime
import json
import numpy as np
import statistics
import time
from gurobipy import *

trueList = ['true', 't', '1']

parser = argparse.ArgumentParser()
parser.add_argument("PhysicalNetwork", help="Path to the physical network file")
parser.add_argument("VirtualNetworkSequenceFile", help="Path to the virtual network sequence file")
parser.add_argument("IsUndirected", help="Flag to indicate whether physical network is undirected or not")
parser.add_argument("MaxVMs", help="The maximum number of VMs we expect to see in any VDC topology", type=int)
parser.add_argument("IsMultiThreaded", help="Flag to indicate whether Gurobi ILP solver can spawn multiple threads for solving")
parser.add_argument("OutputFile", help="File to stream timestamp output to after each allocation is made",type=str)
args = parser.parse_args()

start = datetime.datetime.now()
pn = json.load(open(args.PhysicalNetwork))
maxVMs = args.MaxVMs
out = open(args.OutputFile, "w")
isUndirected = args.IsUndirected.lower() in trueList
isMultiThreaded = args.IsMultiThreaded.lower() in trueList

physicalServerResources = [{}, {}, {}]
for i in pn["Servers"].keys():
    physicalServerResources[0][i] = pn["Servers"][i][0]
    physicalServerResources[1][i] = pn["Servers"][i][1]
    physicalServerResources[2][i] = pn["Servers"][i][2]

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

virtualPhysicalServerPairs = tuplelist([(i, j) for i in range(0, maxVMs) for j in pn["Servers"].keys()])

m = Model('netsolver-gurobi')
m.Params.OutputFlag = 0
if isMultiThreaded:
    m.Params.Threads = 0
else:
    m.Params.Threads = 1
place = m.addVars(virtualPhysicalServerPairs, name="place", vtype=GRB.BINARY)
equal = m.addConstrs((place.sum(i, '*') == 0 for i in range(0, maxVMs)), "equal")

virtualServerCpuRequirements = tupledict()
virtualServerRamRequirements = tupledict()
virtualServerThirdRequirements = tupledict()

for i in range(0, maxVMs):
    for j in virtualPhysicalServerPairs.select(i, '*'):
        virtualServerCpuRequirements[j] = 0
        virtualServerRamRequirements[j] = 0
        virtualServerThirdRequirements[j] = 0

cpuConstrs = {}
ramConstrs = {}
thirdReqConstrs = {}
for i in pn["Servers"].keys():
    cpuConstrs[i] = m.addConstr(virtualServerCpuRequirements.prod(place, '*', i) <= physicalServerResources[0][i], i + "-CpuConstraint")
    ramConstrs[i] = m.addConstr(virtualServerRamRequirements.prod(place, '*', i) <= physicalServerResources[1][i], i + "-RamConstraint")
    thirdReqConstrs[i] = m.addConstr(virtualServerThirdRequirements.prod(place, '*', i) <= physicalServerResources[2][i], i + "-ThirdReqConstraint")

commodities = [i for i in range(0, maxVMs)]

flow = m.addVars(commodities, arcs, name="flow", vtype=GRB.INTEGER, lb=0)

if isUndirected:
    cap = m.addConstrs((flow.sum('*', i, j) + flow.sum('*', j, i) <= capacity[i, j] for i, j in indexArcs), "cap")
else:
    cap = m.addConstrs((flow.sum('*', i, j) <= capacity[i, j] for i, j in arcs), "cap")

bandwidthReqs = np.zeros((maxVMs, maxVMs))
totalFlowsFromSource = np.sum(bandwidthReqs, axis=1)

inflow = tupledict()
for i in commodities:
    for j in nodes:
        if j not in pn["Servers"].keys():
            inflow[(i, j)] = 0
        else:
            inflow[(i, j)] = (totalFlowsFromSource[i] * place[(i, j)])
            for k in range(0, maxVMs):
                inflow[(i, j)] = inflow[(i, j)] - (bandwidthReqs[i, k] * place[(k, j)])

node = m.addConstrs((flow.sum(i, j, '*') - flow.sum(i, '*', j) - inflow[(i, j)] == 0 for i in commodities for j in nodes), "node")

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
        index = 0
        indexToVMMap = {}

        for i in vn["VMs"].keys():
            indexToVMMap[index] = i
            index = index + 1

        VMToIndexMap = dict((v, k) for k, v in indexToVMMap.items())

        for i in range(0, maxVMs):
            if i < index:
                equal[i].setAttr(GRB.Attr.RHS, 1)
                for j in virtualPhysicalServerPairs.select(i, '*'):
                    virtualServerCpuRequirements[j] = vn["VMs"][indexToVMMap[i]][0]
                    virtualServerRamRequirements[j] = vn["VMs"][indexToVMMap[i]][1]
                    virtualServerThirdRequirements[j] = vn["VMs"][indexToVMMap[i]][2]            
            else:
                equal[i].setAttr(GRB.Attr.RHS, 0)
                for j in virtualPhysicalServerPairs.select(i, '*'):
                    virtualServerCpuRequirements[j] = 0
                    virtualServerRamRequirements[j] = 0
                    virtualServerThirdRequirements[j] = 0

        for i in pn["Servers"].keys():
            for j in range(0, maxVMs):
                m.chgCoeff(cpuConstrs[i], place[j, i], virtualServerCpuRequirements[j, i])
                m.chgCoeff(ramConstrs[i], place[j, i], virtualServerRamRequirements[j, i])
                m.chgCoeff(thirdReqConstrs[i], place[j, i], virtualServerThirdRequirements[j, i])

        bandwidthReqs = np.zeros((maxVMs, maxVMs))
        for i in vn["VN"]:
            bandwidthReqs[VMToIndexMap[i[0]]][VMToIndexMap[i[1]]] = i[2]

        totalFlowsFromSource = np.sum(bandwidthReqs, axis=1)

        for i in commodities:
            for j in nodes:
                if j in pn["Servers"].keys():
                    for k in range(0, maxVMs):
                        m.chgCoeff(node[(i, j)], place[(k, j)], bandwidthReqs[i][k])
                    m.chgCoeff(node[(i, j)], place[(i, j)], (totalFlowsFromSource[i] * -1))

        m.reset()
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

            for i in range(0, maxVMs):
                for j in pn["Servers"].keys():
                    if placeSolution[i, j] == 1:
                        physicalServerResources[0][j] = physicalServerResources[0][j] - virtualServerCpuRequirements[i, j]
                        physicalServerResources[1][j] = physicalServerResources[1][j] - virtualServerRamRequirements[i, j]
                        physicalServerResources[2][j] = physicalServerResources[2][j] - virtualServerThirdRequirements[i, j]
                        cpuConstrs[j].setAttr(GRB.Attr.RHS, physicalServerResources[0][j])
                        ramConstrs[j].setAttr(GRB.Attr.RHS, physicalServerResources[1][j])
                        thirdReqConstrs[j].setAttr(GRB.Attr.RHS, physicalServerResources[2][j])
                        break

            if isUndirected:
                for i in commodities:
                    for j, k in indexArcs:
                        capacity[j, k] = capacity[j, k] - flowSolution[i, j, k] - flowSolution[i, k, j]

                for i, j in indexArcs:
                    cap[i, j].setAttr(GRB.Attr.RHS, capacity[i, j])

            else:
                for i in commodities:
                    for j, k in arcs:
                        capacity[j, k] = capacity[j, k] - flowSolution[i, j, k]

                for i, j in arcs:
                    cap[i, j].setAttr(GRB.Attr.RHS, capacity[i, j])

            sys.stdout.flush()
        else:
            break

end = datetime.datetime.now()
print("done {} {} {}".format(counter, (end - init).total_seconds(), (end - start).total_seconds()), file=out, flush=True)
print((end - start).total_seconds())
print(counter)
print('Median time: %g s' % (statistics.median(allocationTimes)))
