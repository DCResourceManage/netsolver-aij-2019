#!/usr/bin/env python3
import os
import itertools
import tempfile
import subprocess
import sys
import re

os.environ['PYTHONHASHSEED']="1"
os.environ['PYTHONPATH'] = os.getcwd()+"/solvers/monosat_py/" 

# TODO(nodir): pass these variables as a runtime parameter
MAIN_PATH = '../../../'
CPU_TIME_LIMIT = 3600 # seconds
RAM_LIMIT = 80000 # 80 GB

if len(sys.argv)>1:
  modes=[sys.argv[1]]
  print("Running vdcmapper on modes " + str(modes))
  mode_name=sys.argv[1]
  
  mode_name=re.sub(r'[^a-z0-9]+', '-', mode_name).strip('-')
  mode_name = re.sub(r'[-]+', '-', mode_name)
  
  summary=open("fmcad13_vdcmapper_%s_summary.txt"%(mode_name),"w")
  runlim_log="fmcad13_vdcmapper_%s_runlim.txt"%(mode_name)
else:
  modes=["--intrinsic_edge_sets --no-rnd-theory-order --theory-order-vsids --rnd-theory-freq 0.99"]

  summary=open("fmcad13_vdcmapper_summary.txt","w")
  runlim_log="fmcad13_vdcmapper_runlim.txt"

runlim_tmp = tempfile.mktemp()

log = open(runlim_log,'w')
print("Logging results to " + runlim_log)
runlim_settings=" -t {} -s {} -o {}".format(CPU_TIME_LIMIT, RAM_LIMIT, runlim_tmp)
#Note: Secondnet, FMCAD, and VDCMapper all take different file formats, which is a pain.
#I have instrumented all three to output logs of the time required for each instance.
#instance_location="instances/fmcad_vdcmapper/"
instance_location=os.path.join(MAIN_PATH, "instances/fmcad/in_json/")
datacenters=["tree_200.20.pn","tree16_200.20.pn","tree_400.20.pn","tree16_400.20.pn","tree_2000.20.pn","tree16_2000.20.pn"]
#datacenters=["tree_200.20.pn","tree16_2000.20.pn"]
vdcs=["vn2_3.1.vn","vn2_3.2.vn","vn2_3.3.vn","vn2_5.1.vn","vn2_5.2.vn","vn2_5.3.vn"]

summary.write("#Solver\tDatacenter\tVDC\tNAllocated\tRuntime\n")
summary.flush()
if not os.path.exists("timelogs"):
    os.makedirs("timelogs")
for mode,datacenter,instance in itertools.product(modes,datacenters,vdcs):
  mode_name=mode
  mode_name=re.sub(r'[^a-z0-9]+', '-', mode_name).strip('-')
  mode_name = re.sub(r'[-]+', '-', mode_name)
  print( datacenter + " " + instance + " " + mode)
  #Using Armin Biere's runlim to limit runtimes, available from http://fmv.jku.at/runlim/
  instance_log="timelogs/fmcad13_vdcmapper_%s_datacenter%s_instance_%s.log"%(mode_name,datacenter,instance)
  if  os.path.exists(instance_log):
    os.remove(instance_log)
  command="runlim " + runlim_settings + " " + "python3 vdcmapper.py "  + " " + mode + " " + "--timelog=" + instance_log + " " + instance_location + datacenter + " " +instance_location + instance
  print("Running: " + command) 
  subprocess.call(command,shell=True,env=os.environ)
  #code = os.system(command)
  runtime=-1
  nvms=-1
  try:
    timefile = open(instance_log)
    for line in timefile:
      if line.startswith("#") or line.startswith("init"):
        continue
      if line.startswith("done"):
        #Get runtime here, including the time for the last, failed VDC allocation
        runtime=float( line.split()[3])
        continue
      else:
        nvms = max(nvms,int(line.split()[0]))
  except Exception as inst:
    print("Error reading " + instance_log)
    print(sys.exc_info()[0])
    break
  
  summary.write("vdcmapper"+mode_name + "\t" + datacenter + "\t" + instance + "\t" + str(nvms) + "\t" + str(runtime) +"\n")
  
  #log all runtimes from runlim, for double-checking purposes

  tmp = open(runlim_tmp)
  log.write(tmp.read())
  log.flush();  
  tmp.close()
  
  summary.flush()
summary.close()
log.close()
os.remove(runlim_tmp)
