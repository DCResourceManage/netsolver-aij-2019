This is fig. 7 in current AIJ draft.

- secondnet folder contains results for the SecondNet algo
```
cd secondnet
./secondnet_bcube_fatree.sh |& tee log.txt
```

- netsolver-smt folder contains results for the NetSolver-SMT algo
```
cd netsolver-smt
python run_bcube_fattree_vdcmapper_config.py |& tee log.txt
```

- netsolver-ilp folder contains results for the NetSolver-ILP algo that uses
single core and has 3600 CPU time limit (`runlim -t` option).
```
cd netsolver-ilp
./run_exp_bcube_inc.sh |& tee log.txt
```

# ILP multi core experiments

These are datapoints for footnote 10 reported in a paper.

## Results

Here we compare single-core CPU-time limit results (`sc`, reported in the paper) to multi-core results (`mc`) with 1 hour wall-clock time limit (`MCR`), and 1 hour CPU-time limit (`MCT`). `total_allocs` contains the number of total VDC allocations in this datacenter with VDC size of (6VMs, 9VMs, 12VMs, 15VMs), respectively. Similarly, `total_time` contains the median VDC allocation time for (6VMs, 9VMs, 12VMs, 15VMs), respectively. All VDC allocation times are reported by Gurobi and are in CPU-time (not wall-clock time).


### MCR

python plot_aij.py -mc
FatTree total_allocs = {'sc': [333, 215, 148, 127], 'mc': [334, 222, 157, 124]}
FatTree total_time = {'sc': [0.3495, 0.5656, 0.9198, 1.3402], 'mc': [0.3726, 0.5645, 0.9537, 1.1226]}
FatTree total_allocs = {'sc': [1152, 768, 562, 453], 'mc': [1152, 768, 557, 457]}
FatTree total_time = {'sc': [1.271, 1.8475, 3.1717, 4.6444], 'mc': [1.3108, 1.969, 3.2749, 4.6199]}
FatTree total_allocs = {'sc': [1274, 779, 485, 327], 'mc': [1250, 764, 453, 377]}
FatTree total_time = {'sc': [2.8897, 4.6435, 7.5047, 10.7506], 'mc': [2.9766, 4.8213, 8.0329, 9.3821]}
BCube total_allocs = {'sc': [1365, 910, 654, 530], 'mc': [1311, 910, 493, 545]}
BCube total_time = {'sc': [1.4517, 2.2494, 3.7779, 5.2845], 'mc': [1.3147, 2.3326, 3.3887, 5.4776]}
BCube total_allocs = {'sc': [1258, 801, 491, 345], 'mc': [1193, 862, 558, 318]}
BCube total_time = {'sc': [2.9684, 4.5809, 7.3894, 10.1686], 'mc': [3.1653, 4.2797, 6.4759, 10.9473]}
BCube total_allocs = {'sc': [719, 466, 278, 191], 'mc': [776, 503, 310, 203]}
BCube total_time = {'sc': [5.2339, 7.8971, 13.0326, 18.4218], 'mc': [4.713, 7.417, 11.615, 17.0881]}


### MCT

FatTree total_allocs = {'sc': [333, 215, 148, 127], 'mc': [337, 214, 155, 131]}
FatTree total_time = {'sc': [0.3495, 0.5656, 0.9198, 1.3402], 'mc': [0.3775, 0.5666, 0.8956, 1.3594]}
FatTree total_allocs = {'sc': [1152, 768, 562, 453], 'mc': [1152, 768, 558, 455]}
FatTree total_time = {'sc': [1.271, 1.8475, 3.1717, 4.6444], 'mc': [1.2799, 1.9217, 3.3861, 4.6138]}
FatTree total_allocs = {'sc': [1274, 779, 485, 327], 'mc': [1179, 677, 431, 297]}
FatTree total_time = {'sc': [2.8897, 4.6435, 7.5047, 10.7506], 'mc': [3.0677, 4.9777, 7.6906, 10.8706]}
BCube total_allocs = {'sc': [1365, 910, 654, 530], 'mc': [1365, 910, 493, 509]}
BCube total_time = {'sc': [1.4517, 2.2494, 3.7779, 5.2845], 'mc': [1.5166, 2.3957, 3.2074, 5.4257]}
BCube total_allocs = {'sc': [1258, 801, 491, 345], 'mc': [1378, 687, 563, 321]}
BCube total_time = {'sc': [2.9684, 4.5809, 7.3894, 10.1686], 'mc': [2.6673, 4.871, 6.4065, 10.5407]}
BCube total_allocs = {'sc': [719, 466, 278, 191], 'mc': [626, 506, 305, 208]}
BCube total_time = {'sc': [5.2339, 7.8971, 13.0326, 18.4218], 'mc': [5.595, 7.3779, 11.7667, 16.4966]}


## How to reproduce these results

- netsolver-ilp-mcr (multi core real) contains ILP results where we let ILP
solver to use multiple cores but we limit the total execution time to 3600 seconds
real-time (wall clock time, `runlim -r` option).
```
cd netsolver-ilp-mcr
./run_exp_bcube_mcr.sh |& tee log.txt
```

- netsolver-ilp-mct (multi core time) contains ILP results where we let ILP
solver to use multiple cores but we limit the total execution time to
3600 CPU seconds (`runlim -t` option).
```
cd netsolver-ilp-mct
./run_exp_bcube_mct.sh |& tee log.txt
```
