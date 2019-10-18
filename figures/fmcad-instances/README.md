This is fig. 6 in AIJ paper.

- secondnet folder contains results for the SecondNet algo
```
cd secondnet
python run_fmcad13_secondnet.py |& tee log.txt
```

- Z3-AR folder contains results for the Z3-AR algo
```
cd Z3-AR
$ python run_fmcad13_z3.py |& tee log.txt
```

- netsolver-smt folder contains results for the NetSolver-SMT algo
```
cd netsolver-smt
python run_fmcad13_vdcmapper.py |& tee log.txt
```

- netsolver-ilp folder contains results for the NetSolver-ILP algo
```
cd netsolver-ilp
python run_exp_tree_inc.sh |& tee log.txt
```
