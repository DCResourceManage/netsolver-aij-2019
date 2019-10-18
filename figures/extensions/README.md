This is fig. 12 in current AIJ draft.

- netsolver-ilp folder contains results for the NetSolver-ILP algo
```
cd netsolver-ilp
./run_extended_gurobi.sh |& tee log.txt
```

- netsolver-smt folder contains results for the NetSolver-SMT algo
```
cd netsolver-smt
python run_hadoop_vdcmapper_config.py |& tee log.txt
```

