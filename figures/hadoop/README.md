This is fig. 8 in current AIJ draft.

- secondnet folder contains results for the SecondNet algo
```
cd secondnet
./secondnet_hadoop.sh |& tee log.txt
```

- netsolver-smt folder contains results for the NetSolver-SMT algo
```
cd netsolver-smt
python run_hadoop_vdcmapper_config.py |& tee log.txt
```

- netsolver-ilp folder contains results for the NetSolver-ILP algo
```
cd netsolver-ilp
./run_exp_hadoop.sh |& tee log.txt
```
