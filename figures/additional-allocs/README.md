This is fig. 10 in current AIJ draft.

- secondnet folder contains results for the SecondNet algo
```
cd secondnet
./secondnet_print_final_pn.sh |& tee log.txt
mkdir final_pns_json
cd converter
python json_converter.py -i ../final_pns -o ../final_pns_json
```


- netsolver-smt folder contains results for the NetSolver-SMT algo
```
cd netsolver-smt
python run_bcube_fattree_vdcmapper_config_additional.py |& tee log.txt
```

- netsolver-ilp folder contains results for the NetSolver-ILP algo
```
cd netsolver-ilp
./run_exp_bcube.sh |& tee log.txt
```
