# Difference between `*.pn` and `hadoop_extra/*.pn` files

Servers and switches in hadoop folder have more capacity (constant multiplier).

# Difference between `./hadoop_*_*bw.vn` and `./hadoop_extra/hadoop_*.*.vn` files

VMs in `./hadoop_*_*bw.vn` have more capacity (constant multiplier).

# Instances for extensions (servermin, hotspot, softaffinity)

Softaffinity instances have `affinity` and `antiaffinity` fields that other
hadoop VNs do not have. These files were created as follows
```
cd aij_repro
mkdir experiments-repoduced/instances/hadoop/hadoop_extensions
cd experiments-repoduced/instances/hadoop/hadoop_extensions
cp ../../../../aij_experiments/netsolver-scripts/aij_experiments/instances/hadoop_extended_small/hadoop_*vn .
ln -s ../hadoop_extra/us-mid1.pn .
ln -s ../hadoop_extra/us-mid2.pn .
ln -s ../hadoop_extra/us-west1.pn .
ln -s ../hadoop_extra/us-west2.pn .
```
