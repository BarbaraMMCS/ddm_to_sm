# cygnss_ddm_to_sm
CYGNSS L1 V3.2 data to Soil Moisture over Australia using AI

#### on HPC

* ssh aion-cluster (CPU)
* OR ssh iris-cluster (GPU)

* cd $SCRATCH/cygnss_ddm_to_sm
* git pull
* source venv/bin/activate
* python ....
*
*
* deactivate
* du -h -d 1 # folder size


#### from HPC to LOCAL
rsync --rsh='ssh -p 8022' -avzu bsymeon@aion-cluster:/scratch/users/bsymeon/cygnss_ddm_to_sm/data/... /Users/barbara.symeon/PycharmProjects/cygnss_ddm_to_sm/data/...
#### from LOCAL to HPC
rsync --rsh='ssh -p 8022' -avzu /Users/barbara.symeon/PycharmProjects/cygnss_ddm_to_sm/data/... bsymeon@aion-cluster:/scratch/users/bsymeon/cygnss_ddm_to_sm/data/...



