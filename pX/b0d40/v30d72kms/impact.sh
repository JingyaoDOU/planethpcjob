#!/bin/bash
#SBATCH -J PLANETimpact_0d0h_0d99640_npt222272_1d99281_v30d7200kms_b0d400_pX
#SBATCH --account=phys006361
#SBATCH --partition=compute
#SBATCH --exclude=bp1-compute[227-228]
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --mem=100G


module purge
module restore swift_intel2020_basic
module load system/slurm/21.08.0

cp ./fake_parameters_file.yml .

~/yq/yq_linux_386 -i '.InitialConditions.file_name="PLANETimpact_0d0h_0d99640_npt222272_1d99281_v30d7200kms_b0d400_pX.hdf5"' parameters_impact.yml
~/yq/yq_linux_386 -i ".SPH.h_max="0.2"" parameters_impact.yml
~/yq/yq_linux_386 -i ".TimeIntegration.dt_max="5.0"" parameters_impact.yml
~/yq/yq_linux_386 -i ".TimeIntegration.time_end="108000"" parameters_impact.yml
~/yq/yq_linux_386 -i ".SPH.CFL_condition="0.2"" parameters_impact.yml
~/yq/yq_linux_386 -i ".Snapshots.delta_time="36000"" parameters_impact.yml
~/yq/yq_linux_386 -i '.Snapshots.subdir="output_30h_36000dt"' parameters_impact.yml
~/yq/yq_linux_386 -i '.EoS.planetary_ANEOS_iron_table_file="/user/home/qb20321/hammer/toolkit/EiEf/ANEOS_iron_S20.txt.40gccdensedome"' parameters_impact.yml


cat parameters_impact.yml >> ./logput_PLANETimpact_0d0h_0d99640_npt222272_1d99281_v30d7200kms_b0d400_pX.txt

/user/home/qb20321/hammer/toolkit/swift_subtask -a -s -G -t 32 ./parameters_impact.yml 2>&1 | tee -a ./logput_PLANETimpact_0d0h_0d99640_npt222272_1d99281_v30d7200kms_b0d400_pX.txt

cp output_30h_36000dt/snapshot_0003.hdf5 snapOUT_PLANETimpact_0d0h_0d99640_npt222272_1d99281_v30d7200kms_b0d400_pX.hdf5


rm *.yml
rm *csv
rm statistics.txt
rm task*
rm timesteps*
rm unused*
rm used*
rm *.out

