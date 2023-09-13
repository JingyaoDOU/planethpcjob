TMPL = """\
#!/bin/bash
#SBATCH -J {name}
#SBATCH --account=phys006361
{header}
{bash_setup}

module purge
module restore swift_intel2020_basic
module load system/slurm/21.08.0

cp {para_impact} .

~/yq/yq_linux_386 -i '.InitialConditions.file_name="{name}.hdf5"' parameters_impact.yml
~/yq/yq_linux_386 -i ".SPH.h_max="{hmax}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".TimeIntegration.dt_max="{dtmax}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".TimeIntegration.time_end="{time_end}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".SPH.CFL_condition="{CFL}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".Snapshots.delta_time="{delta_time}"" parameters_impact.yml
~/yq/yq_linux_386 -i '.Snapshots.subdir="{outdir}"' parameters_impact.yml
~/yq/yq_linux_386 -i '.EoS.planetary_ANEOS_iron_table_file="/user/home/qb20321/hammer/toolkit/EiEf/ANEOS_iron_S20.txt.40gccdensedome"' parameters_impact.yml


cat parameters_impact.yml >> ./logput_{name}.txt

/user/home/qb20321/hammer/toolkit/{swift_exe} -a -s -G -t {thread} ./parameters_impact.yml 2>&1 | tee -a ./logput_{name}.txt

cp {outdir}/snapshot_{outnum}.hdf5 snapOUT_{name}.hdf5


rm *.yml
rm *csv
rm statistics.txt
rm task*
rm timesteps*
rm unused*
rm used*
rm *.out

"""

TMPL_bc = """\

module purge
module restore swift_intel2020_basic

cp {para_impact} .

~/yq/yq_linux_386 -i '.InitialConditions.file_name="{name}.hdf5"' parameters_impact.yml
~/yq/yq_linux_386 -i ".SPH.h_max="{hmax}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".TimeIntegration.dt_max="{dtmax}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".TimeIntegration.time_end="{time_end}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".SPH.CFL_condition="{CFL}"" parameters_impact.yml
~/yq/yq_linux_386 -i ".Snapshots.delta_time="{delta_time}"" parameters_impact.yml
~/yq/yq_linux_386 -i '.Snapshots.subdir="{outdir}"' parameters_impact.yml

cat parameters_impact.yml >> ./logput_{name}.txt

/user/home/qb20321/work/toolkit/{swift_exe} -a -s -G -t {thread} ./parameters_impact.yml 2>&1 | tee -a ./logput_{name}.txt

cp {outdir}/snapshot_{outnum}.hdf5 snapOUT_{name}.hdf5


rm *.yml
rm *csv
rm statistics.txt
rm task*
rm timesteps*
rm unused*
rm used*
rm *.out

"""
