from __future__ import print_function

import sys
import os
import subprocess
import tempfile
import atexit
import hashlib
import datetime
import h5py
import woma
import swiftsimio as sw
import unyt
import numpy as np
from pathlib import Path
from templates import TMPL

global G, M_earth, R_earth

G = 6.67408e-11  # m^3 kg^-1 s^-2
M_earth = 5.97240e24  # kg
R_earth = 6.371e6  # m


def loadsw_to_woma(snapshot, unit="mks", npR=100, atmosphere_matid=200):
    # Load
    data = sw.load(snapshot)
    if unit == "mks":
        box_mid = 0.5 * data.metadata.boxsize[0].to(unyt.m)
        data.gas.coordinates.convert_to_mks()
        pos = np.array(data.gas.coordinates - box_mid)
        data.gas.velocities.convert_to_mks()
        vel = np.array(data.gas.velocities)
        data.gas.smoothing_lengths.convert_to_mks()
        h = np.array(data.gas.smoothing_lengths)
        data.gas.masses.convert_to_mks()
        m = np.array(data.gas.masses)
        data.gas.densities.convert_to_mks()
        rho = np.array(data.gas.densities)
        data.gas.pressures.convert_to_mks()
        p = np.array(data.gas.pressures)
        data.gas.internal_energies.convert_to_mks()
        u = np.array(data.gas.internal_energies)
        matid = np.array(data.gas.material_ids)
        # pid     = np.array(data.gas.particle_ids)

    elif unit == "cgs":
        box_mid = 0.5 * data.metadata.boxsize[0].to(unyt.cm)
        data.gas.coordinates.convert_to_cgs()
        pos = np.array(data.gas.coordinates - box_mid)
        data.gas.velocities.convert_to_cgs()
        vel = np.array(data.gas.velocities)
        data.gas.smoothing_lengths.convert_to_cgs()
        h = np.array(data.gas.smoothing_lengths)
        data.gas.masses.convert_to_cgs()
        m = np.array(data.gas.masses)
        data.gas.densities.convert_to_cgs()
        rho = np.array(data.gas.densities)
        data.gas.pressures.convert_to_cgs()
        p = np.array(data.gas.pressures)
        data.gas.internal_energies.convert_to_cgs()
        u = np.array(data.gas.internal_energies)
        matid = np.array(data.gas.material_ids)
        # pid     = np.array(data.gas.particle_ids)
    else:
        raise TypeError("Wrong unit selection, please check!!")

    pos_centerM = np.sum(pos * m[:, np.newaxis], axis=0) / np.sum(m)
    vel_centerM = np.sum(vel * m[:, np.newaxis], axis=0) / np.sum(m)

    pos -= pos_centerM
    vel -= vel_centerM

    pos_noatmo = pos[matid != atmosphere_matid]

    xy = np.hypot(pos_noatmo[:, 0], pos_noatmo[:, 1])
    r = np.hypot(xy, pos_noatmo[:, 2])
    r = np.sort(r)
    R = np.mean(r[-npR:])

    return pos, vel, h, m, rho, p, u, matid, R


def tmp(suffix=".sh"):
    t = tempfile.mktemp(suffix=suffix)
    atexit.register(os.unlink, t)
    return t


class hpcjob:
    def __init__(
        self,
        name,
        slurm_kwargs=None,
        tmpl=TMPL,
        scripts_dir="slurm-scripts",
        log_dir="logs",
        bash_strict=True,
        hmax=0.1,
        para_impact=None,
        dtmax=5,
        CFL=0.1,
        swift_exe=None,
        time_end=36000,  # in seconds
        delta_time=100,
        thread=32,
    ):
        if slurm_kwargs is None:
            slurm_kwargs = {}
        if tmpl is None:
            tmpl = TMPL
        self.log_dir = log_dir
        self.bash_strict = bash_strict

        header = []
        if "time" not in slurm_kwargs.keys():
            slurm_kwargs["time"] = "24:00:00"
        if "mem" not in slurm_kwargs.keys():
            slurm_kwargs["mem"] = "80GB"

        for k, v in slurm_kwargs.items():
            if len(k) > 1:
                k = "--" + k + "="
            else:
                k = "-" + k + " "
            header.append("#SBATCH %s%s" % (k, v))

        # add bash setup list to collect bash script config
        bash_setup = []
        if bash_strict:
            bash_setup.append("set -eo pipefail -o nounset")

        self.header = "\n".join(header)
        self.bash_setup = "\n".join(bash_setup)
        self.name = name.replace(" ", "_")
        self.tmpl = tmpl
        self.slurm_kwargs = slurm_kwargs
        if scripts_dir is not None:
            self.scripts_dir = os.path.abspath(scripts_dir)
        else:
            self.scripts_dir = None

        self.hmax = hmax
        self.para_impact = para_impact
        self.dtmax = dtmax
        self.CFL = CFL
        self.swift_exe = swift_exe
        self.time_end = time_end
        self.delta_time = delta_time
        self.outdir = "output_%dh_%ddt" % (self.time_end / 3600, self.delta_time)
        self.outnum = "%04d" % (self.time_end / self.delta_time)
        self.thread = thread

    def __str__(self):
        return self.tmpl.format(
            name=self.name,
            header=self.header,
            log_dir=self.log_dir,
            bash_setup=self.bash_setup,
            hmax=self.hmax,
            para_impact=self.para_impact,
            dtmax=self.dtmax,
            CFL=self.CFL,
            swift_exe=self.swift_exe,
            time_end=self.time_end,  # convert to seconds
            delta_time=self.delta_time,
            outdir=self.outdir,
            outnum=self.outnum,
            thread=self.thread,
        )


def setup_impact(
    loc_tar,
    loc_imp,
    save_folder,
    parameter_loc=None,
    tmpl=TMPL,
    b_value=0.0,
    v_value=0.0,
    mX=False,
    mZ=False,
    ifspin=False,
    period=0.0,
    partition_name="test",
    ncpu=16,
    exclude=None,
    test=False,
    hmax=0.05,
    dtmax=5,
    CFL=0.1,
    swift_exe=None,
    time_end=36000,
    delta_time=1000,
    time="0-08:00:00",
    mem="30G",
    boxsize=1000,
    threads=32,
    chmod=False,
    verbose=False,
):
    """_summary_

    Args:
        loc_tar (str): path to the target snapshot
        loc_imp (str): path to the impactor snapshot
        save_folder (str): path of folder to store the impact simulation.
        parameter_loc (str, optional): Path to the parameter file. Defaults to None.
        tmpl (_type_, optional): template of the slurm script. Defaults to TMPL.
        b_value (float, optional): Impact parameters. Defaults to 0.0.
        v_value (float, optional): Impact absolute velocity in km/s. Defaults to 0.0.
        mX (bool, optional): If impactor coming from minus -X direction. Defaults to False.
        mZ (bool, optional): If it's polar impact. Defaults to False.
        period (float, optional): Spinning period of the target. Defaults to 0.0.
        partition_name (str, optional): Name of the parition. Defaults to "test".
        ncpu (int, optional): number of cores request. Defaults to 16.
        exclude (_type_, optional): Which node I would like not to request. Defaults to None.
        hmax (float, optional): Maximux smoothing length. Defaults to 0.05.
        dtmax (int, optional): Maximux time step. Defaults to 5.
        CFL (float, optional): Courant–Friedrichs–Lewy conditon. Defaults to 0.1.
        swift_exe (_type_, optional): Which SWIFT executable I would like to use. Defaults to None.
        time_end (int, optional): Simulation time in second. Defaults to 36000.
        delta_time (int, optional): Output frequency in second. Defaults to 1000.
        mem (str, optional): Memory requested. Defaults to "80G".
        boxsize (int, optional): Size of box in R_earh. Defaults to 1000.
        threads (int, optional): Number of threads to use. Defaults to 32.
        chmod (bool, optional): If change "impact.sh" to runable. Defaults to False.
        verbose (bool, optional): If print out the slurm script when test is true. Defaults to False.

    Raises:
        TypeError: _description_
        TypeError: _description_
    """
    if not os.path.exists(parameter_loc):
        raise FileNotFoundError(f"The path '{parameter_loc}' does not exist.")

    (
        pos_tar,
        vel_tar,
        h_tar,
        m_tar,
        rho_tar,
        p_tar,
        u_tar,
        matid_tar,
        R_tar,
    ) = loadsw_to_woma(loc_tar, unit="mks")
    (
        pos_imp,
        vel_imp,
        h_imp,
        m_imp,
        rho_imp,
        p_imp,
        u_imp,
        matid_imp,
        R_imp,
    ) = loadsw_to_woma(loc_imp, unit="mks")

    M_t = np.sum(m_tar)
    M_i = np.sum(m_imp)
    R_t = R_tar
    R_i = R_imp

    m_tot = (M_t + M_i) / M_earth
    m_TAR = M_t / M_earth
    npt = len(pos_tar)

    # Mutual escape speed
    # v_esc = np.sqrt(2 * G * (M_t + M_i) / (R_t + R_i))

    # Initial position and velocity of the target
    A1_pos_t = np.array([0.0, 0.0, 0.0])
    A1_vel_t = np.array([0.0, 0.0, 0.0])

    A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_t(
        b=b_value,
        v_c=v_value * 1000,  # m/s
        t=3600,
        R_t=R_t,
        R_i=R_i,
        M_t=M_t,
        M_i=M_i,
    )

    if mX:
        A1_pos_i[0] = -A1_pos_i[0]
        A1_vel_i[0] = -A1_vel_i[0]

    if mZ:
        # hit in -z direction
        A1_pos_i[2] = A1_pos_i[0]
        A1_vel_i[2] = A1_vel_i[0]
        A1_pos_i[0] = 0.0
        A1_vel_i[0] = 0.0

    if mX and mZ:
        raise TypeError("WRONG dirction!!!")

    A1_pos_com = (M_t * A1_pos_t + M_i * A1_pos_i) / (M_t + M_i)
    A1_pos_t -= A1_pos_com
    A1_pos_i -= A1_pos_com

    # Centre of momentum
    A1_vel_com = (M_t * A1_vel_t + M_i * A1_vel_i) / (M_t + M_i)
    A1_vel_t -= A1_vel_com
    A1_vel_i -= A1_vel_com

    pos_tar += A1_pos_t
    vel_tar[:] += A1_vel_t

    pos_imp += A1_pos_i
    vel_imp[:] += A1_vel_i
    # entropy =  woma.eos.eos.A1_s_u_rho(np.append(u_tar, u_imp), np.append(rho_tar, rho_imp), np.append(matid_tar, matid_imp))

    if ifspin:
        head = "SPINimpact"
    else:
        head = "PLANETimpact"

    if mX:
        direction = "mX"
    elif mZ:
        direction = "mZ"
    else:
        direction = "pX"

    filename = "_".join(
        [
            head,
            str(period).replace(".", "d") + "h",
            str("%.5f" % m_TAR).replace(".", "d"),
            "npt%d" % npt,
            str("%.5f" % m_tot).replace(".", "d"),
            ("v%.4fkms" % v_value).replace(".", "d"),
            ("b%.3f" % b_value).replace(".", "d"),
            direction,
        ]
    )
    filename = filename + ".hdf5"

    if exclude is not None:
        s = hpcjob(
            "%s" % filename.split(".")[0],
            {
                "partition": partition_name,
                "exclude": "bp1-compute%s" % exclude,
                "cpus-per-task": "%d" % ncpu,
                "tasks-per-node": "1",
                "nodes": "1",
                "time": time,
                "mem": mem,
            },
            tmpl=tmpl,
            bash_strict=False,
            hmax=hmax,
            para_impact=parameter_loc,
            dtmax=dtmax,
            CFL=CFL,
            swift_exe=swift_exe,
            time_end=time_end,
            delta_time=delta_time,
            thread=threads,
        )
    else:
        s = hpcjob(
            "%s" % filename.split(".")[0],
            {
                "partition": partition_name,
                "cpus-per-task": "%d" % ncpu,
                "tasks-per-node": "1",
                "nodes": "1",
                "time": "0-08:00:00",
                "mem": "30G",
            },
            tmpl=tmpl,
            bash_strict=False,
            hmax=hmax,
            para_impact=parameter_loc,
            dtmax=dtmax,
            CFL=CFL,
            swift_exe=swift_exe,
            time_end=time_end,
            delta_time=delta_time,
            thread=threads,
        )
    # creat the dir for each sim
    if period == 0:
        saveloc = save_folder + "/%s/b%s/v%skms/" % (
            direction,
            str("%.2f" % b_value).replace(".", "d"),
            str("%.2f" % v_value).replace(".", "d"),
        )
    else:
        saveloc = save_folder + "/%s/h%s/b%s/v%skms/" % (
            direction,
            str("%.1f" % period).replace(".", "d"),
            str("%.2f" % b_value).replace(".", "d"),
            str("%.2f" % v_value).replace(".", "d"),
        )

    Path(saveloc).mkdir(parents=True, exist_ok=True)

    with open(saveloc + "impact.sh", "w") as f:
        f.write(str(s))

    # print(file)
    file = saveloc + filename
    with h5py.File(file, "w") as f:
        woma.save_particle_data(
            f,
            np.append(pos_tar, pos_imp, axis=0),
            np.append(vel_tar, vel_imp, axis=0),
            np.append(m_tar, m_imp),
            np.append(h_tar, h_imp),
            np.append(rho_tar, rho_imp),
            np.append(p_tar, p_imp),
            np.append(u_tar, u_imp),
            np.append(matid_tar, matid_imp),
            boxsize=boxsize * R_earth,
            file_to_SI=woma.Conversions(M_earth, R_earth, 1),
            verbosity=0,
        )
    if not test:
        os.chdir(saveloc)
        os.system("sbatch impact.sh")
    else:
        if verbose:
            print(file)
            print(str(s))
    if chmod:
        os.chdir(saveloc)
        os.system("chmod u+x impact.sh")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
