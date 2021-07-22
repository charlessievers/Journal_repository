import numpy as np
from mpi4py import MPI
from lammps import lammps
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":

    # data files
    infile = "algmg.lmp"
    ff_file = "../ff.lmp"

    num_pico_s = 4500

    # full output useful for testing
    # lmp = lammps()

    # reduced output useful reducing IO for production runs
    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])

    # lammps commands
    lmp.command("atom_style full")
    lmp.command("units metal")
    lmp.command("processors * * *")
    lmp.command("neighbor 1 bin")
    lmp.command("boundary p p p")
    # 1 2 3 4 5 11/12 11/12 6 7 8 9 10
    # read data and force field file
    lmp.command("read_data {}".format(infile))
    lmp.file("{}".format(ff_file))
    lmp.command("region ledge  block -100 100 -100 100 0  2 ")
    lmp.command("region left   block -100 100 -100 100 8  13")
    lmp.command("region lal    block -100 100 -100 100 13 21")
    lmp.command("region ral    block -100 100 -100 100 37 43")
    lmp.command("region right  block -100 100 -100 100 43 50")
    lmp.command("region redge  block -100 100 -100 100 56 57")
    lmp.command("region system block -100 100 -100 100 2  56")
    lmp.command("group Al type 1")
    lmp.command("group C0 type 2")
    lmp.command("group C1 type 3")
    lmp.command("group MoS2 type 4 5")
    lmp.command("group ledge region ledge")
    lmp.command("group left region left")
    lmp.command("group lal region lal")
    lmp.command("group ral region ral")
    lmp.command("group right region right")
    lmp.command("group redge region redge")
    lmp.command("group system region system")
    lmp.command("group edge union ledge redge")
    lmp.command("fix nve system nve")
    lmp.command("fix lang  system  langevin 300 300 0.10 48279")
    lmp.command("compute temp system temp")
    lmp.command("thermo 10")
    lmp.command("thermo_style custom time c_temp epair etotal press")
    lmp.command("run 20 post no")
    temp = lmp.extract_compute("temp", 0, 0)
    if rank == 0:
        print(temp, flush=True)
    lmp.command("run 480 pre no")
    temp = lmp.extract_compute("temp", 0, 0)
    if rank == 0:
        print(temp, flush=True)
    lmp.command("unfix lang")
    lmp.command("dump dcd all dcd 10000 trajectory.dcd")
    lmp.command("dump_modify dcd unwrap yes")

    lmp.command("compute ke system ke/atom")
    lmp.command("variable temp atom c_ke*1.6E-19/(1.5*1.38E-23)")
    lmp.command("compute Thot  left  temp")
    lmp.command("compute Tcold right temp")
    lmp.command("fix hot  left  langevin 350 350 1.00 48279 tally yes")
    lmp.command("fix cold right langevin 250 250 1.00 48279 tally yes")
    lmp.command("fix_modify hot  temp Thot")
    lmp.command("fix_modify cold temp Tcold")
    lmp.command("variable hot_flux  equal  -f_hot/(0.00001+time)")
    lmp.command("variable cold_flux equal  f_cold/(0.00001+time)")
    lmp.command("thermo_modify flush yes")

    lmp.command("compute lal_temp lal temp")
    lmp.command("compute C0_temp C0 temp")
    lmp.command("compute MoS2_temp MoS2 temp")
    lmp.command("compute C1_temp C1 temp")
    lmp.command("compute ral_temp ral temp")

    lmp.command("run 0")
    left = lal = C0 = M1 = C1 = ral = right = 0
    t_profile = np.zeros((num_pico_s, 7))
    flux = np.zeros((num_pico_s, 2))
    start = time()
    for i in range(num_pico_s):
        for j in range(100):
            lmp.command("run 10 pre no post no")
            if j == 89:
                left = lal = C0 = M1 = C1 = ral = right = 0
            if j > 89:
                left += lmp.extract_compute("Thot", 0, 0)
                lal += lmp.extract_compute("lal_temp", 0, 0)
                C0 += lmp.extract_compute("C0_temp", 0, 0)
                M1 += lmp.extract_compute("MoS2_temp", 0, 0)
                C1 += lmp.extract_compute("C1_temp", 0, 0)
                ral += lmp.extract_compute("ral_temp", 0, 0)
                right += lmp.extract_compute("Tcold", 0, 0)
        hot = lmp.extract_variable("hot_flux", "left", 0)
        cold = lmp.extract_variable("cold_flux", "right", 0)

        if rank == 0:
            print(i, 100*i/num_pico_s, '%', i/1000/((time()-start)/60/60/24), "ns/day", flush=True)
            flux[i] = [hot, cold]
            t_profile[i] = [left/10, lal/10, C0/10, M1/10, C1/10, lal/10, right/10]
            if i % 25 == 0:
                np.savez("data_save", flux=flux, t_profile=t_profile)

    if rank == 0:
        np.savez("data", flux=flux, t_profile=t_profile)

