from lammps import lammps
import numpy as np
from lammps_tools import gather_per_atom_compute, get_aid
from mpi4py import MPI
import ctypes as ctypes
import sys

na = np.newaxis

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Kelvin
temp = 300

# eV*ps
# hbar = 6.582e-4

# J*s
hbar = 1.054e-34
# eV/K
# kb = 8.617e-5

# J/K
kb = 1.381e-23
# 1/energy
beta = 1/kb/temp

# data files
infile = "minimized_algg.lmp"
ff_file = "../ff.lmp"

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

lmp.command("read_data {}".format(infile))
lmp.command("region bot block -100 100 -100 100 0 6")
lmp.command("region top block -100 100 -100 100 43 49")
lmp.command("group bot region bot")
lmp.command("group top region top")
lmp.command("group main subtract all top bot")
lmp.command("group edge union top bot")
lmp.command("fix freeze edge setforce 0.0 0.0 0.0")

natoms = lmp.get_natoms()
pnatoms = natoms
box = lmp.extract_box()

x = lmp.gather_atoms('x', 1, 3)
ptr = ctypes.cast(x, ctypes.POINTER(ctypes.c_double * natoms * 3))
coords = np.frombuffer(ptr.contents, dtype=np.double).reshape((-1, 3))

lmp.file("{}".format(ff_file))

lmp.command("compute mass all property/atom mass")
mass = gather_per_atom_compute(comm, lmp, "mass", 1)

# mass in amu
sqrt_mass_3n = np.repeat(np.sqrt(mass), 3)
mass_matrix = sqrt_mass_3n[:, na] * sqrt_mass_3n[na, :]
odd_mass_matrix = sqrt_mass_3n[:, na] / sqrt_mass_3n[na, :]

main_id = get_aid(lmp, group="main", cleaned=True)-1
top_id = get_aid(lmp, group="top", cleaned=True)-1
saved_coords = np.array(coords, copy=True)

for i in np.arange(3, 4):

    change = 1.12+i*0.001
    coords[main_id] = saved_coords[main_id]
    coords[top_id] = saved_coords[top_id]+[0.0, 0.0, change]

    # Set lammps positions (angstroms)
    ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double * natoms * 3))
    lmp.scatter_atoms("x", 1, 3, ptr)

    lmp.command("min_style fire")
    lmp.command("min_modify line quadratic")
    lmp.command("minimize 1.0e-50 1.0e-50 100000000000 10000000000000")

    # get number of atoms
    natoms = lmp.get_natoms()

    energy = lmp.extract_compute('thermo_pe', 0, 0)

    lmp.command("dynamical_matrix all eskm 0.000001 file graphite.dat binary yes")

    if rank == 0:
        dynmat = np.fromfile("graphite.dat", dtype=float)
        dynlen = int(np.sqrt(len(dynmat)))
        pdynlen = 3*pnatoms
        try:
            assert dynlen == len(coords)*3
            asserted = True
        except AssertionError:
            print("Something went awry")
            exit(0)
        # Units of THz^2
        dynmat = dynmat.reshape((dynlen, dynlen))

        # acoustic sum rule
        for i in range(dynlen):
            diagonal = dynmat[i, i]
            row_difference = np.sum(dynmat[i])
            dynmat[i] -= row_difference / (dynlen - 1)
            dynmat[i, i] = diagonal

        # symmetrize matrix across diagonal
        for i in range(dynlen):
            for j in range(i, dynlen):
                if i != j:
                    ave = 0.5 * (dynmat[i, j] + dynmat[j, i])
                    dynmat[i, j] = ave
                    dynmat[j, i] = ave

        # Units of kg*s^-2 (This might not be the best bro)
        # fcm = dynmat * mass_matrix / 6.022e2
        # Units of eV/A^2
        fcm = 1.0365e-4 * dynmat * mass_matrix

        eigvals, eigvecs = np.linalg.eig(dynmat)
        ieigvecs = np.array(eigvecs, copy=True)
        ieigvecs_0 = np.array(eigvecs, copy=True)
        eigvecs = eigvecs

        # Omega is in Hz
        omegas = np.sqrt(eigvals) * 1e12
        omegas_2 = np.sqrt(np.abs(eigvals)) * 1e12
        omegas = np.nan_to_num(omegas)
        p_energy = 6.242e18*0.5*np.sum(hbar*omegas_2)
        p_entropy = -6.242e18*np.sum(np.log(1+(1/(np.exp(beta*hbar*omegas_2)-1)))/beta)
        free_energy = energy+p_energy+p_entropy
        print(change, free_energy, p_energy+p_entropy)

