"""Made by Charlie Sievers Ph.D. Candidate, UC Davis, Donadio Lab 2019"""

from mpi4py import MPI
import numpy as np
import ctypes as ctypes

""" USEFULL LAMMPS FUNCTION """
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_nlocal(lmp):

    nlocal = lmp.extract_global("nlocal", 0)

    return nlocal


def get_aid(lmp, group=None, cleaned=False):

    if group is None:
        c_aid = lmp.extract_atom("id", 0)
        ptr = ctypes.cast(c_aid, ctypes.POINTER(ctypes.c_int32 * get_nlocal(lmp)))
        aid = np.frombuffer(ptr.contents, dtype=np.int32)
    else:
        try:
            c_aid = lmp.extract_variable("aid", group, 1)
            ptr = ctypes.cast(c_aid, ctypes.POINTER(ctypes.c_double * get_nlocal(lmp)))
            aid = np.frombuffer(ptr.contents, dtype=np.double)
        except ValueError:
            lmp.command("variable aid atom id")
            aid = get_aid(lmp, group)
    if cleaned:
        aid = aid[aid != 0]
        aid = aid.astype(int)
    return aid


def get_gaid(lmp, group=None, cleaned=False):
    
    aid = get_aid(lmp, group, cleaned)
    
    ngroup = comm.allgather(aid)
    for array in ngroup:
        try:
            aid = np.concatenate((aid, array))
        except UnboundLocalError:
            aid = array

    return np.unique(aid)


def gather_per_atom_compute(comm, lmp, name, dim=1, dtype="double", group=None):
    laid = get_aid(lmp, group)
    nlocal = get_nlocal(lmp)
    ngroup = comm.allgather(laid)
    type = dim
    if dim > 1:
        type = 2
    for array in ngroup:
        try:
            aid = np.concatenate((aid, array))
        except UnboundLocalError:
            aid = array
    if dtype == "double":
        mem_type = ctypes.c_double
    elif dtype == "integer":
        mem_type = ctypes.c_int
    elif dtype == "bigint":
        mem_type = ctypes.c_int32
    else:
        print("{} not implemented".format(dtype))
        return

    tmp = lmp.extract_compute(name, 1, type)
    if type == 1:
        ptr = ctypes.cast(tmp, ctypes.POINTER(mem_type * nlocal))
    else:
        ptr = ctypes.cast(tmp[0], ctypes.POINTER(mem_type * nlocal * dim))
    lcompute = comm.allgather(np.frombuffer(ptr.contents).reshape((-1, dim)))
    for array in lcompute:
        try:
            compute = np.concatenate((compute, array))
        except UnboundLocalError:
            compute = array

    aid = np.expand_dims(aid, axis=1)

    compute = np.concatenate((aid, compute), axis=-1)
    compute = compute[compute[..., 0] != 0]
    compute = compute[compute[..., 0].argsort()][..., 1:]

    if dim == 1:
        compute = np.squeeze(compute, axis=-1)

    return compute


def gather_per_atom(comm, lmp, name, dim=1, dtype="double", group=None):
    laid = get_aid(lmp, group)
    nlocal = get_nlocal(lmp)
    ngroup = comm.allgather(laid)
    if dim > 1:
        if dtype == "double":
            type = 3
        else:
            type = 1
    if dim == 1:
        if dtype == "double":
            type = 2
        else:
            type = 0
    for array in ngroup:
        try:
            aid = np.concatenate((aid, array))
        except UnboundLocalError:
            aid = array
    if dtype == "double":
        mem_type = ctypes.c_double
    elif dtype == "integer":
        mem_type = ctypes.c_int
    elif dtype == "bigint":
        mem_type = ctypes.c_int32
    else:
        print("{} not implemented".format(dtype))
        return

    tmp = lmp.extract_atom(name, type)
    if type == 0 or type == 2:
        ptr = ctypes.cast(tmp, ctypes.POINTER(mem_type * nlocal))
    else:
        ptr = ctypes.cast(tmp[0], ctypes.POINTER(mem_type * nlocal * dim))
    latom = comm.allgather(np.frombuffer(ptr.contents).reshape((-1, dim)))
    for array in latom:
        try:
            atom = np.concatenate((atom, array))
        except UnboundLocalError:
            atom = array

    aid = np.expand_dims(aid, axis=1)

    atom = np.concatenate((aid, atom), axis=-1)
    atom = atom[atom[..., 0] != 0]
    atom = atom[atom[..., 0].argsort()][..., 1:]

    if dim == 1:
        atom = np.squeeze(atom, axis=-1)

    return atom


def get_compute_vector(lmp, name, length):
    tmp = lmp.extract_compute(name, 0, 1)
    ptr = ctypes.cast(tmp, ctypes.POINTER(ctypes.c_double * length))
    return np.frombuffer(ptr.contents)
