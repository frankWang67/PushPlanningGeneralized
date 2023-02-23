"""
    The plugin for planar pushing numerical simulation with contact.
    Use this script to compute contact force, use any physics engine as collision detector.
"""

import numpy as np

from polytope_symbolic_system.common.utils import *
from r3t.polygon.solver import LCP


def compute_contact_force(contact, l_surf, input):
    """
    :param contact: the list that stores contact information [{}, {}, ..., {}]
                    each element is a dict {}
                    keys are:
                        - 'ID': list, ids of two objects, i.e., [id0, id1], id0 is always the planar slider
                        - 'J_mat': list, contact jacobian of two objects
                        - 'D_mat': list, direction matrix of two objects
                        - 'N_mat': contact normals, unit vectors,
                                   obtain contact tangents by rotating 90 deg ccw
                        - 'Sign': 1 or -1 (1: contact normal is on the planar slider,
                                           -1: contact normal is not on the planar slider)
                        - 'Miu: frictional coefficient
    :param l_surf: the list that stores limit surface matrix A of all objects, indexed by id, i.e., [A0, A1, ..., Ak]
    :param input: the pusher wrench in the planar slider's local frame

    :return f_opt: the contact forces (N_contact, 2)
    """
    N_contact = len(contact)
    N_obj = len(l_surf)
    
    # force -> twist matrix
    M_part = [np.zeros(3, 3*N_contact) for i in range(N_obj)]
    
    for i in range(N_contact):
        id0, id1 = contact[i]['ID'][0], contact[i]['ID'][1]
        Jc0, Jc1 = contact[i]['J_mat'][0], contact[i]['J_mat'][1]
        Dc0, Dc1 = contact[i]['D_mat'][0], contact[i]['D_mat'][1]
        
        M_part[id0][:, 3*i:3*(i+1)] += l_surf[id0] @ Jc0.T @ Dc0
        M_part[id1][:, 3*i:3*(i+1)] += l_surf[id1] @ Jc1.T @ Dc1

    # LCP matrices
    M = np.zeros(3*N_contact+N_contact, 3*N_contact+N_contact)
    q = np.zeros(3*N_contact+N_contact, 1)

    for i in range(N_contact):
        # contact normal and tangential
        Nci = contact[i]['N_mat']
        Tci = np.c_[rotation_matrix(np.pi/2) @ Nci, -rotation_matrix(np.pi/2) @ Nci]

        # contact objects and jacobian
        id0, id1 = contact[i]['ID'][0], contact[i]['ID'][1]
        Jc0, Jc1 = contact[i]['J_mat'][0], contact[i]['J_mat'][1]

        # add constraints to matrix
        Sign = contact[i]['Sign']
        Miu = contact[i]['Miu']
        assert Sign == 1 or Sign == -1

        # add constraints
        M[3*i+0, 0:3*N_contact] = Sign * Nci.T @ (Jc0 @ M_part[id0] - Jc1 @ M_part[id1])
        M[3*i+1:3*i+3, 0:3*N_contact] = Sign * Tci.T @ (Jc0 @ M_part[id0] - Jc1 @ M_part[id1])

        M[3*i+1:3*i+3, 3*N_contact+i] = 1
        M[3*N_contact+i, 3*i+1:3*i+3] = 1

        M[3*N_contact+i, 3*i+0] = Miu

        # add constants
        q[3*i+0] = Sign * Nci.T @ l_surf[id0] @ input
        q[3*i+1:3*i+3] = Sign * Tci.T @ l_surf[id0] @ input

    z_opt = LCP(M, q)
    f_opt = z_opt[0:3*N_contact].reshape(N_contact, 3)
    f_opt = f_opt @ np.array([[1, 0, 0], [0, 1, -1]]).T

    return f_opt
