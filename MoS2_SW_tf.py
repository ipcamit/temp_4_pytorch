import tensorflow as tf
import numpy as np
from typing import List, Tuple


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

# =============================================================================
# StillingerWeber Model
# =============================================================================
# SW subroutines (PyTorch gets fussy if function are part of class)


# @tf.function
def calc_d_sw2(A, B, p, q, sigma, cutoff, rij):
    if rij < cutoff:
        sig_r = sigma / rij
        Bpq = (B * sig_r ** p - sig_r ** q)
        exp_sigma = tf.exp(sigma / (rij - cutoff))
        E2 = A * Bpq * exp_sigma 
    else:
        return tf.constant(0.0)
    return E2


# @tf.function
def calc_d_sw3(lam, cos_beta0, gamma_ij, gamma_ik,
               cutoff_ij, cutoff_ik, cutoff_jk, rij, rik, rjk):
    if ((rij > cutoff_ij) or 
        (rik > cutoff_ik) or 
        (rjk > cutoff_jk)):
        return tf.constant(0.0)
    else: 
        cos_beta_ikj = (rij**2 + rik**2 - rjk**2) / (2 * rij * rik)
        cos_diff = cos_beta_ikj - cos_beta0

        exp_ij_ik = tf.exp(gamma_ij/(rij - cutoff_ij) + gamma_ik/(rik - cutoff_ik))
        E3 = lam * exp_ij_ik * cos_diff ** 2
    return E3


# @tf.function#
def energy_and_forces(
    nl: List[List[int]],
    elements_nl: List[List[int]],
    coords_all,
    A,
    B,
    p,
    q,
    sigma,
    gamma,
    cutoff,
    lam,
    cos_beta0,
    cutoff_jk
    ):
    """
    Calculatd Energy for a given list of coordiates, assuming first coordinate
    to be of query atom i, and remaining in the list to be neighbours. 
    """
    energy = tf.constant(0.0)
    E2 = tf.constant(0.0)
    E3 = tf.constant(0.0)
    gamma_ij = tf.constant(0.0)
    gamma_ik = tf.constant(0.0)
    cutoff_ij = tf.constant(0.0)
    cutoff_ik = tf.constant(0.0)
    xyz_i = tf.zeros(3)
    xyz_j = tf.zeros(3)
    xyz_k = tf.zeros(3)
    rij = tf.zeros(3)
    rik = tf.zeros(3)
    rjk = tf.zeros(3)
    for i, (nli, elements) in enumerate(zip(nl, elements_nl)):
        num_elem = len(nli)
        xyz_i = coords_all[nli[0]]
        elem_i = elements[0]
        for j in range(1, num_elem):
            elem_j = elements[j]
            xyz_j = coords_all[nli[j]]
            rij = xyz_j - xyz_i
            norm_rij = tf.norm(rij)
            # if elem_i == elem_j:
            ij_sum = elem_i + elem_j
            ij_sum = ij_sum.numpy()
            E2 = calc_d_sw2(A[ij_sum], B[ij_sum], p[ij_sum], q[ij_sum],
                            sigma[ij_sum], cutoff[ij_sum], norm_rij)
            energy = energy + 0.5 * E2
            gamma_ij = gamma[ij_sum]
            cutoff_ij = cutoff[ij_sum]

            for k in range(j + 1, num_elem):
                elem_k = elements[k]
                if (elem_i != elem_j) and \
                   (elem_j == elem_k):
                    ijk_sum = 2 + -1 * (elem_i + elem_j + elem_k)
                    ik_sum = elem_i + elem_k
                    xyz_k = coords_all[nli[k]]
                    rik = xyz_k - xyz_i
                    norm_rik = tf.norm(rik)
                    rjk = xyz_k - xyz_j
                    norm_rjk = tf.norm(rjk)
                    gamma_ik = gamma[ik_sum]
                    cutoff_ik = cutoff[ik_sum]
                    E3 = calc_d_sw3(lam[ijk_sum], cos_beta0[ijk_sum], gamma_ij,
                                    gamma_ik, cutoff_ij, cutoff_ik, cutoff_jk[ijk_sum],
                                    norm_rij, norm_rik, norm_rjk)
                    energy = energy + E3
    return energy


# @tf.function
def get_forces(tape,
               n_atoms: int,
               energy,
               coords,
               padding):
    model_forces = tape.gradient(energy, coords)
    F = tf.Variable(model_forces[:n_atoms])
    if len(padding) != 0:
        pad_forces = model_forces[n_atoms:]
        n_padding = len(pad_forces)
        if n_atoms < n_padding:
            for i in range(n_atoms):
                indices = tf.where(padding == i)
                F[i].assign(F[i] + tf.reduce_sum(model_forces[indices]))
        else:
            for f, org_index in zip(pad_forces, padding):
                F[org_index].assign(F[org_index] + f)
    F = -1.0 * F
    return F

# =============================================================================


class StillingerWeberLayer(tf.keras.layers.Layer):
    """
    Stillinger-Weber single species layer for Si atom for use in PyTorch model
    Before optimization, the parameter to be optimized need to be set using 
    set_optim function. Forward method returns energy of the configuration and 
    force array.
    """

    def __init__(self):
        super().__init__()
        self.elements = elements

        self.num_elements = 2

        self.A = [
                    tf.constant(3.9781804791),
                    tf.constant(11.3797414404),
                    tf.constant(1.1907355764)
                    ]
        self.B = [
                    tf.constant(0.4446021306),
                    tf.constant(0.5266688197),
                    tf.constant(0.9015152673)
                    ]
        self.p = [tf.constant(5.), tf.constant(5.), tf.constant(5.)]
        self.q = [tf.constant(0.), tf.constant(0.), tf.constant(0.)]
        self.sigma = [
                    tf.constant(2.85295),
                    tf.constant(2.17517),
                    tf.constant(2.84133)
                    ]
        self.gamma = [
                    tf.constant(1.3566322033),
                    tf.constant(1.3566322033),
                    tf.constant(1.3566322033)
                    ]
        self.cutoff = [
                    tf.constant(5.54660),
                    tf.constant(4.02692),
                    tf.constant(4.51956)
                    ]
        self.lam = [
                    tf.constant(7.4767529158),
                    tf.constant(8.1595181220)
                    ]
        self.cos_beta0 = [
                    tf.constant(0.1428569579923222),
                    tf.constant(0.1428569579923222)
                    ]
        self.cutoff_jk = [
                    tf.constant(3.86095),
                    tf.constant(5.54660)
                    ]

    def call(self, 
             elements: List[List[int]],
             coords,
             nl: List[List[int]],
             ):
        total_conf_energy = energy_and_forces(nl, elements, coords, self.A,
                                              self.B, self.p, self.q, self.sigma,
                                              self.gamma, self.cutoff, self.lam,
                                              self.cos_beta0, self.cutoff_jk)
        return total_conf_energy
# =============================================================================


# =============================================================================
# Read data

elements = []
with open("elements.txt") as infile:
    for line in infile:
        elements.append(list(map(int, line.split())))

coords = []
with open("coords.txt") as infile:
    for line in infile:
        coords.append(list(map(float, line.split())))
coords = tf.Variable(coords)

nl = []
with open("nl.txt") as infile:
    for line in infile:
        nl.append(list(map(int, line.split())))

padding = []
with open("padding.txt") as infile:
    for line in infile:
        padding.append(list(map(int, line.split())))
# =============================================================================

# =============================================================================
# Example implementation

SWL = StillingerWeberLayer()

for i in range(10):
    tic()
    with tf.GradientTape() as tape:
        E = SWL(elements, coords, nl)
        # F = get_forces(tape, len(nl), E, coords, padding)
    toc()
