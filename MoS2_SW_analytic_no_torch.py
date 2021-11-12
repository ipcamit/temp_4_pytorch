import numpy as np
from typing import List, Tuple

def tic():
    """https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions"""
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
# SW subroutines

def calc_d_sw2(A, B, p, q, sigma, cutoff, rij):
    if rij < cutoff:
        sig_r = sigma / rij
        one_by_delta_r = 1.0 / (rij - cutoff)
        Bpq = (B * sig_r ** p - sig_r ** q)
        exp_sigma = np.exp(sigma * one_by_delta_r )
        E2 = A * Bpq * exp_sigma 
        F = (q * sig_r ** (q + 1)) - p * B * sig_r ** (p + 1) - Bpq * (sigma * one_by_delta_r) ** 2
        F = F * (1./sigma) * A * exp_sigma
    else:
        return 0.0, 0.0 
    return E2, F


def calc_d_sw3(lam, cos_beta0, gamma_ij, gamma_ik,
                 cutoff_ij, cutoff_ik, cutoff_jk, rij, rik, rjk, dE3_dr):
    if ((rij > cutoff_ij) or 
        (rik > cutoff_ik) or 
        (rjk > cutoff_jk)):
        dE3_dr[0] = 0.0; dE3_dr[1] = 0.0; dE3_dr[2] = 0.0
        return 0.0
    else: 
        cos_beta_ikj = (rij**2 + rik**2 - rjk**2) / (2 * rij * rik)
        cos_diff = cos_beta_ikj - cos_beta0

        exp_ij_ik = np.exp(gamma_ij/(rij - cutoff_ij) + gamma_ik/(rik - cutoff_ik))

        dij = - gamma_ij/(rij - cutoff_ij)**2
        dik = - gamma_ik/(rik - cutoff_ik)**2

        E3 = lam * exp_ij_ik * cos_diff ** 2

        dcos_drij = (rij**2 - rik**2 + rjk**2) / (2 * rij**2 * rik)
        dcos_drik = (rik**2 - rij**2 + rjk**2) / (2 * rik**2 * rij)
        dcos_drjk = (- rjk) / (rij * rik)

        dE3_dr[0] = lam * cos_diff * exp_ij_ik * (dij * cos_diff + 2 * dcos_drij)
        dE3_dr[1] = lam * cos_diff * exp_ij_ik * (dik * cos_diff + 2 * dcos_drik)
        dE3_dr[2] = lam * cos_diff * exp_ij_ik * 2 * dcos_drjk
    return E3


def energy_and_forces(nl, elements_nl, coords_all, A, B, p, q, sigma, gamma, 
                        cutoff, lam, cos_beta0, cutoff_jk):
    """
    Calculatd Energy for a given list of coordiates, assuming first coordinate
    to be of query atom i, and remaining in the list to be neighbours. 
    """
    energy = 0.0
    F2 = 0.0
    F3 = np.zeros(3)
    E2 = 0.0
    E3 = 0.0
    gamma_ij = 0.0
    gamma_ik = 0.0
    cutoff_ij = 0.0
    cutoff_ik = 0.0
    xyz_i = np.zeros(3)
    xyz_j = np.zeros(3)
    xyz_k = np.zeros(3)
    rij = np.zeros(3)
    rik = np.zeros(3)
    rjk = np.zeros(3)
    F = np.zeros_like(coords_all)
    F_comp = np.zeros(3)
    for i, (nli, elements) in enumerate(zip(nl,elements_nl)):
        num_elem = len(nli)
        xyz_i = coords_all[nli[0]]
        elem_i = elements[0]
        for j in range(1, num_elem):
            elem_j = elements[j]
            xyz_j = coords_all[nli[j]]
            rij = xyz_j - xyz_i
            norm_rij = np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2)
            # if elem_i == elem_j:
            ij_sum = elem_i + elem_j

            E2, F2 = calc_d_sw2(A[ij_sum], B[ij_sum], p[ij_sum], q[ij_sum], sigma[ij_sum], cutoff[ij_sum], norm_rij)
            energy = 0.5 * E2
            F_comp =  0.5 * F2/norm_rij * rij
            F[i,:] = F[i,:] + F_comp
            F[nli[j], :] = F[nli[j],:] - F_comp
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
                    norm_rik = np.sqrt(rik[0]**2 + rik[1]**2 + rik[2]**2)
                    rjk = xyz_k - xyz_j
                    norm_rjk = np.sqrt(rjk[0]**2 + rjk[1]**2 + rjk[2]**2)
                    gamma_ik = gamma[ik_sum]
                    cutoff_ik = cutoff[ik_sum]
                    E3 =  calc_d_sw3(lam[ijk_sum], cos_beta0[ijk_sum], gamma_ij, gamma_ik,
                                    cutoff_ij, cutoff_ik, cutoff_jk[ijk_sum], norm_rij, norm_rik, norm_rjk, F3)
                    energy = energy + E3
                    F_comp[:] = F3[0]/norm_rij * rij
                    F[i, :] = F[i, :] + F_comp
                    F[nli[j], :] = F[nli[j], :] - F_comp
                    F_comp[:] = F3[1]/norm_rik * rik
                    F[i, :] = F[i, :] + F_comp
                    F[nli[k], :] = F[nli[k], :] - F_comp
                    F_comp[:] = F3[2]/norm_rjk * rjk
                    F[nli[j], :] = F[nli[j], :] + F_comp
                    F[nli[k], :] = F[nli[k], :] - F_comp
    return energy, F


# =============================================================================
class StillingerWeberLayer():
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

        self.A = [3.9781804791, 11.3797414404, 1.1907355764]
        self.B = [0.4446021306, 0.5266688197, 0.9015152673]
        self.p = [5, 5, 5]
        self.q = [0, 0, 0]
        self.sigma = [2.85295, 2.17517, 2.84133]
        self.gamma = [1.3566322033, 1.3566322033, 1.3566322033]
        self.cutoff = [5.54660, 4.02692, 4.51956]
        self.lam = [7.4767529158, 8.1595181220]
        self.cos_beta0 = [0.1428569579923222, 0.1428569579923222]
        self.cutoff_jk = [3.86095, 5.54660]

    def __call__(self, elements, coords, nl, padding):
        n_atom = len(nl)
        F = np.zeros((n_atom, 3))
        total_conf_energy, forces = energy_and_forces(nl, elements, coords, self.A, 
                                            self.B, self.p, self.q, self.sigma, self.gamma,
                                            self.cutoff, self.lam, self.cos_beta0, self.cutoff_jk)
        F[:n_atom] = forces[:n_atom]

        if len(padding) != 0:
            pad_forces = forces[n_atom:]
            n_padding = len(pad_forces)

            if n_atom < n_padding:
                for i in range(n_atom):
                    indices = np.where(padding == i)
                    F[i] = F[i] + np.sum(pad_forces[indices], axis=0)
            else:
                for f, org_index in zip(pad_forces, padding):
                    F[org_index] = F[org_index] + f
        return total_conf_energy, F
# ==========================================================================================================


# ================================================================================
# Read data

elements = []
with open("elements.txt") as infile:
    for line in infile:
        elements.append(list(map(int, line.split())))

coords = []
with open("coords.txt") as infile:
    for line in infile:
        coords.append(list(map(float, line.split())))
coords = np.array(coords)

nl = []
with open("nl.txt") as infile:
    for line in infile:
        nl.append(list(map(int, line.split())))

padding = []
with open("padding.txt") as infile:
    for line in infile:
        padding.append(list(map(int, line.split())))
# =============================================================================

# ==========================================================================================================
# Example implementation

SWL = StillingerWeberLayer()

for i in range(10):
    tic()
    _ = SWL(elements, coords, nl, padding)
    toc()
