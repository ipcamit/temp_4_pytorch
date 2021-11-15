import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Optional

torch.set_default_dtype(torch.float64)

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
# @torch.jit.script
def calc_d_sw2(A, B, p, q, sigma, cutoff, rij):
    if rij < cutoff:
        sig_r = sigma / rij
        Bpq = (B * sig_r ** p - sig_r ** q)
        exp_sigma = torch.exp(sigma / (rij - cutoff))
        E2 = A * Bpq * exp_sigma
    else:
        return torch.tensor(0.0)
    return E2


# @torch.jit.script
def calc_d_sw3(lam, cos_beta0, gamma_ij, gamma_ik,
               cutoff_ij, cutoff_ik, cutoff_jk, rij, rik, rjk):
    if ((rij > cutoff_ij) or 
        (rik > cutoff_ik) or 
        (rjk > cutoff_jk)):
        return torch.tensor(0.0)
    else: 
        cos_beta_ikj = (rij**2 + rik**2 - rjk**2) / (2 * rij * rik)
        cos_diff = cos_beta_ikj - cos_beta0
        exp_ij_ik = torch.exp(gamma_ij/(rij - cutoff_ij) + gamma_ik/(rik - cutoff_ik))
        E3 = lam * exp_ij_ik * cos_diff ** 2
    return E3


# @torch.jit.script
def energy_and_forces(
    nl: List[List[int]],
    elements_nl: List[List[int]],
    coords_all,
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    p: List[torch.Tensor],
    q: List[torch.Tensor],
    sigma: List[torch.Tensor],
    gamma: List[torch.Tensor],
    cutoff: List[torch.Tensor],
    lam: List[torch.Tensor],
    cos_beta0: List[torch.Tensor],
    cutoff_jk: List[torch.Tensor]
    ):
    """
    Calculatd Energy for a given list of coordiates, assuming first coordinate
    to be of query atom i, and remaining in the list to be neighbours. 
    """
    energy = torch.tensor(0.0)
    E2 = torch.tensor(0.0)
    E3 = torch.tensor(0.0)
    gamma_ij = torch.tensor(0.0)
    gamma_ik = torch.tensor(0.0)
    cutoff_ij = torch.tensor(0.0)
    cutoff_ik = torch.tensor(0.0)
    xyz_i = torch.zeros(3)
    xyz_j = torch.zeros(3)
    xyz_k = torch.zeros(3)
    rij = torch.zeros(3)
    rik = torch.zeros(3)
    rjk = torch.zeros(3)
    for i, (nli, elements) in enumerate(zip(nl, elements_nl)):
        num_elem = len(nli)
        xyz_i = coords_all[nli[0]]
        elem_i = elements[0]
        for j in range(1, num_elem):
            elem_j = elements[j]
            xyz_j = coords_all[nli[j]]
            rij = xyz_j - xyz_i
            norm_rij = torch.norm(rij)
            # if elem_i == elem_j:
            ij_sum = elem_i + elem_j

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
                    norm_rik = torch.norm(rik)
                    rjk = xyz_k - xyz_j
                    norm_rjk = torch.norm(rjk)
                    gamma_ik = gamma[ik_sum]
                    cutoff_ik = cutoff[ik_sum]
                    E3 = calc_d_sw3(lam[ijk_sum], cos_beta0[ijk_sum], gamma_ij, 
                                    gamma_ik, cutoff_ij, cutoff_ik, 
                                    cutoff_jk[ijk_sum], norm_rij, norm_rik, norm_rjk)
                    energy = energy + E3
    return energy


# https://github.com/pytorch/pytorch/issues/46483
# @torch.jit.script
def get_forces(n_atoms: int,
               energy: torch.Tensor,
               coords: torch.Tensor,
               padding: torch.Tensor):
    model_forces = torch.autograd.grad([energy], [coords],
                                       create_graph=True)[0]

    if model_forces is not None:
        F = torch.zeros((n_atoms, 3))
        F[:n_atoms] = model_forces[:n_atoms]
        if len(padding) != 0:
            pad_forces: torch.Tensor = model_forces[n_atoms:]
            n_padding = len(pad_forces)
            print(n_padding)
            exit()
            if n_atoms < n_padding:
                for i in range(n_atoms):
                    indices = torch.where(padding == i)
                    F[i] = F[i] + torch.sum(pad_forces[indices], 0)
            else:
                for f, org_index in zip(pad_forces, padding):
                    F[org_index] = F[org_index] + f
        F = -1.0 * F
        return F
    else:
        print("GRAD IS NONE ")
        return torch.tensor(-1)

# =============================================================================
class StillingerWeberLayer(nn.Module):
    """
    Stillinger-Weber single species layer for Si atom for use in PyTorch model
    Before optimization, the parameter to be optimized need to be set using 
    set_optim function. Forward method returns energy of the configuration and 
    force array.
    """
    def __init__(self):
        super().__init__()
        self.num_elements = 2

        self.A = [
                    nn.Parameter(torch.tensor(3.9781804791)),
                    nn.Parameter(torch.tensor(11.3797414404)),
                    nn.Parameter(torch.tensor(1.1907355764))
                    ]
        self.B = [
                    torch.tensor(0.4446021306),
                    torch.tensor(0.5266688197),
                    torch.tensor(0.9015152673)
                    ]
        self.p = [torch.tensor(5), torch.tensor(5), torch.tensor(5)]
        self.q = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        self.sigma = [
                    torch.tensor(2.85295),
                    torch.tensor(2.17517),
                    torch.tensor(2.84133)
                    ]
        self.gamma = [
                    torch.tensor(1.3566322033),
                    torch.tensor(1.3566322033),
                    torch.tensor(1.3566322033)
                    ]
        self.cutoff = [
                    torch.tensor(5.54660),
                    torch.tensor(4.02692),
                    torch.tensor(4.51956)
                    ]
        self.lam = [
                    torch.tensor(7.4767529158),
                    torch.tensor(8.1595181220)
                    ]
        self.cos_beta0 = [
                    torch.tensor(0.1428569579923222),
                    torch.tensor(0.1428569579923222)
                    ]
        self.cutoff_jk = [
                    torch.tensor(3.86095),
                    torch.tensor(5.54660)
                    ]

    def forward(self, 
                elements: List[List[int]],
                coords: torch.Tensor,
                nl: List[List[int]],
                ):
        total_conf_energy = energy_and_forces(nl, elements, coords, 
                                              self.A, self.B, self.p, self.q, 
                                              self.sigma, self.gamma, self.cutoff, 
                                              self.lam, self.cos_beta0, self.cutoff_jk)
        return total_conf_energy
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
coords = torch.tensor(coords, requires_grad=True)

nl = []
with open("nl.txt") as infile:
    for line in infile:
        nl.append(list(map(int, line.split())))

padding = []
with open("padding.txt") as infile:
    for line in infile:
        padding.append(list(map(int, line.split())))
padding = torch.as_tensor(padding)
# =============================================================================

# ==========================================================================================================
# Example implementation

SWL = StillingerWeberLayer()

for i in range(10):
    tic()
    _ = SWL(elements, coords, nl)
    F = get_forces(len(nl), _, coords, padding)
    toc()
