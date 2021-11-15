using Flux, DelimitedFiles, BenchmarkTools
# for proper ml need to use reversediff over zygote

# Load Data ============================================
coords = readdlm("coords.txt", ' ')

elements = []
for line in eachline("elements.txt")
    push!(elements, parse.(Int64, split(line," ")))
end

nl = []
for line in eachline("nl.txt")
    push!(nl, parse.(Int64, split(line," ")))
end
padding = readdlm("padding.txt", Int64)
# ======================================================

# Energy Functions =====================================
function norm(x)
    return sqrt(sum(x.^2))
end

function calc_d_sw2(A, B, p, q, σ, cutoff, rij)
    if (rij < cutoff)
        sig_r = σ / rij
        exp_dr = exp(σ / (rij - cutoff))
        Bpq = (B * sig_r ^ p - sig_r ^ q)
        E2 = A * Bpq *  exp_dr
    else
        return 0.0
    end
    return E2
end

function calc_d_sw3(λ, cosβ0, γ_ij, γ_ik,cutoff_ij, cutoff_ik, cutoff_jk, rij, rik, rjk)
    if ((rij > cutoff_ij) || (rik > cutoff_ik) || (rjk > cutoff_jk))
        return 0.0
    else
        cosβikj = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
        cos_diff = cosβikj - cosβ0
        exp_ij_ik = exp(γ_ij/(rij - cutoff_ij) + γ_ik/(rik - cutoff_ik))
        dij = - γ_ij/(rij - cutoff_ij)^2
        dik = - γ_ik/(rik - cutoff_ik)^2
        E3 = λ * exp_ij_ik * cos_diff * cos_diff
    end
    return E3
end

function energy_and_forces(nl, elements_nl, coords, A, B, p, q, 
                            σ, γ, cutoff, λ, cosβ0, cutoff_jk)
    """
    Calculatd Energy for a given list of coordiates, assuming first coordinate
    to be of query atom i, and remaining in the list to be neighbours. 
    """
    energy = 0.0
    E2 = 0.0
    E3 = 0.0
    γ_ij = 0.0
    γ_ik = 0.0
    cutoff_ij = 0.0
    cutoff_ik = 0.0
    xyz_i = zeros(3)
    xyz_j = zeros(3)
    xyz_k = zeros(3)
    rij = zeros(3)
    rik = zeros(3)
    rjk = zeros(3)
    for (i, (nli, element)) in enumerate(zip(nl, elements_nl))
        num_elem = length(nli)
        xyz_i = coords[nli[1] + 1, : ]
        elem_i = element[1]
        for j  = 2 : num_elem
            elem_j = element[j]
            xyz_j = coords[nli[j] + 1, : ]
            rij = xyz_j .- xyz_i
            norm_rij = norm(rij)
            ij_sum = elem_i + elem_j + 1
            E2 = calc_d_sw2(A[ij_sum], B[ij_sum], p[ij_sum], q[ij_sum], 
                                    σ[ij_sum], cutoff[ij_sum], norm_rij)
            energy = energy + 0.5 * E2
            γ_ij = γ[ij_sum]
            cutoff_ij = cutoff[ij_sum]
            for k = (j + 1) : num_elem
                elem_k = element[k]
                if ((elem_i != elem_j) && (elem_j == elem_k))
                    ijk_sum = 3 + -1 * (elem_i + elem_j + elem_k)
                    ik_sum = elem_i + elem_k + 1
                    xyz_k = coords[nli[k] + 1, : ]
                    rik = xyz_k .- xyz_i
                    norm_rik = norm(rik)
                    rjk = xyz_k .- xyz_j
                    norm_rjk = norm(rjk)
                    γ_ik = γ[ik_sum]
                    cutoff_ik = cutoff[ik_sum]
                    E3 = calc_d_sw3(λ[ijk_sum], cosβ0[ijk_sum], γ_ij, γ_ik, cutoff_ij, cutoff_ik,
                             cutoff_jk[ijk_sum], norm_rij, norm_rik, norm_rjk)
                    energy = energy + E3
                end
            end
        end
    end
    return energy
end


# ======================================================
function StillingerWeber()
    A = [3.9781804791, 11.3797414404, 1.1907355764]
    B = [0.4446021306, 0.5266688197, 0.9015152673]
    p = [5, 5, 5]
    q = [0, 0, 0]
    σ = [2.85295, 2.17517, 2.84133]
    γ = [1.3566322033, 1.3566322033, 1.3566322033]
    cutoff = [5.54660, 4.02692, 4.51956]

    λ = [7.4767529158, 8.1595181220]
    cosβ0 = [0.1428569579923222, 0.1428569579923222]
    cutoff_jk = [3.86095, 5.54660]
    return (nl, elements, coords) -> 
            energy_and_forces(nl, elements, coords, A, B, p, q, 
                                σ, γ, cutoff, λ, cosβ0, cutoff_jk)
end

function pad_forces(forces, n_atom, padding)
    F = zeros((n_atom, 3))
    F[1 : n_atom,:] = forces[1 : n_atom,:]

    if length(padding) != 0
        pad_forces = forces[n_atom + 1 : end,:]
        n_padding = length(pad_forces)
        if n_atom < n_padding
            for i = 1 : n_atom
                for j = 1: length(padding)
                    if (padding[j] + 1 == i)
                        F[i,:] += pad_forces[j,:]
                    end
                end
            end
        else
            for (f, org_index) in zip(pad_forces, padding)
                F[org_index,:] += f
            end
        end
    end
    return -1 * F
end

# =========================================================

function single_step()
    SWL = StillingerWeber()
    gd = gradient(Flux.params(coords)) do
        E = SWL(nl,elements,coords)
    end
    F_unwrapped = gd[coords] 
    F = pad_forces(F_unwrapped, length(nl), padding)
    return F
end


# Zygote
# BenchmarkTools.Trial: 4 samples with 1 evaluation.
#  Range (min … max):  4.855 s …    5.294 s  ┊ GC (min … max): 17.18% … 19.65%
#  Time  (median):     4.955 s               ┊ GC (median):    16.53%
#  Time  (mean ± σ):   5.015 s ± 207.035 ms  ┊ GC (mean ± σ):  16.84% ±  2.23%

#   █                        ▁                               ▁
#   █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#   4.85 s         Histogram: frequency by time         5.29 s <

#  Memory estimate: 527.54 MiB, allocs estimate: 8614424.

# Analytical
# BenchmarkTools.Trial: 33 samples with 1 evaluation.
#  Range (min … max):  605.209 ms … 645.850 ms  ┊ GC (min … max): 2.56% … 2.32%
#  Time  (median):     624.230 ms               ┊ GC (median):    2.50%
#  Time  (mean ± σ):   624.909 ms ±  10.209 ms  ┊ GC (mean ± σ):  2.37% ± 0.38%

#   █ ▁   ▁  ▁        █▁▁ ▁▁ █▁█▁  ▁█ ▁ █  ▁█▁ ▁    █  ▁▁       ▁
#   █▁█▁▁▁█▁▁█▁▁▁▁▁▁▁▁███▁██▁████▁▁██▁█▁█▁▁███▁█▁▁▁▁█▁▁██▁▁▁▁▁▁▁█ ▁
#   605 ms           Histogram: frequency by time          646 ms <

#  Memory estimate: 135.90 MiB, allocs estimate: 3812606.