module MpsBenchmarks

using BenchmarkTools
using TensorOperations
using Random
using StableRNGs

const SUITE = BenchmarkGroup()

const ELTYPE = ComplexF64
const D_PHYS = (2,)
const D_MPO_VIRT = (5,)
const D_MPS_VIRT = round.(Int, (exp2.(range(3, 8, 5))))


samerand(args...) = rand(StableRNG(1), args...)
mps_mat(D) = samerand(ELTYPE, (D, D))
mps_tensor(d, D) = samerand(ELTYPE, (D, d, D))
mpo_tensor(d, D) = samerand(ELTYPE, (D, d, d, D))

mps_transfer(A, B, ρ_in) =
    @tensor opt=(a, d, b, e) ρ_out[a, b] := A[a, c, d] * conj(B[b, c, e]) * ρ_in[d, e]


mps_expval_1(A, B, GL, O, GR) =
    @tensor opt=(a, b, g, h) A[a, c, b] * GL[g, d, a] * O[d, f, c, e] * GR[b, e, h] * conj(B[g, f, h])

mps_expval_2(A₁, A₂, B₁, B₂, GL, O₁, O₂, GR) =
    @tensor opt=(a, f, k, c, h, m) A₁[a, d, f] * A₂[f, i, k] * conj(B₁[c, e, h]) * conj(B₂[h, j, m]) *
                                  GL[c, b, a] * O₁[b, e, d, g] * O₂[g, j, i, l] * GR[k, l, m]

# mps_expval_3(A₁, A₂, B₁, B₂, GL, O, GR) =
#     @tensoropt (a, f, j, l, g, c) A₁[a, d, f] * A₂[f, h, j] * conj(B₁[c, e, g]) * conj(B₂[g, i, l]) *
#                                   GL[c, b, a] * O[b, e, i, d, h, k] * GR[j, k, l]

mps_dAC_1(A, GL, O, GR) =
    @tensor opt=(a, f, c, h) AC′[c, e, h] := A[a, d, f] * GL[c, b, a] * O[b, e, d, g] * GR[f, g, h]

mps_dAC_2(A₁, A₂, GL, O₁, O₂, GR) =
    @tensor opt=(a, f, j, c, l) AC′[c, e, i, l] := A₁[a, d, f] * A₂[f, h, j] *
                                                  GL[c, b, a] * O₁[b, e, d, g] * O₂[g, i, h, k] * GR[j, k, l]

function benchmark_mps_transfer(d, D_mps)
    A = mps_tensor(d, D_mps)
    B = mps_tensor(d, D_mps)
    ρ = mps_mat(D_mps)
    return @benchmarkable mps_transfer($A, $B, $ρ)
end

function benchmark_mps_expval_1(d, D_mps, D_mpo)
    A = mps_tensor(d, D_mps)
    B = mps_tensor(d, D_mps)
    GL = mps_tensor(D_mpo, D_mps)
    GR = mps_tensor(D_mpo, D_mps)
    O = mpo_tensor(d, D_mpo)
    return @benchmarkable mps_expval_1($A, $B, $GL, $O, $GR)
end

function benchmark_mps_expval_2(d, D_mps, D_mpo)
    A₁ = mps_tensor(d, D_mps)
    A₂ = mps_tensor(d, D_mps)
    B₁ = mps_tensor(d, D_mps)
    B₂ = mps_tensor(d, D_mps)
    GL = mps_tensor(D_mpo, D_mps)
    GR = mps_tensor(D_mpo, D_mps)
    O₁ = mpo_tensor(d, D_mpo)
    O₂ = mpo_tensor(d, D_mpo)
    return @benchmarkable mps_expval_2($A₁, $A₂, $B₁, $B₂, $GL, $O₁, $O₂, $GR)
end

function benchmark_mps_dAC_1(d, D_mps, D_mpo)
    A = mps_tensor(d, D_mps)
    GL = mps_tensor(D_mpo, D_mps)
    GR = mps_tensor(D_mpo, D_mps)
    O = mpo_tensor(d, D_mpo)
    return @benchmarkable mps_dAC_1($A, $GL, $O, $GR)
end

function benchmark_mps_dAC_2(d, D_mps, D_mpo)
    A₁ = mps_tensor(d, D_mps)
    A₂ = mps_tensor(d, D_mps)
    GL = mps_tensor(D_mpo, D_mps)
    GR = mps_tensor(D_mpo, D_mps)
    O₁ = mpo_tensor(d, D_mpo)
    O₂ = mpo_tensor(d, D_mpo)
    return @benchmarkable mps_dAC_2($A₁, $A₂, $GL, $O₁, $O₂, $GR)
end

g_transfer = addgroup!(SUITE, "transfer")
g_expval_1 = addgroup!(SUITE, "expval_1")
g_expval_2 = addgroup!(SUITE, "expval_2")
g_dAC_1 = addgroup!(SUITE, "dAC_1")
g_dAC_2 = addgroup!(SUITE, "dAC_2")

for d in D_PHYS, D_mps in D_MPS_VIRT
    g_transfer[d, D_mps] = benchmark_mps_transfer(d, D_mps)
    for D_mpo in D_MPO_VIRT
        g_expval_1[d, D_mps, D_mpo] = benchmark_mps_expval_1(d, D_mps, D_mpo)
        g_expval_2[d, D_mps, D_mpo] = benchmark_mps_expval_2(d, D_mps, D_mpo)
        g_dAC_1[d, D_mps, D_mpo] = benchmark_mps_dAC_1(d, D_mps, D_mpo)
        g_dAC_2[d, D_mps, D_mpo] = benchmark_mps_dAC_2(d, D_mps, D_mpo)
    end
end

for g in (g_transfer, g_expval_1, g_expval_2, g_dAC_1, g_dAC_2)
    for b in values(g)
        b.params.seconds = 20
    end
end

end