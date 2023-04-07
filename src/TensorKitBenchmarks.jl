module TensorKitBenchmarks

using BenchmarkTools
using TensorOperations
using Random
using StableRNGs

samerand(args...) = rand(StableRNG(1), args...)

const SUITE = BenchmarkGroup()
SUITE["mpo"] = BenchmarkGroup()
SUITE["pepo"] = BenchmarkGroup()
SUITE["mera"] = BenchmarkGroup()

function mpo_fun(A, M, FL, FR)
    @tensor C = FL[4, 2, 1] * A[1, 3, 6] * M[2, 5, 3, 7] * conj(A[4, 5, 8]) * FR[6, 7, 8]
    return C
end

function benchmark_mpo(eltype, Dmps, Dmpo, Dphys)
    A = samerand(eltype, Dmps, Dphys, Dmps)
    M = samerand(eltype, Dmpo, Dphys, Dphys, Dmpo)
    FL = samerand(eltype, Dmps, Dmpo, Dmps)
    FR = samerand(eltype, Dmps, Dmpo, Dmps)

    return @benchmarkable mpo_fun($A, $M, $FL, $FR)
end

function pepo_fun(A, P, FL, FD, FR, FU)
    @tensor C = FL[18, 7, 4, 2, 1] * FU[1, 3, 6, 9, 10] * A[2, 17, 5, 3, 11] *
                P[4, 16, 8, 5, 6, 12] * conj(A[7, 15, 8, 9, 13]) * FR[10, 11, 12, 13, 14] *
                FD[14, 15, 16, 17, 18]
    return C
end

function benchmark_pepo(eltype, Dpeps, Dpepo, Dphys, Denv)
    A = samerand(eltype, Dpeps, Dpeps, Dphys, Dpeps, Dpeps)
    P = samerand(eltype, Dpepo, Dpepo, Dphys, Dphys, Dpepo, Dpepo)
    FL = samerand(eltype, Denv, Dpeps, Dpepo, Dpeps, Denv)
    FD = samerand(eltype, Denv, Dpeps, Dpepo, Dpeps, Denv)
    FR = samerand(eltype, Denv, Dpeps, Dpepo, Dpeps, Denv)
    FU = samerand(eltype, Denv, Dpeps, Dpepo, Dpeps, Denv)

    return @benchmarkable pepo_fun($A, $P, $FL, $FD, $FR, $FU)
end

function mera_fun(u, w, ρ, h)
    @tensor C = (((((((h[9, 3, 4, 5, 1, 2] * u[1, 2, 7, 12]) * conj(u[3, 4, 11, 13])) *
                     (u[8, 5, 15, 6] * w[6, 7, 19])) *
                    (conj(u[8, 9, 17, 10]) * conj(w[10, 11, 22]))) *
                   ((w[12, 14, 20] * conj(w[13, 14, 23])) * ρ[18, 19, 20, 21, 22, 23])) *
                  w[16, 15, 18]) * conj(w[16, 17, 21]))
    return C
end

function benchmark_mera(eltype, Dmera)
    u = samerand(eltype, Dmera, Dmera, Dmera, Dmera)
    w = samerand(eltype, Dmera, Dmera, Dmera)
    ρ = samerand(eltype, Dmera, Dmera, Dmera, Dmera, Dmera, Dmera)
    h = samerand(eltype, Dmera, Dmera, Dmera, Dmera, Dmera, Dmera)

    return @benchmarkable mera_fun($u, $w, $ρ, $h)
end

const ELTYPES = (Float64, ComplexF64)
const MPO_DIMS = (
    [10, 40, 160, 640, 2560, 100, 200, 300],    # mps
    [4, 4, 4, 4, 4, 10, 10, 20],                # mpo
    [3, 3, 3, 3, 3, 10, 10, 20],                # phys
)
const PEPO_DIMS = (
    [3, 3, 4, 4, 5, 5, 6, 6],               # peps
    [2, 3, 2, 3, 2, 2, 2, 3],               # pepo
    [2, 3, 2, 3, 2, 3, 2, 2],               # phys
    [50, 100, 50, 100, 50, 100, 50, 100],   # env
)
const MERA_DIMS = (
    [2, 3, 4, 8, 12, 16],
)

function generate_benchmarks!(group::BenchmarkGroup=SUITE;
    eltypes=ELTYPES, mpo_dims=MPO_DIMS, pepo_dims=PEPO_DIMS, mera_dims=MERA_DIMS)

    for eltype in eltypes
        for (Dmps, Dmpo, Dphys) in zip(mpo_dims...)
            group["mpo"][eltype, Dmps, Dmpo, Dphys] =
                benchmark_mpo(eltype, Dmps, Dmpo, Dphys)
        end
        for (Dpeps, Dpepo, Dphys, Denv) in zip(pepo_dims...)
            group["pepo"][eltype, Dpeps, Dpepo, Dphys, Denv] =
                benchmark_pepo(eltype, Dpeps, Dpepo, Dphys, Denv)
        end
        for (Dmera,) in zip(MERA_DIMS...)
            group["mera"][eltype, Dmera] = benchmark_mera(eltype, Dmera)
        end
    end
end

generate_benchmarks!(SUITE)

for (key, group) in SUITE
    for g in values(group)
        g.params.evals = 10
        g.params.samples = 100
        g.params.seconds = 600
    end
end

end
