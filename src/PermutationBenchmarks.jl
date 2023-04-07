module PermutationBenchmarks

using BenchmarkTools
using TensorOperations
using Random
using StableRNGs

samerand(args...) = rand(StableRNG(1), args...)

const SUITE = BenchmarkGroup()

function benchmark_perm(eltype, sz, p; conj=false)
    A = samerand(eltype, sz)
    B = samerand(eltype, getindex.(Ref(sz), p))
    pA = (p, ())
    conjA = conj ? :C : :N
    α = one(eltype)
    β = zero(eltype)

    return @benchmarkable tensoradd!($backend, $B, $A, $pA, $conjA, $α, $β)
end

# distribute `n` elements over an array where the dimensions are divided according to some ratio
function distribute_size(n, ratio::NTuple{N,Int}) where N
    k = (n / prod(ratio))^(1 / length(ratio))
    return map(r -> round(Int, k * r), ratio)
end

const ELTYPES = (Float64, ComplexF64)
const PERMS = (
    ((2,1),),
    ((1,3,2), (3,2,1), (2,1,3)),
    ((1,2,4,3), (4,3,2,1), (1,3,2,4), (4,1,3,2))
)
const RATIOS = (
    ((1,1), (1,2), (2,1)),
    ((1,1,1), (1,2,3), (1,1,5), (5,1,1)),
    ((1,1,1,1), (1,2,3,4), (4,3,2,1), (1,1,1,5))
)
const SIZES = round.(Int, (exp2.(range(3, 25, 30))))

function generate_benchmarks!(group::BenchmarkGroup=SUITE, ndims=2;
    perms=PERMS[ndims-1], eltypes=ELTYPES, sizes=SIZES, ratios=RATIOS[ndims-1])
    for eltype in eltypes, p in perms, r in ratios, n in sizes
        sz = distribute_size(n, r)
        actual_n = prod(sz)
        for conj in (eltype === complex(eltype) ? (true, false) : false)
            group[eltype, p, r, actual_n, conj] = benchmark_perm(eltype, sz, p; conj=conj)
        end
    end
end

generate_benchmarks!(SUITE, 2)
generate_benchmarks!(SUITE, 3)
generate_benchmarks!(SUITE, 4)

for g in values(SUITE)
    g.params.evals = 10
    g.params.samples = 100
    g.params.seconds = 600
end

end