module ContractBenchmarks

using TensorOperations
using BenchmarkTools

const CONTRACTIONS_PATH = joinpath(@__DIR__, "..", "..", "etc", "contractions.dat")

# Load contractions
#-------------------
function extract_labels(contraction::AbstractString)
    symbolsC = match(r"C\[([^\]]*)\]", contraction)
    labelsC = split(symbolsC.captures[1], ","; keepempty=false)
    symbolsA = match(r"A\[([^\]]*)\]", contraction)
    labelsA = split(symbolsA.captures[1], ","; keepempty=false)
    symbolsB = match(r"B\[([^\]]*)\]", contraction)
    labelsB = split(symbolsB.captures[1], ","; keepempty=false)
    return labelsC, labelsA, labelsB
end

function generate_benchmark(line::AbstractString;
                            T=Float64, backend=TensorOperations.StridedBLAS())
    contraction, sizes = split(line, " & ")
    contract_expr = Meta.parse(contraction)

    # extract labels
    labelsC, labelsA, labelsB = extract_labels(contraction)
    pA, pB, pC = TensorOperations.contract_indices(tuple(labelsA...), tuple(labelsB...), tuple(labelsC...))

    # extract sizes
    subsizes = Dict{String,Int}()
    for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref("="))
        subsizes[label] = parse(Int, sz)
    end
    szA = getindex.(Ref(subsizes), labelsA)
    szB = getindex.(Ref(subsizes), labelsB)
    szC = getindex.(Ref(subsizes), labelsC)
    
    return @benchmarkable tensorcontract!(C, $pC, A, $pA, :N, B, $pB, :N, true, false, $backend) setup = (A = rand($T, $szA...); B = rand($T, $szB...); C = rand($T, $szC...)) evals = 1
end

function generate_benchmarks(filename::AbstractString=CONTRACTIONS_PATH; kwargs...)
    suite = BenchmarkGroup([filename])
    for (i, line) in enumerate(eachline(filename))
        suite[i] = generate_benchmark(line; kwargs...)
    end
    return suite
end

end # module
