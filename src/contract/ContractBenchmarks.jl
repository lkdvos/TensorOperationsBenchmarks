module ContractBenchmarks

using TensorOperations
using BenchmarkTools

const SUITE = BenchmarkGroup()
const CONTRACTIONS_PATH = joinpath(@__DIR__, "..", "..", "etc", "contractions.dat")

# Load contractions
#-------------------
function extract_sizes(contraction::AbstractString, sizes::Dict{String,Int})
    symbolsA = match(r"A\[([^\]]*)\]", contraction)
    szA = collect(sizes[s] for s in split(symbolsA.captures[1], ","; keepempty=false))
    symbolsB = match(r"B\[([^\]]*)\]", contraction)
    szB = collect(sizes[s] for s in split(symbolsB.captures[1], ","; keepempty=false))
    symbolsC = match(r"C\[([^\]]*)\]", contraction)
    szC = collect(sizes[s] for s in split(symbolsC.captures[1], ","; keepempty=false))
    return szA, szB, szC
end

for line in eachline(CONTRACTIONS_PATH)
    contraction, sizes = split(line, " & ")
    contract_expr = Meta.parse(contraction)
    
    subsizes = Dict{String, Int}()
    for (label, sz) in split.(split(sizes, "; "; keepempty=false), Ref(":"))
        subsizes[label] = parse(Int, sz)
    end

    szA, szB, szC = extract_sizes(contraction, subsizes)
    
    SUITE[contraction, szC, szA, szB] = @benchmarkable @tensor(esc($contract_expr)) setup = (A = rand($szA...); B = rand($szB...); C = rand($szC...))
end


end # module
