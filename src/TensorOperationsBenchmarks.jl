module TensorOperationsBenchmarks

using BenchmarkTools

const PARAMS_PATH = joinpath(dirname(@__FILE__), "..", "etc", "params.json")
const SUITE = BenchmarkGroup()
const MODULES = Dict(
    "contract" => :ContractBenchmarks,
)

include("contract/ContractBenchmarks.jl")

"""
    load!([group::BenchmarkGroup], id::AbstractString; tune=true)

Load a benchmark group. If `tune` is `true`, also load the precomputed benchmark parameters.
"""
load!(id::AbstractString; kwargs...) = load!(SUITE, id; kwargs...)
function load!(group::BenchmarkGroup, id::AbstractString; tune::Bool=true, kwargs...)
    modsym = MODULES[id]
    modsuite = Core.eval(TensorOperationsBenchmarks, modsym).generate_benchmarks(; kwargs...)
    group[id] = modsuite
    if tune
        results = BenchmarkTools.load(PARAMS_PATH)[1]
        haskey(results, id) && loadparams!(modsuite, results[id], :evals)
    end
    return group
end

"""
    loadall!([group::BenchmarkGroup]; tune=true)

Load all benchmark groups. If `tune` is `true`, also load the precomputed benchmark parameters.
"""
loadall!(; kwargs...) = loadall!(SUITE; kwargs...)
function loadall!(group::BenchmarkGroup; verbose::Bool=true, tune::Bool=true)
    for id in keys(MODULES)
        if verbose
            print("loading group $(repr(id))... ")
            time = @elapsed load!(group, id, tune=false)
            println("done (took $time seconds)")
        else
            load!(group, id, tune=false)
        end
    end
    if tune
        results = BenchmarkTools.load(PARAMS_PATH)[1]
        for (id, suite) in group
            haskey(results, id) && loadparams!(suite, results[id], :evals)
        end
    end
    return group
end

function __init__()
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000
    BenchmarkTools.DEFAULT_PARAMETERS.time_tolerance = 0.15
    BenchmarkTools.DEFAULT_PARAMETERS.memory_tolerance = 0.01
end

end
