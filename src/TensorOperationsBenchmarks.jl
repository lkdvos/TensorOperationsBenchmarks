module TensorOperationsBenchmarks

using BenchmarkTools
using TensorOperations
using DrWatson: recursively_clear_path

const PARAMS_PATH = joinpath(dirname(@__FILE__), "..", "etc", "params.json")
const RESULTS_PATH = joinpath(dirname(@__FILE__), "..", "etc")
const SUITE = BenchmarkGroup()
const MODULES = Dict(
    # "permutations" => :PermutationBenchmarks,
    # "mps" => :MpsBenchmarks,
    "tensorkit" => :TensorKitBenchmarks,
)

load!(id::AbstractString; kwargs...) = load!(SUITE, id; kwargs...)

function load!(group::BenchmarkGroup, id::AbstractString; tune::Bool=true)
    modsym = MODULES[id]
    modpath = joinpath(dirname(@__FILE__), "$(modsym).jl")
    Core.eval(TensorOperationsBenchmarks, :(include($modpath)))
    modsuite = Core.eval(TensorOperationsBenchmarks, modsym).SUITE
    group[id] = modsuite
    if tune
        results = BenchmarkTools.load(PARAMS_PATH)[1]
        haskey(results, id) && loadparams!(modsuite, results[id], :evals)
    end
    return group
end

loadall!(; kwargs...) = loadall!(SUITE; kwargs...)

function loadall!(group::BenchmarkGroup; verbose::Bool=true, tune::Bool=false)
    for id in keys(MODULES)
        if verbose
            @info "loading group $(repr(id))... "
            time = @elapsed load!(group, id, tune=false)
            @info "done (took $time seconds)"
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

function generate_data(group::BenchmarkGroup=SUITE, filename="results.json";
    filepath=RESULTS_PATH, verbose=true, force=false)
    file = joinpath(filepath, filename)
    !force && isfile(file) && return BenchmarkTools.load(file)
    results = BenchmarkTools.run(group; verbose=verbose)
    recursively_clear_path(file)
    BenchmarkTools.save(file, results)
    return results
end



end
