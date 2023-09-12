datadir = joinpath(joinpath(@__DIR__, "..", "data"))
!isdir(datadir) && mkdir(datadir)

#take a path of a results file and increment its prefix backup number
function increment_backup_num(filepath)
    path, filename = splitdir(filepath)
    fname, suffix = splitext(filename)
    m = match(r"^(.*)_#([0-9]+)$", fname)
    if m === nothing
        return joinpath(path, "$(fname)_#1$(suffix)")
    end
    newnum = string(parse(Int, m.captures[2]) +1)
    return joinpath(path, "$(m.captures[1])_#$newnum$(suffix)")
end

#recursively move files to increased backup number
function recursively_clear_path(cur_path)
    ispath(cur_path) || return
    new_path=increment_backup_num(cur_path)
    if ispath(new_path)
        recursively_clear_path(new_path)
    end
    mv(cur_path, new_path)
end

result_path = joinpath(datadir, endswith(ARGS[1], ".json") ? ARGS[1] : "$(ARGS[1]).json")
if isfile(result_path)
    @warn "File $result_path already exists"
    recursively_clear_path(result_path)
end
println("Saving results to $result_path")

# Setup
# -----
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MKL, ThreadPinning
using LinearAlgebra; BLAS.set_num_threads(1)
ThreadPinning.mkl_set_dynamic(false)
pinthreads(:cores)
threadinfo(; blas=true, hints=true)

# Load and run benchmarks
# -----------------------
import TensorOperationsBenchmarks as TOB
using BenchmarkTools
suite = TOB.loadall!(;)
result = run(suite; verbose=true)
BenchmarkTools.save(result_path, result)
