result_path = joinpath(@__DIR__, "..", "data", endswith(ARGS[1], ".json") ? ARGS[1] : "$(ARGS[1]).json")

# Setup
# -----
using MKL, ThreadPinning
using LinearAlgebra; BLAS.set_num_threads(1)
ThreadPinning.mkl_set_dynamic(false)
pinthreads(:cores)
threadinfo(; blas=true, hints=true)

# Load and run benchmarks
# -----------------------
import TensorOperationsBenchmarks as TOB
using BenchmarkTools
suite = TOB.loadall!(; tune=false)
result = run(suite; verbose=true)
BenchmarkTools.save(result_path, result)
