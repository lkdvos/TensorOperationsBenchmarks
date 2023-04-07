using TensorOperationsBenchmarks: TensorOperationsBenchmarks as TOB
using TensorOperations
using Strided
using TBLIS

TOB.loadall!()
const THREAD_NUMBERS = (8, 4, 1)


## Strided without blas, no cache
useblas = false
allocationbackend!(JuliaAllocator())
operationbackend!(StridedBackend(useblas))

for t in THREAD_NUMBERS
    Strided.set_num_threads(t)
    TOB.generate_data(TOB.SUITE, "Strided($useblas)_t$t.json")
end

## Strided without blas, with cache
useblas = false
allocationbackend!(TensorCache())
operationbackend!(StridedBackend(useblas))

for t in THREAD_NUMBERS
    Strided.set_num_threads(t)
    TOB.generate_data(TOB.SUITE, "Strided($useblas)_cached_t$t.json")
end

## TBLIS, no cache
allocationbackend!(JuliaAllocator())
operationbackend!(TBLISBackend())

for t in THREAD_NUMBERS
    TBLIS.set_num_threads(t)
    TOB.generate_data(TOB.SUITE, "TBLIS_t$t.json")
end

## TBLIS, with cache
allocationbackend!(TensorCache())
operationbackend!(TBLISBackend())

for t in THREAD_NUMBERS
    TBLIS.set_num_threads(t)
    TOB.generate_data(TOB.SUITE, "TBLIS_cached_t$t.json")
end