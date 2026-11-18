# Apply f(i, workspace) for i = 1,…,n in parallel over tasks, with dynamic load balancing. Each task creates its own
# workspace by calling `workspace()` (e.g., `workspace = PottsWorkspace` for the hidden Potts sampler).
function _tforeach(f, n::Integer; ntasks::Integer = Threads.nthreads(), workspace = () -> nothing)
    n <= 0 && return nothing
    ntasks = max(1, min(ntasks, n))
    next = Threads.Atomic{Int}(1)
    if ntasks == 1
        _tforeach_worker(f, next, n, workspace)
    else
        @sync for _ = 1:ntasks
            Threads.@spawn _tforeach_worker(f, next, n, workspace)
        end
    end
    return nothing
end

function _tforeach_worker(f, next::Threads.Atomic{Int}, n::Integer, workspace)
    ws = workspace()
    while true
        i = Threads.atomic_add!(next, 1)
        i > n && break
        f(i, ws)
    end
    return nothing
end
