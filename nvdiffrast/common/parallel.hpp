#pragma once
#include "BS_thread_pool.hpp"

inline void ParallelFor(size_t Num, int NumThreads, std::function<void(size_t)> Func)
{
    if (Num < NumThreads)
        NumThreads = Num;
    size_t num_per_thread = Num / NumThreads;
    static BS::thread_pool pool;
    BS::multi_future<void> loop_future = pool.parallelize_loop(0, Num, [&](size_t first, size_t second) {
        for (size_t i = first; i < second; i++)
            Func(i);
    }, num_per_thread);
    loop_future.wait();
}