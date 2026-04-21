#include <cuda_runtime.h>

// 定义策略参数
#define THREADS_PER_BLOCK   256
#define ITEMS_PER_THREAD    8
#define ITEMS_PER_BLOCK     (ITEMS_PER_THREAD * THREADS_PER_BLOCK)

// 定义状态常量与掩码
#define STATE_EMPTY         0x00000000
#define STATE_HANGUP        0x40000000
#define STATE_FINISHED      0x80000000
#define HANGUP_MASK         0x3fffffff
#define FINISHED_MASK       0x7fffffff

// warp 级前缀和, 返回当前位置的前缀和
template <int warp_count>
__device__ __forceinline__ int warp_scan(int val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < warp_count; offset <<= 1) {
        int tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += tmp;
    }
    return val;
}

// 打包并设置计算结果以及状态
template <int state>
__device__ __forceinline__ void set_result_and_state(int logical_block_id, int result, int* __restrict__ status_vector) {
    atomicExch(status_vector + logical_block_id, result | state);
}

// 获取并解包计算结果以及状态
__device__ __forceinline__ void get_result_and_state(int logical_block_id, int& result, int& state, int* __restrict__ status_vector) {
    volatile int* p_state = status_vector + logical_block_id;
    int raw_state = *p_state;
    int finished = raw_state & STATE_FINISHED;
    state = finished ? finished : (raw_state & STATE_HANGUP);
    result = raw_state & (finished ? FINISHED_MASK : HANGUP_MASK);
}

// 整数向量结构体
struct __align__(ITEMS_PER_THREAD * sizeof(int)) vector_int {
    int data[ITEMS_PER_THREAD];
    __device__ __forceinline__ int& operator [] (int index) { return data[index]; }
};

__global__ void kernel_accumulate_blocks(
    const int* __restrict__ d_in, int* __restrict__ d_out,
    int n, int num_logical_blocks,
    int* __restrict__ next_unreached_logical_block,
    int* __restrict__ status_vector
) {
    // 声明共享内存
    __shared__ int s_warp_sums[32];
    __shared__ int s_logical_block;
    __shared__ int s_block_result;
    __shared__ int s_last_result;

    // 声明线程常量
    const int tid = threadIdx.x;
    const int base = tid * ITEMS_PER_THREAD;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int warp_count = THREADS_PER_BLOCK >> 5;

    while (true) {

        // 获取逻辑块
        if (tid == 0) s_logical_block = atomicAdd(next_unreached_logical_block, 1);
        __syncthreads();
        int logical_block = s_logical_block;
        if (logical_block >= num_logical_blocks) break;

        // 加载当前线程负责的元素
        const int g_idx = logical_block * ITEMS_PER_BLOCK + base;
        register vector_int thread_items;
        if (g_idx + ITEMS_PER_THREAD <= n) thread_items = *reinterpret_cast<const vector_int*>(d_in + g_idx);
        else if (g_idx >= n); // 什么也不干
        else [[unlikely]] { // 测试样例不会到达这一分支
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD - 1; ++i) if (g_idx + i < n) thread_items[i] = d_in[g_idx + i];
        }
        
        // thread 级前缀和
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_THREAD; ++i) thread_items[i] += thread_items[i - 1];
        int thread_sum = thread_items[ITEMS_PER_THREAD - 1];

        // warp 级前缀和
        int warp_sum = warp_scan<32>(thread_sum, lane_id);
        
        // warp 间前缀和
        if (lane_id == 31) s_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        if (warp_id == 0) {
            int _warp_sum = (lane_id < warp_count) ? s_warp_sums[lane_id] : 0;
            _warp_sum = warp_scan<warp_count>(_warp_sum, lane_id);
            s_warp_sums[lane_id] = _warp_sum;
            if (lane_id == warp_count - 1) s_block_result = _warp_sum;
        }
        __syncthreads();

        // 获取局部偏移
        int warp_exclusive = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
        int local_offset = warp_exclusive + warp_sum - thread_sum;
        
        // 获取依赖
        if (tid == 0) {
            int block_result = s_block_result;
            // 第 0 个逻辑块没有依赖
            if (logical_block == 0) {
                // 标记为已完成
                set_result_and_state<STATE_FINISHED>(logical_block, block_result, status_vector);
                s_last_result = 0;
            }
            // 依赖前一个逻辑块的结果
            else {
                // 标记为挂起
                set_result_and_state<STATE_HANGUP>(logical_block, block_result, status_vector);
                int last_logical_block = logical_block - 1;
                int accumulate = 0;
                // 循环地向前累加
                while (last_logical_block >= 0) {
                    int last_state;
                    int last_result;
                    get_result_and_state(last_logical_block, last_result, last_state, status_vector);
                    // 如果上一个逻辑块已完成, 则直接拿到结果, 结束累加
                    if (last_state == STATE_FINISHED) {
                        accumulate += last_result;
                        break;
                    }
                    // 如果上一个逻辑块被挂起, 则说明计算了块内总和, 继续向前累加
                    else if (last_state == STATE_HANGUP) {
                        accumulate += last_result;
                        --last_logical_block;
                    }
                    // 否则上一个逻辑块未完成计算, 等待即可 (为 SMX 多分配几个 block , 可以减小等待开销)
                    else;
                }
                s_last_result = accumulate;
                // 设置全局总和并完成
                set_result_and_state<STATE_FINISHED>(logical_block, block_result + accumulate, status_vector);
            }
        }
        __syncthreads();
        
        // 获取全局偏移
        int global_offset = local_offset + s_last_result;
        
        // 应用偏移
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            thread_items[i] += global_offset;
        }

        // 写回全局内存
        if (g_idx + ITEMS_PER_THREAD <= n) *reinterpret_cast<vector_int*>(d_out + g_idx) = thread_items;
        else if (g_idx >= n); // 什么也不干
        else [[unlikely]] { // 测试用例不会到达这个分支
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD - 1; ++i) if (g_idx + i < n) d_out[g_idx + i] = thread_items[i];
        }
    }
}



// ------------------------------------------------------------
//   d_in  - device 端输入数组（长度 n, int）
//   d_out - device 端输出数组（长度 n, int）
//   n     - 元素个数
// ------------------------------------------------------------
void prefix_sum(const int* d_in, int* d_out, int n) {
    int* logical_block_counter;
    int* status_vector;
    
    // 声明策略参数
    const int num_logical_blocks = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    constexpr int max_blocks = 13 * 16; // 根据 GPU 型号修改, 第一个值等于 SM 数量, 第二个值根据经验判断(20 左右)
    const int num_blocks = (num_logical_blocks > max_blocks) ? max_blocks : num_logical_blocks;

    cudaError_t error;

    // 分配逻辑块计数器, 并设置为 0
    error = cudaMalloc(&logical_block_counter, sizeof(int));
    while (error != cudaSuccess) error = cudaMalloc(&logical_block_counter, sizeof(int));
    error = cudaMemset(logical_block_counter, 0, sizeof(int));

    // 分配状态向量, 并设置为 EMPTY
    error = cudaMalloc(&status_vector, num_logical_blocks * sizeof(int));
    while (error != cudaSuccess) error = cudaMalloc(&status_vector, num_logical_blocks * sizeof(int));
    error = cudaMemset(status_vector, 0, num_logical_blocks * sizeof(int));

    kernel_accumulate_blocks <<<num_blocks, THREADS_PER_BLOCK>>> (d_in, d_out, n, num_logical_blocks, logical_block_counter, status_vector);
    cudaDeviceSynchronize();

    error = cudaFree(logical_block_counter);
    error = cudaFree(status_vector);
}
