/*

Summary: contineous physical memory seems to have almost same performance with malloc with aligned 64 bytes. 


[READ CACHE BW] :  1024 KB  99.915642 Byte/Cycle
[READ CACHE BW] : 20480 KB  13.302236 Byte/Cycle
[READ CACHE BW] :    10 KB  181.351273 Byte/Cycle
[WRITE CACHE BW] :  1024 KB  31.369221 Byte/Cycle
[WRITE CACHE BW] : 20480 KB  8.585435 Byte/Cycle
[WRITE CACHE BW] :    10 KB  105.450256 Byte/Cycle

MMAP physical contineous memory:
READ CACHE BW] :  1024 KB  100.574318 Byte/Cycle
[READ CACHE BW] : 20480 KB  13.301422 Byte/Cycle
[READ CACHE BW] :    10 KB  180.907867 Byte/Cycle
[WRITE CACHE BW] :  1024 KB  31.534115 Byte/Cycle
[WRITE CACHE BW] : 20480 KB  8.757294 Byte/Cycle
[WRITE CACHE BW] :    10 KB  105.984833 Byte/Cycle


Access latency: Physical contineous memory would have lower access latency because of avoiding cache conflicts. 
Access BW: same bandwidth between physical contineous memory and malloc.

*/
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>
#include "misc.hpp"
#include "timeit.hpp"

#include "jit.hpp"

#define ALLOC_VIA_MMAP 1

timeit timer({
    //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
    
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x02d1, "L2_HIT"},  // https://github.com/intel/perfmon/blob/2dfe7d466d46e89899645c094f8a5a2b8ced74f4/SPR/events/sapphirerapids_core.json#L7397
    //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x08d1, "L1_MISS"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x40d1, "FB_HIT"}, 
    {PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x04d1, "L3_HIT"}, {PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    //{PERF_TYPE_RAW, 0x81d0, "ALL_LOADS"},        // MEM_INST_RETIRED.ALL_LOADS

    //{PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    
    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});

class MeasureBW : public jit_generator {
 public:
  
  MeasureBW(size_t cl_size, bool store_test = false): cacheline_size(cl_size), store(store_test) {
    create_kernel("MeasureBW");
  }

  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;     //RDI

  Xbyak::Reg64 reg_repeat = abi_param2;    //RSI
  Xbyak::Reg64 reg_loop = abi_param4;     //RCX
  Xbyak::Reg64 reg_addr_bak = abi_param5; //R8
  Xbyak::Reg64 reg_store = r10; //R8

  Xbyak::Reg64 reg_tsc_0 = r11; //r11

  size_t cacheline_size;
  size_t unroll_max_cl = 500;
  bool store;

  void generate() {
    Xbyak::Label repeat_loop;
    Xbyak::Label loop_unroll;
    mov(reg_addr_bak, reg_addr);
    //Set the store value.
    //The value can be used to check whether loop works properly.
    mov(reg_store, 0x0202020202020202);
    auto zmm0 = Xbyak::Zmm(0);
    vpbroadcastq(zmm0, reg_store);

    mfence();
    lfence();

    // reg_tsc_0
    rdtsc(); // EDX:EAX
    lfence();

    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    mov(reg_tsc_0, rax);
    mfence();

    align(64, false);
    L(repeat_loop);
    size_t loop = cacheline_size / unroll_max_cl;
    size_t tail =  cacheline_size % unroll_max_cl;
    if (loop) {
        mov(reg_loop, loop);
        L(loop_unroll);
        for (auto i = 0 ; i < unroll_max_cl; i ++){
            if (store) {
                // vmovups(zmm, ptr[reg_addr + i * 64]);
                // vpsllw(zmm, zmm, 1);
                vmovups(ptr[reg_addr + i * 64], zmm0);
            } else {
                Xbyak::Zmm zmm = Xbyak::Zmm(i % 32);
                vmovups(zmm, ptr[reg_addr + i * 64]);
            }
        }
        mfence();
        add(reg_addr,  64*unroll_max_cl);
        dec(reg_loop);
        jnz(loop_unroll, T_NEAR);
    }
    if (tail) {
        for (auto i = 0 ; i < tail; i ++) {
            Xbyak::Zmm zmm  = Xbyak::Zmm(i % 32);
            if (store) {
                vmovups(ptr[reg_addr + i * 64], zmm0);
            } else {
                vmovups(zmm, ptr[reg_addr + i * 64]);
            }

        }
        mfence();
    }

    dec(reg_repeat);
    mov(reg_addr, reg_addr_bak);
    jnz(repeat_loop, T_NEAR);

    mfence();
    lfence();

    // delta tsc
    rdtsc(); // EDX:EAX
    lfence();

    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    sub(rax, reg_tsc_0);
    mfence();

    ret();
  }
};

class Prefetch : public jit_generator {
 using Vmm = Xbyak::Zmm;

 public:
  Prefetch(bool prefetch, uint32_t loop = 20) : via_prefetch(prefetch), unroll_loop(loop) {
    create_kernel("PREFETCH data");
  }
  Xbyak::Reg64 reg_addr = abi_param1;

  void generate() {
    if (via_prefetch) {
        // prefetchnta(ptr[reg_addr]);
        // prefetcht0(ptr[reg_addr]);
        prefetcht1(ptr[reg_addr]);

        for (uint32_t i = 0; i < unroll_loop; i++) {
            vfmadd231ps(Vmm(0), Vmm(1), Vmm(2));
        }
    } else {
        // mfence();
        vpxorq(zmm0, zmm0, zmm0);
        vmovups(zmm1, ptr[reg_addr]);
        vaddps(zmm0, zmm0, zmm1);
        mfence();
  }
  ret();
  }

  private:
  uint32_t unroll_loop;
  bool via_prefetch;
};

void test_cache_bandwidth(size_t cl_num, size_t repeat, bool test_store=false) {
    Prefetch prefetcher(false, 3);
    size_t nbytes = cl_num *64;
    MeasureBW measure_bw(cl_num, test_store);
#if ALLOC_VIA_MMAP
    //Allocate 32M huge pages.
    auto* data = reinterpret_cast<uint8_t*>(mmap(NULL, 32 * 1 << 21, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                 -1, 0));
#else
    //only Allocated needed memory.
    auto*p = reinterpret_cast<int*>(aligned_alloc(64, nbytes));
    auto* data = reinterpret_cast<uint8_t*>(p);
#endif
    double tsc = 0.0;
    // initialize data value
    for(int i = 0; i < nbytes; i++) data[i] = 255;

    for(int cache_line = 0; cache_line < nbytes/64 ; cache_line ++) {
        _mm_clflush(data + cache_line*64);
    }
    _mm_mfence();

    for (int cache_line = 0; cache_line < nbytes/64; cache_line ++) {
        //prefetch cache line via loading
        prefetcher(data + cache_line*64);
    }
    _mm_mfence();

    uint64_t cnt0 = _rdtsc();
    uint64_t timing = measure_bw(data, repeat);
    uint64_t cnt1 = _rdtsc();
    float diff_perc = (float)(cnt1 - cnt0 - timing) / (float)timing;
    //Ensure rdtsc works in JIT code.
    assert(abs(diff_perc) <  0.001);
    if (test_store) {
        for (int i = 0; i < nbytes; i ++) {
            //Ensure all cachelines can be stored. JIT code logic is ok.
            assert(data[i] == 2);
        }
    }

    printf("[%s CACHE BW] : %5lu KB  %3f Byte/Cycle\n", test_store ? "WRITE": "READ", nbytes/1024, (float)(cl_num*64*repeat)/(float)(timing));

#if ALLOC_VIA_MMAP
    munmap(data, 8*1<<21);
#else
    free(data);
#endif
}

//Enusre the cache line would be in L1, L2, L3 .
static std::vector<std::tuple<size_t, size_t>> access_cl = {{10*1024/64, 10000},{(1<<20)/64, 20000}, {(20*1<<20)/64, 2000}};

int main(int args_n, char** args_list) {
    // test read bandwidth
    for (auto cl : access_cl) {
        test_cache_bandwidth(std::get<0>(cl), std::get<1>(cl), false);
    }
    //test write bandwidth
    for (auto cl : access_cl) {
        test_cache_bandwidth(std::get<0>(cl), std::get<1>(cl), true);
    }
}