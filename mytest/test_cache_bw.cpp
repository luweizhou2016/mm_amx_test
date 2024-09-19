/*

*/

/*

Summary:

[CACHE JIT READ BW TEST] :  1024 KB read BW 105.917641 Byte/Cycle
[CACHE SUM READ BW TEST] :  1024 KB read BW 68.070602 Byte/Cycle
[CACHE JIT READ BW TEST] : 20480 KB read BW 13.528506 Byte/Cycle
[CACHE SUM READ BW TEST] : 20480 KB read BW 13.761051 Byte/Cycle
[CACHE JIT READ BW TEST] :    10 KB read BW 182.145477 Byte/Cycle
[CACHE SUM READ BW TEST] :    10 KB read BW 62.038120 Byte/Cycle

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
  
  MeasureBW(size_t cl_size): cacheline_size(cl_size)  {
    create_kernel("MeasureBW");
  }

  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;     //RDI

  Xbyak::Reg64 reg_repeat = abi_param2;    //RSI
  Xbyak::Reg64 reg_loop = abi_param4;     //RCX
  Xbyak::Reg64 reg_addr_bak = abi_param5; //R8

  Xbyak::Reg64 reg_tsc_0 = r11;

  size_t cacheline_size;
  size_t unroll_max_cl = 1024;

  void generate() {
    Xbyak::Label loop_over_steps;
    Xbyak::Label outerloop;
    mov(reg_addr_bak, reg_addr);

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
    L(loop_over_steps);
    size_t outer_loop = cacheline_size / unroll_max_cl;
    size_t tail =  cacheline_size % unroll_max_cl;
    if (outer_loop) {
        mov(reg_loop, outer_loop);
        L(outerloop);
        for (auto i = 0 ; i < unroll_max_cl; i ++){
        // dummy access
            Xbyak::Zmm zmmload  = Xbyak::Zmm(i % 32);
            vmovups(zmmload, ptr[reg_addr + i * 64]);
        }
        mfence();
        add(reg_addr,  64*unroll_max_cl);
        dec(reg_loop);
        jnz(outerloop, T_NEAR);
    }
    if (tail) {
        for (auto i = 0 ; i < tail; i ++){
            Xbyak::Zmm zmmload  = Xbyak::Zmm(i % 32);
            vmovups(zmmload, ptr[reg_addr + i * 64]);

        }
        mfence();

    }

    dec(reg_repeat);
    mov(reg_addr, reg_addr_bak);
    jnz(loop_over_steps, T_NEAR);

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

void test_cache_bandwidth(size_t nbytes, size_t repeat) {

    Prefetch prefetcher(false, 3);
    size_t access_cl_num = nbytes/64;
    MeasureBW measure_bw(access_cl_num);
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
    for(int i = 0; i < nbytes; i++) data[i] = 1;

    for(int cache_line = 0; cache_line < nbytes/64 ; cache_line ++) {
        _mm_clflush(data + cache_line*64);
    }
    _mm_mfence();

    for (int cache_line = 0; cache_line < nbytes/64; cache_line ++) {
        prefetcher(data + cache_line*64);
    }
    _mm_mfence();

    uint64_t cnt0 = _rdtsc();
    uint64_t timing = measure_bw(data, repeat);
    uint64_t cnt1 = _rdtsc();
    float diff_perc = (float)(cnt1 - cnt0 - timing) / (float)timing;

    assert(abs(diff_perc) <  0.001);

    printf("[CACHE JIT READ BW TEST] : %5lu KB read BW %3f Byte/Cycle\n", nbytes/1024, (float)(access_cl_num*64*repeat)/(float)(timing));
    // uint64_t sum = 0;
    //  _mm_mfence();

    // for (int cache_line = 0; cache_line < nbytes/64; cache_line ++) {
    //     prefetcher(data + cache_line*64);
    // }
    // _mm_mfence();

    // _mm_mfence();
    // cnt0 = _rdtsc();
    // uint64_t* data_qw = (uint64_t*)(data);
    // for (auto i = 0; i < repeat; i ++) {
    //     for (size_t idx = 0; idx < nbytes / 8; idx++) {
    //        sum += data_qw[idx];
    //     }
    // }
    // _mm_mfence();
    // cnt1 = _rdtsc();
    // assert(sum!=0);

    // printf("[CACHE SUM READ BW TEST] : %5lu KB read BW %3f Byte/Cycle\n", nbytes/1024, (float)(access_cl_num*64*repeat)/(float)(cnt1-cnt0));
#if ALLOC_VIA_MMAP
    munmap(data, 8*1<<21);
#else
    free(data);
#endif
}

int main(int args_n, char** args_list) {
    test_cache_bandwidth(2*1024*1024/2, 20000);
    test_cache_bandwidth(20*1024*1024, 2000);
    test_cache_bandwidth(10*1024, 10000);

    return 0;
}