/*

 https://abertschi.ch/blog/2022/prefetching/

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

class MeasureAccess : public jit_generator {
 public:
  
  MeasureAccess() {
    create_kernel("MeasureAccess");
  }
  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;
  Xbyak::Reg64 reg_cycles = rax;
  Xbyak::Reg64 reg_tsc_0 = r9;
  Xbyak::Reg64 reg_dummy = r10;

  void generate() {

    mfence();

    // reg_tsc_0
    rdtsc(); // EDX:EAX
    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    mov(reg_tsc_0, rax);

    mfence();

    // dummy access
    vmovups(zmm0, ptr[reg_addr]);
    mfence();

    // delta tsc
    rdtsc(); // EDX:EAX
    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    sub(rax, reg_tsc_0);

    ret();
  }
};

class Prefetch : public jit_generator {
 using Vmm = Xbyak::Zmm;

 public:
  Prefetch(bool prefetch, uint32_t loop = 20) : via_prefetch(prefetch), unroll_loop(loop) {
    create_kernel("DummyFMA");
  }
  Xbyak::Reg64 reg_addr = abi_param1;

  void generate() {
    if (via_prefetch) {
        prefetchnta(ptr[reg_addr]);
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

// void spin_wait(double seconds){
//     auto wait_tsc = second2tsc(seconds);
//     auto t0 = __rdtsc();
//     while(__rdtsc() - t0 < wait_tsc);
// }

// void wait(){
//         auto wait_tsc = second2tsc(1e-5);
//         auto t0 = __rdtsc();
//         while(__rdtsc() - t0 < wait_tsc);
// };


class AccessLoad : public jit_generator {
 public:
  AccessLoad() {
    create_kernel("AccessLoad");
  }
  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;
  void generate() {
    mfence();
    vpxorq(zmm0, zmm0, zmm0);
    vmovups(zmm1, ptr[reg_addr]);
    vaddps(zmm0, zmm0, zmm1);
    mfence();
    ret();
  }
};

//The prefetch distacne needs to be tuned based on programing loop latency.
#define DIS_SCHEDULE 3
#define FMA_UNROLL 20

void test_prefetch_nta() {
    MeasureAccess measure_access;
    Prefetch prefetcher(true, FMA_UNROLL);
    // 1MB
    uint64_t nbytes = 40*1024;
    auto* data = reinterpret_cast<uint8_t*>(aligned_alloc(4096, nbytes + 64 * DIS_SCHEDULE));
    auto *nbars =  reinterpret_cast<int*>(aligned_alloc(64, nbytes/64*sizeof(int)));
    auto *access_timing =  reinterpret_cast<double*>(aligned_alloc(64, nbytes/64*sizeof(double)));
    double tsc = 0.0;
    for(int i = 0; i < nbytes; i++) data[i] = 1;
    int32_t sum = 0;

    // std::vector<uint8_t> big_buffer(1024*1024*128, 0);
    // memset(&big_buffer[0], 1, big_buffer.size());
    // load_prefetch_L2(&big_buffer[0], big_buffer.size());
    for(int cache_line = 0; cache_line < nbytes/64 + DIS_SCHEDULE; cache_line ++) {
        _mm_clflush(data + cache_line*64);
    }
    _mm_mfence();

    for (int cache_line = 0; cache_line < DIS_SCHEDULE; cache_line ++) {
        prefetcher(data + cache_line*64);
    }

    // clear cache by memset & load
    for(int cache_line = 0; cache_line < nbytes/64; cache_line ++) {
        prefetcher(data + (cache_line + DIS_SCHEDULE)*64);
        auto access_time = measure_access(data + cache_line*64);
        tsc = tsc2second(access_time);

        nbars[cache_line] = static_cast<int>(tsc*1e9 * 100/256);
        access_timing[cache_line] = tsc;
    }

    for (int cache_line = 0; cache_line < nbytes/64; cache_line ++) {
        // https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
        //if (std::find(read_pattern.begin(), read_pattern.end(), cache_line) != read_pattern.end()) {
        //    fg_color = "32";
        //}
        const char * fg_color = "90";
        std::string progress_bar;
        // https://github.com/Changaco/unicode-progress-bars/blob/master/generator.html
        auto const& bar = nbars[cache_line];
        for(int i = 0; i < bar; i++) progress_bar += "▅";// "█";
        printf(" cache_line[%3d] : %6.2f ns : \033[1;%sm %s \033[0m\n", cache_line, access_timing[cache_line]*1e9, fg_color, progress_bar.c_str());
    }
    ::free(access_timing);
    ::free(nbars);
    ::free(data);
}

int main() {
    test_prefetch_nta();
    return 0;
}