/*

 https://abertschi.ch/blog/2022/prefetching/

*/

/*

Summary: in this code even removing the prefetcher realted code, the memory access still shows cache hit.
It should be realted with the hardware cache prefetcher enabled , the access pattern is one cacheline after another,
which is easy to detect. so after several cache misses, hw prefetcher is enabled. Turn off the hardware prefetcher and
comments the prefecher code, latency would increase. It proves hardware prefetcher is disalbed.


How to turn off the hw prefetcher on core 55 of SPR.
https://askubuntu.com/questions/1284908/disable-prefetching-on-intel-processor-on-20-04


```
sudo modprobe msr
echo on | sudo tee /sys/module/msr/parameters/allow_writes
sudo rdmsr 0x1a4h -p 55 //Read the MSR registers about the prefetcher disable, default is 20(0x20)
sudo wrmsr 0x1a4h  0x27 -p 55
```
MSR offset address is in IA64_software_dev_manual 2.17, volume 4

file:///C:/Users/luweizho/Downloads/325462-sdm-vol-1-2abcd-3abcd-4.pdf

Apply sw prefetcher after disabling hw prefetcher, manually enalbe sw prefetcher would have some overhead in the first 3-6 cache line access.

The following test data size is in L1, in L2 and out L2, but via prefetching in advance, we can ensure the data is prefetched into specified cache(L1 or L2) via difference prefetch ISAs.

test :
data size :1 kB, 2M, 4M
prefetchnt0:  18-20ns for L1 hit
prefetchnta:  18-20ns for L1 hit
prefetcht1:  22-24ns for L2 hit
prefetcht2:  22-24ns for L2 hit

diff is about 2ns for L1 and L2 access.

but the software overhead timing should be same for nta and t2. So let us check the diff between L1 and L2 with mlc tool

sudo ./mlc --idle_latency -b2k -c0 -t10  //2k < 32 K so L1 cache hit.

sudo ./mlc --idle_latency -b1024k -c0 -t10  // 2M /2k =1024, so only 1/1024 is L1 hit,  almost L2 hit.

L1 hit is 1.3 ns, L2 hit is 4.1 ns. the diff is 2.8 ns. So diff is almost same with above. L3 is 40ns

Double check the overhead of timmer??

1. mlc --idle_latency -b1024m -c0 -t20 , read mostly from DDR is about 109ns
2. disable prefetcher in this code  and hw prefetch, latency is about 120-130. So it means there are about 20-30 ns overhead to calculate the timing.


Remaining: the first 3-6 access would have big latency,software prefetcher pipeline warm up?? not sure.

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
    lfence();

    // delta tsc
    rdtsc(); // EDX:EAX
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


//The prefetch distacne needs to be tuned based on programing loop latency.
#define DIS_SCHEDULE 3
#define FMA_UNROLL 80

void test_software_prefetch() {
    MeasureAccess measure_access;
    Prefetch prefetcher(true, FMA_UNROLL);
    // 1MB. Half L2 size on SPR.
    uint64_t nbytes = 512 * 1024;
#if ALLOC_VIA_MMAP
    // Allocate 8 * 2M
    auto* data = reinterpret_cast<uint8_t*>(mmap(NULL, 8 * 1 << 21, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                 -1, 0));
#else
    auto*p = reinterpret_cast<int*>(aligned_alloc(64, nbytes));
    auto* data = reinterpret_cast<uint8_t*>(p);
#endif

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
#if ALLOC_VIA_MMAP
    munmap(data, 8*1<<21);
#else
    free(data);
#endif
}

int main(int args_n, char** args_list) {

    test_software_prefetch();
    return 0;
}