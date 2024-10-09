/*

Summary: Test needs to run after disabling prefetch and with sudo to check physical address.

L2/L1 cache line on SPR is physically indexed based on test_cache_conflict.cpp test result.
 **virtual address from malloc()/alignmalloc() on SPR can't ensure virtual addresshas same set index with physical address**. 
Using **mmap() with huge page** can get physical contineous address within 2MB. So on SPR, L2 size is 2MB, so it means when allocating 2MB with mmap() hugepage, we can ensure these memory would introcue
L2 cache line conflict.


The test read data (size is 0.5 -0.75 * l2_size )into cache(at least L2), and probe cache line latency based on stride.
Based on latency data, we can conclude whether  the cache line conflict exist. 

When using malloc, we can see cache line conflict is easy to reproduce even using half of cache, which is caused physical address L2 cache index conflict
from the log.


https://stackoverflow.com/questions/5748492/is-there-any-api-for-determining-the-physical-address-from-virtual-address-in-li

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
  
  MeasureAccess(bool readOnly = false) :  read_only(readOnly) {
    create_kernel("MeasureAccess");
  }
  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;
  Xbyak::Reg64 reg_cycles = rax;
  Xbyak::Reg64 reg_tsc_0 = r9;
  Xbyak::Reg64 reg_dummy = r10;
  bool read_only;

  void generate() {

    mfence();
    if (!read_only) {
        // reg_tsc_0
        rdtsc(); // EDX:EAX
        sal(rdx, 32);
        or_(rax, rdx); // 64bit
        mov(reg_tsc_0, rax);

        mfence();
    }

    // dummy access
    vmovups(zmm0, ptr[reg_addr]);

    mfence();
    if (!read_only) {
        // delta tsc
        rdtsc(); // EDX:EAX
        sal(rdx, 32);
        or_(rax, rdx); // 64bit
        sub(rax, reg_tsc_0);
    }

    ret();
  }
};

void test_read_prefetch() {

    EnvVar WRMEM("WRMEM", 0);
    EnvVar CLSTRIDE("CLSTRIDE", 1);
    pid_t pid = getpid();

    MeasureAccess read_access(true);
    MeasureAccess measure_access;
    MSRConfig _msr1;
    size_t cache_line_sz = 64;
    size_t access_stride = cache_line_sz;
    size_t l2_sets = 2048;
    size_t l1_sets = 64;

    size_t l2_ways = 16;
    size_t l2_cache_set[2048] = {0};
    size_t l1_cache_set[64] = {0};

    constexpr static std::size_t huge_page_size = 1 << 21; // 2 MiB

    uint64_t nbytes = cache_line_sz * l2_sets * 8;
    // void* p;
    // posix_memalign(&p, 64, nbytes);  //posix_memalign(&p, huge_page_size, nbytes); seems can also allocate physical contineous pages sometimes.??
    // auto* data = reinterpret_cast<uint8_t*>(p);

    auto* data = reinterpret_cast<uint8_t*>(mmap(NULL, 8 * huge_page_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                 -1, 0));
    memset(data, 1, nbytes);
    std::vector<int64_t> access_times(nbytes/64, 0);

    const int repeats = 3;
    _mm_mfence();
    for(int r = 0; r < repeats; r ++) {

        for (int i = 0; i < nbytes; i += 64) {
            _mm_clflush(data + i);
        }
        _mm_mfence();
        size_t sum =0;
        for(int cache_line = 0; cache_line < access_times.size();  cache_line += access_stride/64) {
            // Read access to cached data into at least L2
            read_access(data + cache_line*64);
        }
        _mm_mfence();
        for(int cache_line = 0; cache_line < access_times.size();  cache_line += access_stride/64) {
            access_times[cache_line] += measure_access(data + cache_line*64);
        }
    }

    {
        //Calculate consumption times  on every cache line set.
        uintptr_t vaddr, paddr = 0;
        vaddr = (uintptr_t)data;
        for(int cache_line = 0; cache_line < access_times.size(); cache_line += access_stride/64) {
            if (virt_to_phys_user(&paddr, pid, vaddr)) {
                fprintf(stderr, "error: virt_to_phys_user\n");
            };
            // L2 is 0x800, L1 is 0x40 on SPR.
            size_t l2_set_idx = (paddr >> 6) & 0x7ff;
            size_t l1_set_idx = (paddr >> 6) & 0x3f;            
            l2_cache_set[l2_set_idx] = l2_cache_set[l2_set_idx] + 1;
            l1_cache_set[l1_set_idx] = l1_cache_set[l1_set_idx] + 1;
            vaddr +=access_stride;
        }
    }

    uintptr_t vaddr, paddr = 0;
    vaddr = (uintptr_t)data;
    // show access_times
    for(int cache_line = 0; cache_line < access_times.size(); cache_line += access_stride/64) {
        if (virt_to_phys_user(&paddr, pid, vaddr)) {
            fprintf(stderr, "error: virt_to_phys_user\n");
        };
        // https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
        const char * fg_color = "90";
        // if (std::find(read_pattern.begin(), read_pattern.end(), cache_line) != read_pattern.end()) {
        //     fg_color = "32";
        // }
        auto nbars = static_cast<int>(tsc2second(access_times[cache_line]/repeats)*1e9 * 100/256);
        std::string progress_bar;
        // https://github.com/Changaco/unicode-progress-bars/blob/master/generator.html
        for(int i = 0; i < nbars; i++) progress_bar += "▅";// "█";

        size_t l1_v_set_idx = (vaddr >> 6) & 0x3f;
        size_t l1_p_set_idx = (paddr >> 6) & 0x3f;
        size_t l2_v_set_idx = (vaddr >> 6) & 0x7ff;
        size_t l2_p_set_idx = (paddr >> 6) & 0x7ff;

        printf(" cache_line[%3d] %6.2f ns v2p[0x%jx] = 0x%jx l1set[%3lu:%3lu] = %3lu l2set[%4lu:%4lu] = %3lu : \033[1;%sm %s \033[0m\n", cache_line, tsc2second(access_times[cache_line]/repeats)*1e9, 
                                                vaddr,  paddr, l1_v_set_idx, l1_p_set_idx, l1_cache_set[l1_p_set_idx], l2_v_set_idx, l2_p_set_idx, l2_cache_set[l2_p_set_idx],
                                                fg_color, progress_bar.c_str());

        vaddr +=access_stride;
    }
    munmap(data,8 * (1 << 21));
    // free(data);
}

int main() {
    test_read_prefetch();
    return 0;
}