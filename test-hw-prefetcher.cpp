/*

 https://abertschi.ch/blog/2022/prefetching/
 https://community.intel.com/t5/Software-Tuning-Performance/How-does-the-next-PAGE-hardware-prefetcher-NPP-work/m-p/1588052
 https://www.felixcloutier.com/x86/prefetchh



Some prefecher words from "Intel® 64 and IA-32 Architectures Optimization Reference Manual:"
L1 data cache hw prefetcher.

Software can gain from the first-level data cache prefetchers in two cases:
• If data is not in the second-level cache, the first-level data cache prefetcher enables early trigger of the secondlevel cache prefetcher.
• If data is in the second-level cache and not in the first-level data cache, then the first-level data cache prefetcher
triggers earlier data bring-up of sequential cache line to the first-level data cache

The Intel Core microarchitecture(not xeon) contains two second-level cache prefetchers:
• Streamer — Loads data or instructions from memory to the second-level cache. To use the streamer, organize
the data or instructions in blocks of 128 bytes, aligned on 128 bytes. The first access to one of the two cache lines
in this block while it is in memory triggers the streamer to prefetch the pair line. To software, the L2 streamer’s
functionality is similar to the adjacent cache line prefetch mechanism found in processors based on Intel NetBurst
microarchitecture.
• Data prefetch logic (DPL) — DPL and L2 Streamer are triggered only by writeback memory type. They
prefetch only inside page boundary (4 KBytes). Both L2 prefetchers can be triggered by software prefetch
instructions and by prefetch request from DCU prefetchers. DPL can also be triggered by read for ownership (RFO)
operations. The L2 Streamer can also be triggered by DPL requests for L2 cache misses

Unused data prefech(both hw prefecher or sw prefech) would cause cache polution. prefetchnta command can be used to minimize this kind of 
cache pollustion caused by hw prefecher or sw prefech

Summary:

1. When using prefetchnta() hint to  prefetch pattern data, the pattern data would be prefeched into a place close to processor(L1),  minimizing cache pollution.
   prefetchnta() command  can minimize the cache pollution when hwprefetch is enabled. once the cache line is prefetched via software prefetchnta(). The cache line is non-temproal, 
    prefetcher would by pass L2 prefecher.  The subsequent cache line forward/backward access pattern would not be prefetched.

2. confirm the prefecher can't across linux page. Linux page is 4K address aligned. When access pattern(not using prefetchnta) is at the end page, hw prefetcher would prefetcher backward within the page, not across the page.

3. When we want to prefetch data in accurate domain(such as reading/writing tensor in parallel multiple cores, we don't one core to prefetch data which should be access by other cores),
    based on prefetchnta(), we can use "prefetchnta()" to accurately prefetch data into L1 in the core access boundary.

4. HW prefetching automically would be tiggered by cache miss on L1 or L2.  After using NTA hint to prefech the cache line A, 
    the following reading/writing cache A would be a cache hit even cacheline A hasn't been installed into L1. So the reading/writing on still prefetching cache line would not trigger hw prefech automitically..
   That is why using NTA hint prefetch before access A would not trigger prefeching  


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


class MemAccessPattern : public jit_generator {
 public:
  std::vector<int> m_cache_lines;
  bool m_is_write;
  bool m_use_prefetch_nta;
  MemAccessPattern(const std::vector<int>& cache_lines, bool is_write, bool use_prefetch_nta) : m_cache_lines(cache_lines), m_is_write(is_write), m_use_prefetch_nta(use_prefetch_nta) {
    create_kernel("MemAccessPattern");
  }
  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;
  void generate() {
    mfence();
    if (!m_is_write) {
        vpxorq(zmm0, zmm0, zmm0);
        for(auto& i : m_cache_lines) {
            if (m_use_prefetch_nta) {
                // SW prefetch also triggers HW prefetch
                //prefetcht2(ptr[reg_addr + i*64]); 
                prefetchnta(ptr[reg_addr + i*64]);
                // prefetcht0(ptr[reg_addr + i*64]);
            } else {
                vmovntdqa(zmm1, ptr[reg_addr + i*64]);
                // vmovups(zmm1, ptr[reg_addr + i*64]);
                vaddps(zmm0, zmm0, zmm1);
            }
        }
    } else {
        // write
        vpxorq(zmm0, zmm0, zmm0);
        for(auto& i : m_cache_lines) {
            if (m_use_prefetch_nta) {
                prefetchnta(ptr[reg_addr + i*64]);
            } else {
                vmovups(ptr[reg_addr + i*64], zmm0);
            }
        }
    }
    mfence();
    ret();
  }
};

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

void test_prefetch_nta() {

    EnvVar WRMEM("WRMEM", 0);
    EnvVar CLSTRIDE("CLSTRIDE", 1);

    std::vector<int> read_pattern;


    MeasureAccess measure_access;

    MSRConfig _msr1;

    std::vector<uint8_t> big_buffer(1024*1024*128, 0);

    uint64_t nbytes = 8192;
    void* p;
    posix_memalign(&p, 64, nbytes);  //posix_memalign(&p, huge_page_size, nbytes); seems can also allocate physical contineous pages sometimes.??
    auto* data = reinterpret_cast<uint8_t*>(p);
    for(int i = 0; i < 5; i++) {
        read_pattern.push_back(i*(int)CLSTRIDE);
    }
    //std::vector<int> read_pattern = {0, 4, 4*2};    // +3
    //std::vector<int> read_pattern = {0, 4};       // +2~3

    MemAccessPattern mem_prefetch_nta(read_pattern, WRMEM, true);
    MemAccessPattern mem_reads_writes(read_pattern, WRMEM, false);

    // auto* data = reinterpret_cast<uint8_t*>(mmap(NULL, 8 * 1 << 21, PROT_READ | PROT_WRITE,
    //              MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
    //              -1, 0));
    for(int i = 0; i < nbytes; i++) data[i] = 1;

    auto wait_tsc = second2tsc(1e-5);
    auto wait = [&](){
        auto t0 = __rdtsc();
        while(__rdtsc() - t0 < wait_tsc);
    };

    // 8192 = 128 x 64 = 128 cacheline
    std::vector<int64_t> access_times(nbytes/64, 0);

    const int repeats = 5;

    for(int r = 0; r < repeats; r ++) {
        for(int cache_line = 0; cache_line < access_times.size(); cache_line++) {

            {
                for (int i = 0; i < nbytes; i += 64) {
                    _mm_clflush(data + i);
                    _mm_mfence();
                }

                // mflush prevent HW prefetcher to work for some reason, need memset
                memset(&big_buffer[0], cache_line, big_buffer.size());
                load_prefetch_L2(&big_buffer[0], big_buffer.size());
                _mm_mfence();
            }

            // prefetch NTA can bypass L2 cache, trigger L1 prefech to fetch data based on pattern.
            mem_prefetch_nta(data);

            // even data has not be prefetched to L1. But reading/writing data would be a cache hit. So data followed by pattern would be prefetched.  No cache miss to trigger automatic hardware prefetch.
            mem_reads_writes(data);
            wait();

            // check which elements have been prefetched
            access_times[cache_line] += measure_access(data + cache_line*64);
        }
    }
    ::free(data);
    // munmap(data, 8 * 1 << 21);

    // show access_times
    for(int cache_line = 0; cache_line < access_times.size(); cache_line++) {
        // https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
        const char * fg_color = "90";
        if (std::find(read_pattern.begin(), read_pattern.end(), cache_line) != read_pattern.end()) {
            fg_color = "32";
        }

        auto nbars = static_cast<int>(tsc2second(access_times[cache_line]/repeats)*1e9 * 100/256);
        std::string progress_bar;
        // https://github.com/Changaco/unicode-progress-bars/blob/master/generator.html
        for(int i = 0; i < nbars; i++) progress_bar += "▅";// "█";
        printf(" cache_line[%3d] : %6.2f ns : \033[1;%sm %s \033[0m\n", cache_line, tsc2second(access_times[cache_line]/repeats)*1e9, fg_color, progress_bar.c_str());
    }
}

void test_prefetch_within_page() {

    EnvVar WRMEM("WRMEM", 0);
    EnvVar CLSTRIDE("CLSTRIDE", 1);

    std::vector<int> read_pattern;

    MeasureAccess measure_access;

    std::vector<uint8_t> big_buffer(1024*1024*128, 0);
    // 8192 = 128 x 64 = 128 cacheline
    uint64_t nbytes = 8192;
    void* p;
    posix_memalign(&p, 64, nbytes);  //posix_memalign(&p, huge_page_size, nbytes); seems can also allocate physical contineous pages sometimes.??
    auto* data = reinterpret_cast<uint8_t*>(p);
    // auto* data = reinterpret_cast<uint8_t*>(mmap(NULL, 8 * 1 << 21, PROT_READ | PROT_WRITE,
//              MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
//              -1, 0));
    uintptr_t vaddr, paddr = 0;
    vaddr = (uintptr_t)data;
    if (virt_to_phys_user(&paddr, getpid(), vaddr)) {
        fprintf(stderr, "error: virt_to_phys_user\n");
        return;
    };
    size_t page_offset = (size_t)(paddr) % 4096;
    size_t cl_to_boundary  = 64 - page_offset/64;
    size_t probe_number = 5;

    for (size_t i = 0; i < std::min(cl_to_boundary, probe_number); i++)
        read_pattern.push_back(cl_to_boundary - i - 1);

    MemAccessPattern mem_prefetch_nta(read_pattern, WRMEM, true);
    MemAccessPattern mem_reads_writes(read_pattern, WRMEM, false);


    for(int i = 0; i < nbytes; i++) data[i] = 1;

    auto wait_tsc = second2tsc(1e-5);
    auto wait = [&](){
        auto t0 = __rdtsc();
        while(__rdtsc() - t0 < wait_tsc);
    };

    std::vector<int64_t> access_times(nbytes/64, 0);

    const int repeats = 5;

    for(int r = 0; r < repeats; r ++) {
        for(int cache_line = 0; cache_line < access_times.size(); cache_line++) {
            {
                for (int i = 0; i < nbytes; i += 64) {
                    _mm_clflush(data + i);
                    _mm_mfence();
                }

                // mflush prevent HW prefetcher to work for some reason, need memset
                memset(&big_buffer[0], cache_line, big_buffer.size());
                load_prefetch_L2(&big_buffer[0], big_buffer.size());
                _mm_mfence();
            }

            /* here can't use NTA((either T0, T1 hint is okay)) hint to prefech, After NTA hint is applied, prefetcher would not prefetch based onr ead_pattern*/
            // mem_prefetch_nta(data);
            // wait();

            mem_reads_writes(data);
            wait();

            // check which elements have been prefetched
            access_times[cache_line] += measure_access(data + cache_line*64);
        }
    }
    ::free(data);
    // munmap(data, 8 * 1 << 21);

    // show access_times
    for(int cache_line = 0; cache_line < access_times.size(); cache_line++) {
        // https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
        const char * fg_color = "90";
        if (std::find(read_pattern.begin(), read_pattern.end(), cache_line) != read_pattern.end()) {
            fg_color = "32";
        }

        auto nbars = static_cast<int>(tsc2second(access_times[cache_line]/repeats)*1e9 * 100/256);
        std::string progress_bar;
        // https://github.com/Changaco/unicode-progress-bars/blob/master/generator.html
        for(int i = 0; i < nbars; i++) progress_bar += "▅";// "█";
        printf(" cache_line[%3d] : %6.2f ns : \033[1;%sm %s \033[0m\n", cache_line, tsc2second(access_times[cache_line]/repeats)*1e9, fg_color, progress_bar.c_str());
    }
}


int main() {
    printf("---------------------------------------test_prefetch_within_page---------------\n");
    // test: hw prefetcher pattern and hw prefetch would not cross page boundary.
    test_prefetch_within_page();
    printf("---------------------------------------test_prefetch_nta---------------\n");
    // test: sw prefetcher with NTA hint before reading/writing read-pattern would not trigger cache miss. The subsequesnt cache line after the pattern would not prefetched automatically.
    test_prefetch_nta();
    return 0;
}