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


#define _XOPEN_SOURCE 700
#include <fcntl.h> /* open */
#include <stdint.h> /* uint64_t  */
#include <stdio.h> /* printf */
#include <stdlib.h> /* size_t */
#include <unistd.h> /* pread, sysconf */
#include <string.h>

typedef struct {
    uint64_t pfn : 55;
    unsigned int soft_dirty : 1;
    unsigned int file_page : 1;
    unsigned int swapped : 1;
    unsigned int present : 1;
} PagemapEntry;

/* Parse the pagemap entry for the given virtual address.
 *
 * @param[out] entry      the parsed entry
 * @param[in]  pagemap_fd file descriptor to an open /proc/pid/pagemap file
 * @param[in]  vaddr      virtual address to get entry for
 * @return 0 for success, 1 for failure
 */
int pagemap_get_entry(PagemapEntry *entry, int pagemap_fd, uintptr_t vaddr)
{
    size_t nread;
    ssize_t ret;
    uint64_t data;
    uintptr_t vpn;

    vpn = vaddr / sysconf(_SC_PAGE_SIZE);
    nread = 0;
    while (nread < sizeof(data)) {
        ret = pread(pagemap_fd, ((uint8_t*)&data) + nread, sizeof(data) - nread,
                vpn * sizeof(data) + nread);
        nread += ret;
        if (ret <= 0) {
            return 1;
        }
    }
    entry->pfn = data & (((uint64_t)1 << 55) - 1);
    entry->soft_dirty = (data >> 55) & 1;
    entry->file_page = (data >> 61) & 1;
    entry->swapped = (data >> 62) & 1;
    entry->present = (data >> 63) & 1;
    return 0;
}

/* Convert the given virtual address to physical using /proc/PID/pagemap.
 *
 * @param[out] paddr physical address
 * @param[in]  pid   process to convert for
 * @param[in] vaddr virtual address to get entry for
 * @return 0 for success, 1 for failure
 */
int virt_to_phys_user(uintptr_t *paddr, pid_t pid, uintptr_t vaddr)
{
    char pagemap_file[BUFSIZ];
    int pagemap_fd;

    snprintf(pagemap_file, sizeof(pagemap_file), "/proc/%ju/pagemap", (uintmax_t)pid);
    pagemap_fd = open(pagemap_file, O_RDONLY);
    if (pagemap_fd < 0) {
        return 1;
    }
    PagemapEntry entry;
    if (pagemap_get_entry(&entry, pagemap_fd, vaddr)) {
        return 1;
    }
    close(pagemap_fd);
    *paddr = (entry.pfn * sysconf(_SC_PAGE_SIZE)) + (vaddr % sysconf(_SC_PAGE_SIZE));
    return 0;
}

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
                //prefetchw(ptr[reg_addr + i*64]);
                prefetcht2(ptr[reg_addr + i*64]); 
                // prefetchnta(ptr[reg_addr + i*64]);
                // prefetchwt1(ptr[reg_addr + i*64]);
            } else {
                //prefetchnta(ptr[reg_addr + i*64]);
                vmovups(zmm1, ptr[reg_addr + i*64]);
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

void test_read_prefetch() {

    EnvVar WRMEM("WRMEM", 0);
    EnvVar CLSTRIDE("CLSTRIDE", 1);
    pid_t pid = getpid();

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

    uint64_t nbytes = cache_line_sz * l2_sets * 1;
    std::vector<int> read_pattern(nbytes / access_stride, 0);
    for (int i = 0; i < read_pattern.size(); i ++)
        read_pattern[i] = access_stride/64 * i;
    // MemAccessPattern mem_reads_writes(read_pattern, WRMEM, false);
    void* p;
    // printf("****pid is %d\n", pid);
    // getchar();
    // posix_memalign(&p, huge_page_size, nbytes);
    // auto* data = reinterpret_cast<uint8_t*>(p);
    // auto* data = reinterpret_cast<uint8_t*>(aligned_alloc(huge_page_size, nbytes));
    // auto* data = reinterpret_cast<uint8_t*>(malloc(nbytes + 4*1024*1024));

    auto* data = reinterpret_cast<uint8_t*>(mmap(NULL, 8 * (1 << 21), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                 -1, 0));
    
    for(int i = 0; i < 8 * (1 << 21); i++) data[i] = i;
    std::vector<int64_t> access_times(nbytes/64, 0);

    const int repeats = 3;

    int64_t sum = 0;

    _mm_mfence();
    for(int r = 0; r < repeats; r ++) {

        for (int i = 0; i < nbytes; i += 64) {
            _mm_clflush(data + i);
        }
        _mm_mfence();
        size_t sum = 0;
        for(int cache_line = 0; cache_line < access_times.size();  cache_line += access_stride/64) {
            // check which elements have been prefetched
            sum += measure_access(data + cache_line*64);
        }
        _mm_mfence();
        // mem_reads_writes(data);
        for(int cache_line = 0; cache_line < access_times.size();  cache_line += access_stride/64) {
            access_times[cache_line] += measure_access(data + cache_line*64);

        }
    }

    {
        uintptr_t vaddr, paddr = 0;
        vaddr = (uintptr_t)data;
        for(int cache_line = 0; cache_line < access_times.size(); cache_line += access_stride/64) {
            if (virt_to_phys_user(&paddr, pid, vaddr)) {
                fprintf(stderr, "error: virt_to_phys_user\n");
            };
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
        if (std::find(read_pattern.begin(), read_pattern.end(), cache_line) != read_pattern.end()) {
            fg_color = "32";
        }
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
}

int main() {
    test_read_prefetch();
    return 0;
}