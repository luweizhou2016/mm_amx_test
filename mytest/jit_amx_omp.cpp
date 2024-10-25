// The test is used to use OMP to test AMX kernel.
// AMX kernel(2x2) would first accumulate all the K dimensions. m/32 * n /32 would be split via cores.
//  OMP_NUM_THREADS=8 ./a.out
// ENV: USE_NUMA = 0
// ENV: CLFLUSH = ""
// omp_get_num_threads() = 8
// initXTILE success!
// ===============================BF16========================
// [all pass] with max rtol,atol=0,0
// rdtsc is calibrating ... done.
// amx_jit_(M=_1024_,N=_1024_,K=_128_)_[PASS]      : 0.16 us x 2557015, HW_CYCLES=517 CPU~3.19GHz 519217.52(Ops/cycle) 1655.937(TOps/s)
// [all pass] with max rtol,atol=0.00,0.00
// amx_jit_(M=_4096_,N=_4096_,K=_128_)_[PASS]      : 0.16 us x 348593, HW_CYCLES=514 CPU~3.19GHz

#include "jit.hpp"
#include <vector>
#include <omp.h>

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif


class Linear32x32_AMX : public jit_generator {
public:
    int m_K;
    TileConfig m_tile_cfg;
    Linear32x32_AMX(int K) : m_K(K) {
        create_kernel("Linear32x32_AMX");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // A0:4
                             {16, 64}, // A1:5
                             {16, 64}, // B0:6
                             {16, 64}, // B1:7
                         });
    }

    const TileConfig& tile_config() { return m_tile_cfg; }

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_A1_addr = r11;
    Xbyak::Reg64 reg_ktiles = r9;

    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC10 = tmm1;
    Xbyak::Tmm tmmC01 = tmm2;
    Xbyak::Tmm tmmC11 = tmm3;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmA1 = tmm5;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;


    void generate() {
        Xbyak::Label loop_k_tile;
        assert(m_K % 32 == 0);
        tilezero(tmmC00);
        tilezero(tmmC01);
        tilezero(tmmC10);
        tilezero(tmmC11);

        mov(reg_B_stride, 64);
        mov(reg_ktiles, m_K / 32);
        const size_t step_a = 64;
        const size_t step_b = 2048;
        const size_t step_c = 64;
        align(64, false);
        bool do_sw_prefetch = std::getenv("SWPF") != nullptr;

        L(loop_k_tile);
        if (do_sw_prefetch) {
            for (auto cl = 0; cl < 16; cl++) {
                auto cl_offset = 64 * cl;
                prefetcht0(ptr[reg_A_addr + step_a + cl_offset]);
            }
        }
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        tdpbf16ps(tmmC00, tmmA0, tmmB0);

        if (do_sw_prefetch) {
            for (auto cl = 0; cl < 16; cl++) {
                auto cl_offset = 64 * cl;
                prefetcht0(ptr[reg_B_addr + step_b + cl_offset]);
            }
        }

        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);
        if (do_sw_prefetch) {
            for (auto cl = 0; cl < 16; cl++) {
                auto cl_offset = 64 * cl;
                prefetcht0(ptr[reg_A1_addr + step_a + cl_offset]);
            }
        }
        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        if (do_sw_prefetch) {
            for (auto cl = 0; cl < 16; cl++) {
                auto cl_offset = 64 * cl;
                 prefetcht0(ptr[reg_B_addr + 1024 +  step_b + cl_offset]);
            }
        }
        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride + 1024]);
        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        lea(reg_A_addr, ptr[reg_A_addr + step_a]);
        lea(reg_B_addr, ptr[reg_B_addr + step_b]);

        dec(reg_ktiles);
        jnz(loop_k_tile, T_NEAR);
    
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + step_c], tmmC01);
        // lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 16]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);

        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
        tilestored(ptr[reg_C_addr + reg_C_stride + step_c], tmmC11);
        ret();
    }
};


#include "kernels_amx.hpp"
// #include "kernels_avx512.hpp"
#include "tensor2D.hpp"
#include "timeit.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>

timeit timer({
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});

template <typename LinearAMX>
int amx_jit(const int M, const int N, const int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();
    tensor2D<ov::bfloat16> BPacked(K * N, 1, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearAMX mm_jit(K);
    // TileConfigScope tcfg(mm_jit.tile_config());
    #pragma omp parallel
    {
        TileConfiger tile_config;
        auto ptr_conf = mm_jit.tile_config();
        tile_config(&ptr_conf);
    }
    // the memory layout would be changed to NK2n16k16n2k
    for (int n = 0, i = 0; n < N; n += 32) {
        for (int k = 0; k < K; k += 32) {
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);
            i++;
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n + 16, k), Bt.stride);
            i++;
        }
    }
    C0 = 0;
    matmul(A, B, C0);
    #pragma omp parallel for collapse(2)       //split M/32*N/32 acrosss cores.
    //#pragma omp parallel for                 //split M/32 across cores.
    for (int j = 0; j < M/32; j++) {
        for (int i = 0; i < N/32; i++) {
            mm_jit(&A(j*32,0), A.stride, &BPacked[i*32*K], &C1(j*32, i*32), C1.stride);
        }
    }

    std::string acc;
    std::string acc_color;
    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        if (std::getenv("SHOW_ERR")) {
            std::cout << "============= A ================ " << std::endl;
            std::cout << A << std::endl;
            std::cout << "============= B ================ " << std::endl;
            std::cout << B << std::endl;
            logger() << C0 << std::endl;
            logger() << C1 << std::endl;
        }
        acc = "[FAIL]";
        acc_color = "1;31";
    }

    timer.tag(__func__, "(M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mm_jit(&A[0], A.stride, &BPacked[0], &C1[0], C1.stride); },
            M * N * K * 2 // OPS per call
        );

    return 0;
}

int main(int argc, const char* argv[]) {
    srand(0);
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
        }
    }
    bool initAMX = initXTILE();
    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << "===============================BF16========================\n";
    amx_jit<Linear32x32_AMX>(1024, 1024, 128);
    amx_jit<Linear32x32_AMX>(4096, 4096, 128);

}