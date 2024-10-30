

#include "jit.hpp"
#include <vector>
#include <omp.h>

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif


class Linear32x32_AMX_N: public jit_generator {
public:
    int m_N;
    int m_K;
    TileConfig m_tile_cfg;
    Linear32x32_AMX_N(int N, int K = 32) : m_N(N), m_K(K) {
        create_kernel("Linear32x32_AMX_N");
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
    Xbyak::Reg64 reg_A_base = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_base = abi_param3;
    Xbyak::Reg64 reg_C_base = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_C_tmp_addr = r11;
    Xbyak::Reg64 reg_ntiles = r9;
    Xbyak::Reg64 reg_ktiles = r12;

    Xbyak::Reg64 reg_A_addr = r13;
    Xbyak::Reg64 reg_B_addr = r14;
    Xbyak::Reg64 reg_C_addr = r15;


    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC10 = tmm1;
    Xbyak::Tmm tmmC01 = tmm2;
    Xbyak::Tmm tmmC11 = tmm3;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmA1 = tmm5;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;


    void generate() {

        Xbyak::Label loop_n_tile;
        Xbyak::Label loop_k_tile;

        assert(m_N % 32 == 0);
        bool do_sw_prefetch = std::getenv("SWPF") != nullptr;
        const size_t N_step_b = 2048;
        const size_t N_step_c = 64;
        const size_t K_step_a = 64;
        // const size_t K_step_c = 128;
    
        preamble();
        mov(reg_B_stride, 64);
        mov(reg_ktiles, m_K / 32);
        mov(reg_B_addr, reg_B_base);

        align(64, false);
        L(loop_k_tile);
        mov(reg_ntiles, m_N / 32);
        mov(reg_A_addr, reg_A_base);
        mov(reg_C_addr, reg_C_base);

        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);

        lea(reg_A_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        tileloadd(tmmA1, ptr[reg_A_addr + reg_A_stride]);
            L(loop_n_tile);
            mov(reg_C_tmp_addr, reg_C_addr);
            // For the first accumulation on K can just tilezero C, no need to tileload
            // Unify to use tileload for easiness.
            tileloadd(tmmC00, ptr[reg_C_tmp_addr + reg_C_stride]);
            tileloadd(tmmC01, ptr[reg_C_tmp_addr + reg_C_stride + N_step_c]);
            lea(reg_C_tmp_addr, ptr[reg_C_tmp_addr + reg_C_stride * 8]);
            lea(reg_C_tmp_addr, ptr[reg_C_tmp_addr + reg_C_stride * 8]);
            tileloadd(tmmC10, ptr[reg_C_tmp_addr + reg_C_stride]);
            tileloadd(tmmC11, ptr[reg_C_tmp_addr + reg_C_stride + N_step_c]);

            mov(reg_C_tmp_addr, reg_C_addr);
            tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
            tdpbf16ps(tmmC00, tmmA0, tmmB0);

            // if (do_sw_prefetch) {
            //     for (auto cl = 0; cl < 16; cl++) {
            //         auto cl_offset = 64 * cl;
            //         prefetcht0(ptr[reg_B_addr + N_step_b + cl_offset]);
            //     }
            // }

            tdpbf16ps(tmmC10, tmmA1, tmmB0);

            // if (do_sw_prefetch) {
            //     for (auto cl = 0; cl < 16; cl++) {
            //         auto cl_offset = 64 * cl;
            //          prefetcht0(ptr[reg_B_addr + 1024 +  N_step_b + cl_offset]);
            //     }
            // }
            tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride + 1024]);
            tdpbf16ps(tmmC01, tmmA0, tmmB1);
            tdpbf16ps(tmmC11, tmmA1, tmmB1);

            //Update Matrix C[32x32]
            tilestored(ptr[reg_C_tmp_addr + reg_C_stride], tmmC00);
            tilestored(ptr[reg_C_tmp_addr + reg_C_stride + N_step_c], tmmC01);
            lea(reg_C_tmp_addr, ptr[reg_C_tmp_addr + reg_C_stride * 8]);
            lea(reg_C_tmp_addr, ptr[reg_C_tmp_addr + reg_C_stride * 8]);
            tilestored(ptr[reg_C_tmp_addr + reg_C_stride], tmmC10);
            tilestored(ptr[reg_C_tmp_addr + reg_C_stride + N_step_c], tmmC11);
            //udpate C address 
            lea(reg_C_addr, ptr[reg_C_addr + N_step_c * 2]);
            //update B address
            lea(reg_B_addr, ptr[reg_B_addr + N_step_b]);

            dec(reg_ntiles);
            jnz(loop_n_tile, T_NEAR);
    
        lea(reg_A_base, ptr[reg_A_base + K_step_a]);

        dec(reg_ktiles);
        jnz(loop_k_tile, T_NEAR);
            
        postamble();
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
    int k_block = 128;
    int n_block = N;
    assert(K%k_block == 0);
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();
    tensor2D<ov::bfloat16> BPacked(K * N, 1, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearAMX mm_jit(n_block, k_block);
    // TileConfigScope tcfg(mm_jit.tile_config());
    #pragma omp parallel
    {
        TileConfiger tile_config;
        auto ptr_conf = mm_jit.tile_config();
        tile_config(&ptr_conf);
    }
    // the memory layout would be changed to KN2n16k16n2k
    for (int k = 0,i = 0; k < K; k += 32) {
            for (int n = 0; n < N; n += 16) {
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);
            i++;
        }
    }

    C0 = 0;
    C1 = 0;
    matmul(A, B, C0);
    #pragma omp parallel for                 //split M/32 across cores. would not split K across core . So can't use "omp parallel for collapse(2)"
    for (int m = 0; m < M; m+=32) {
        for (int k = 0; k < K; k+=k_block) {
            mm_jit(&A(m,k), A.stride, &BPacked[k*N], &C1(m, 0), C1.stride);
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
    amx_jit<Linear32x32_AMX_N>(128, 512, 256);
    // amx_jit<Linear32x32_AMX>(4096, 4096, 128);

}