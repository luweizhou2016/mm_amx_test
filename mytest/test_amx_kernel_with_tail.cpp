/*
This test supports: 2x2 tile AMX matmul case.
        M                N                   K
 16 < M  <=32          16 < N <=32        16 < K <=32 
    
This test Mtail, Ntail and K tial case.
For the weight part [K,N] would be padding to [K_Padding, N_Padding]
K_Padding is 64 byte aigned when transpose, which means K dimension on BF16 tensor would be 32 element aligned.
N_Padding is 32 byte aligned when transpose, which means N dimension on BF16 tensor would be 16 element aligned.

[K,N] would be reordered to [KN16k16n2k] blocking layout via transpose_epi32_16xN(), transpose_epi32_16x16().
The layout fits the AMX tile config for Matrix B. The tile config would ensure only valid data is loaded into tile, the
padding data would not be loaded into tile. For example, [18, 28] * [28, 21] = [18, 21]

for the Matrix B [28, 21], the K_tail = 28, N_Tail = [5], [28, 8] would be padding and reorder to [16k,16n,2k]
 16k: only first 14 is valid
 16n: only first 5 is valid.

 The tile config for matrix would be k =28, n = 5, ensure only valid data is loaded into tile.
 */

#include "jit.hpp"
#include <vector>

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

class Linear32x32_AMX_TAIL_2x2 : public jit_generator {
public:
    int m_M, m_K, m_N;
    int M1, M2;

    TileConfig m_tile_cfg;
    Linear32x32_AMX_TAIL_2x2(int M, int K, int N) : m_M(M), m_K(K), m_N(N) {
        assert(K %2 == 0 && K <= 32);
        assert(N > 16 && N <= 32);
        assert(M > 16 && M <= 32);
        int tail_N = m_N % 16;
        M2 =  m_M / 2;
        M1 = m_M % 2 ? (M2 + 1) : M2;
        create_kernel("Linear32x32_AMX_TAIL_2x2");
        m_tile_cfg.reset(1, 0,
                         {
                             {M1, 64}, // C:0
                             {M2, 64}, // C:1
                             {M1, tail_N * 4}, // C:2
                             {M2, tail_N * 4}, // C:3
                             {M1, m_K*2}, // A0:4
                             {M2, m_K*2}, // A1:5
                             {m_K/2, 64}, // B0:6
                             {m_K/2, tail_N * 4}, // B1:7
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
    Xbyak::Reg64 reg_tmp = r9;

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
        assert(m_K % 2 == 0);
        tilezero(tmmC00);
        tilezero(tmmC01);
        tilezero(tmmC10);
        tilezero(tmmC11);
        mov(reg_B_stride, 64);
        const size_t step_a = 64;
        const size_t step_b = 2048;
        const size_t step_c = 64;
        bool do_sw_prefetch = std::getenv("SWPF") != nullptr;   
        align(64, false);
        L(loop_k_tile);
        // if (do_sw_prefetch) {
        //     for (auto cl = 0; cl < 16; cl++) {
        //         auto cl_offset = 64 * cl;
        //         prefetcht0(ptr[reg_A_addr + step_a + cl_offset]);
        //     }
        // }
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        tdpbf16ps(tmmC00, tmmA0, tmmB0);

        // if (do_sw_prefetch) {
        //     for (auto cl = 0; cl < 16; cl++) {
        //         auto cl_offset = 64 * cl;
        //         prefetcht0(ptr[reg_B_addr + step_b + cl_offset]);
        //     }
        // }
        mov(reg_tmp, reg_A_stride);
        imul(reg_tmp, reg_tmp, M1);
        mov(reg_A1_addr, reg_A_addr);
        add(reg_A1_addr,  reg_tmp);
        // if (do_sw_prefetch) {
        //     for (auto cl = 0; cl < 16; cl++) {
        //         auto cl_offset = 64 * cl;
        //         prefetcht0(ptr[reg_A1_addr + step_a + cl_offset]);
        //     }
        // }
        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        // if (do_sw_prefetch) {
        //     for (auto cl = 0; cl < 16; cl++) {
        //         auto cl_offset = 64 * cl;
        //          prefetcht0(ptr[reg_B_addr + 1024 +  step_b + cl_offset]);
        //     }
        // }
        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride + 1024]);
        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        tdpbf16ps(tmmC11, tmmA1, tmmB1);

        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + step_c], tmmC01);

        mov(reg_tmp, reg_C_stride);
        imul(reg_tmp, reg_tmp, M1);
        add(reg_C_addr,  reg_tmp);

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
int amx_jit_tail_2x2(const int M, const int N, const int K, int times = -1000) {
    assert(N >16 && N < 32 && K <= 32 && K%2 == 0);
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    // Default transpose would contruct new tensor with K alignment with 64 byte.
    auto Bt = B.Tr_Align_Dim0();
    //Padding the Ktail to make K aligh with 32 bf16 elements(64 bytes),
    int K_padding = ((K + 31) / 32 ) * 32;
    int N_padding = ((N + 15) / 16 ) * 16;
    tensor2D<ov::bfloat16> BPacked(K_padding * N_padding, 1, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearAMX mm_jit(M, K, N);
    TileConfigScope tcfg(mm_jit.tile_config());

    // the memory layout would be reorder to KN16k16n2k
    // The K Padding value would be also reordered. But would not be loaded into tile because of tile config.
    // when having K padding, both input and output of reordering would be 64 byter
    for (int k = 0, i = 0; k < K_padding; k += 32) {
        int n;
        for (n = 0; n < N; n += 16) {
            if (n == (N_padding - 16 ) && N_padding != N)
                amx_kernel::functional::transpose_epi32_16xN(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride, 64);  //Each tile block is 16k16n2k
            else
                amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);  //Each tile block is 16k16n2k
            i++;
        }
    }

    C0 = 0;
    matmul(A, B, C0);

    std::string acc;
    std::string acc_color;
    mm_jit(&A[0], A.stride, &BPacked[0], &C1[0], C1.stride);

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

int amx_mm(const int M, const int N, int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();
    std::vector<ov::bfloat16> BPacked(K * N, 0);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    amx_kernel::Matmul<ov::bfloat16, ov::bfloat16> mm32x32(true, true);
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(C1);

    std::string acc;
    std::string acc_color;
    C0 = 0;
    matmul(A, B, C0);

    mm32x32(A, Bt, 0, N, pp);
    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        acc_color = "1;31";
        acc = "[FAIL]";
    }

    timer.tag(__func__, " (M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mm32x32(A, Bt, 0, N, pp); },
            M * N * K * 2 // OPS per call
        );

    return 0;
}

int main(int argc, const char* argv[]) {
    srand(0);
    bool initAMX = initXTILE();

    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();

    std::cout << "===============================BF16========================\n";
    // amx_jit_tail_2x2<Linear32x32_AMX_TAIL_2x2>(21, 20, 30);
    for (auto m = 17; m < 31; m++)
        for (auto n = 17; n <=31; n++)
            for (auto k=2; k < 32; k+=2) {
            amx_jit_tail_2x2<Linear32x32_AMX_TAIL_2x2>(m, n, k);
    }
}
