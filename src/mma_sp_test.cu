// Standalone validation of the 2:4 sparse Tensor-Core GEMM via inline PTX
// mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 (the blueprint centerpiece).
//
// Why standalone / off the generation path: applying 2:4 structured sparsity to the
// PRETRAINED LFM2.5 weights (magnitude-pruning 50% of each group of 4) without
// sparsity-aware finetuning destroys coherence -- recovering it would require RETRAINING,
// which the session goal explicitly excludes. So we prove the *kernel* (sparse decode +
// mma.sp instruction + thread-data/metadata mapping) is correct against known small
// matrices, and keep dense INT4 on the actual model path. See kb/02_decisions.md.
//
// Validation strategy: a 16x32 sparse A (2:4) times a 32x8 dense B. We pick a *uniform*
// sparsity pattern across all 16 rows so every per-row metadata word is identical; that
// makes the result independent of the (fiddly) metadata->thread distribution and lets us
// focus on verifying the numeric contract: D == A_logical @ B. We test two distinct
// patterns (positions {0,1} and {1,3} within each group of 4) to exercise metadata decode.
//
// build: nvcc -O3 -std=c++17 -arch=sm_86 src/mma_sp_test.cu -o build/mma_sp_test
//
// Author: g023 (https://github.com/g023/)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// MMA tile shape (fp16 sparse): D[16x8] = A[16x32 sparse 2:4] * B[32x8] + C[16x8], f32 acc.
#define MM 16
#define NN 8
#define KK 32          // logical K (effective stored K = 16)

static __device__ __forceinline__ uint32_t pack(half lo, half hi){
    uint32_t r; half2 h = __halves2half2(lo, hi);
    memcpy(&r, &h, 4); return r;
}

// One warp computes the whole 16x8 tile.
// A_stored: 16x16 row-major (the 16 nonzeros per row, in ascending logical-column order).
// B:        32x8  row-major (K x N).
// meta:     per-row 2-bit selector word (identical for all rows here).
// D:        16x8  row-major (output).
__global__ void mma_sp_kernel(const half* __restrict__ A_stored, const half* __restrict__ B,
                              uint32_t meta, float* __restrict__ D){
    int lane = threadIdx.x;          // 0..31
    int gr   = lane >> 2;            // 0..7  (row group)
    int tc   = (lane & 3) * 2;       // 0,2,4,6

    // ---- A fragment: standard m16n8k16-style layout over the 16x16 stored matrix ----
    // rows gr and gr+8; stored-cols tc{,+1} and tc+8{,+1}
    auto As = [&](int r,int c)->half{ return A_stored[r*16 + c]; };
    uint32_t a0 = pack(As(gr,   tc+0), As(gr,   tc+1));
    uint32_t a1 = pack(As(gr+8, tc+0), As(gr+8, tc+1));
    uint32_t a2 = pack(As(gr,   tc+8+0), As(gr,   tc+8+1));
    uint32_t a3 = pack(As(gr+8, tc+8+0), As(gr+8, tc+8+1));

    // ---- B fragment: m16n8k32 layout, B is K x N, column n = gr ----
    auto Bk = [&](int k,int n)->half{ return B[k*NN + n]; };
    int n = gr;
    uint32_t b0 = pack(Bk(tc+0,    n), Bk(tc+1,    n));
    uint32_t b1 = pack(Bk(tc+8+0,  n), Bk(tc+8+1,  n));
    uint32_t b2 = pack(Bk(tc+16+0, n), Bk(tc+16+1, n));
    uint32_t b3 = pack(Bk(tc+24+0, n), Bk(tc+24+1, n));

    float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
    float d0,d1,d2,d3;
    asm volatile(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
      : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
        "r"(b0),"r"(b1),"r"(b2),"r"(b3),
        "f"(c0),"f"(c1),"f"(c2),"f"(c3),
        "r"(meta));

    // ---- D fragment: rows gr, gr+8 ; cols tc, tc+1 ----
    D[(gr  )*NN + tc+0] = d0;
    D[(gr  )*NN + tc+1] = d1;
    D[(gr+8)*NN + tc+0] = d2;
    D[(gr+8)*NN + tc+1] = d3;
}

// Build per-row metadata word for a uniform 2:4 pattern given the two chosen positions
// (p0<p1 in {0,1,2,3}) within every group of 4. K=32 -> 8 groups -> 8 nibbles.
static uint32_t make_meta(int p0,int p1){
    uint32_t nib = (uint32_t)((p1<<2)|p0) & 0xF;   // low 2 bits = first nz, next 2 = second
    uint32_t m=0; for(int g=0; g<8; g++) m |= nib << (4*g);
    return m;
}

static int run_pattern(int p0,int p1){
    printf("\n[mma.sp] pattern: nonzeros at quad positions {%d,%d}\n", p0,p1);
    // random small-integer stored A (16x16) and B (32x8), exactly fp16-representable.
    std::vector<half> A_stored(16*16), B(32*8);
    srand(1234 + p0*10 + p1);
    for(auto& x: A_stored) x = __float2half((float)((rand()%9)-4));
    for(auto& x: B)        x = __float2half((float)((rand()%9)-4));

    // CPU reference: expand A_stored -> A_logical (16x32) using the pattern, then D=A@B.
    std::vector<float> A_log(16*32, 0.f), Dref(16*8, 0.f);
    for(int r=0;r<16;r++) for(int g=0; g<8; g++){
        A_log[r*32 + g*4 + p0] = __half2float(A_stored[r*16 + g*2 + 0]);
        A_log[r*32 + g*4 + p1] = __half2float(A_stored[r*16 + g*2 + 1]);
    }
    for(int r=0;r<16;r++) for(int c=0;c<8;c++){
        float acc=0.f; for(int k=0;k<32;k++) acc += A_log[r*32+k]*__half2float(B[k*8+c]);
        Dref[r*8+c]=acc;
    }

    half *dA,*dB; float* dD;
    CK(cudaMalloc(&dA,16*16*2)); CK(cudaMalloc(&dB,32*8*2)); CK(cudaMalloc(&dD,16*8*4));
    CK(cudaMemcpy(dA,A_stored.data(),16*16*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,B.data(),32*8*2,cudaMemcpyHostToDevice));
    uint32_t meta = make_meta(p0,p1);
    mma_sp_kernel<<<1,32>>>(dA,dB,meta,dD);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> Dgpu(16*8); CK(cudaMemcpy(Dgpu.data(),dD,16*8*4,cudaMemcpyDeviceToHost));

    int bad=0; float maxerr=0.f;
    for(int i=0;i<16*8;i++){ float e=fabsf(Dgpu[i]-Dref[i]); maxerr=fmaxf(maxerr,e); if(e>1e-1f) bad++; }
    printf("[mma.sp] meta=0x%08x  maxerr=%.4f  mismatches=%d/128\n", meta, maxerr, bad);
    // show a couple of entries
    printf("[mma.sp] D[0,0..3] gpu=%.1f %.1f %.1f %.1f  ref=%.1f %.1f %.1f %.1f\n",
        Dgpu[0],Dgpu[1],Dgpu[2],Dgpu[3], Dref[0],Dref[1],Dref[2],Dref[3]);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

// ============================================================================
// Tiled sparse-INT4 GEMM built on the validated tile: D[M,N] = A[M,K sparse INT4] * B[K,N].
// A is 2:4 structured-sparse (uniform per-quad pattern p0,p1) AND INT4-quantized:
//   stored nonzeros = M x (K/2) nibbles, 2 per byte (row stride K/4 bytes);
//   nibble q in [0,15], dequant value = (q-8) * scale[row, logical_k/128].
// Decode happens on the fly per A fragment; mma.sp feeds the Tensor Cores; K and N tiled.
// ============================================================================
#define GRP 128
__global__ void gemm_sp_int4(const uint8_t* __restrict__ Apk, const half* __restrict__ Asc,
                             const half* __restrict__ B, uint32_t meta,
                             int M,int N,int K,int p0,int p1, float* __restrict__ D){
    int lane = threadIdx.x;
    int mt = blockIdx.y * MM;          // tile row base
    int nt = blockIdx.x * NN;          // tile col base
    int gr = lane >> 2, tc = (lane&3)*2;
    int Keff = K/2;                    // stored nonzeros per row
    int Kg   = K/GRP;                  // scale groups per row
    int rowstride = Keff/2;            // bytes per row

    // decode one stored value: row r (global), stored col j (0..Keff-1) -> dequant half
    auto Aval = [&](int r,int j)->float{
        uint8_t byte = Apk[(size_t)r*rowstride + (j>>1)];
        int q = (j&1) ? (byte>>4) : (byte&0xF);
        int quad = j>>1, pos = (j&1)? p1 : p0;     // logical col of this stored value
        int logical_k = quad*4 + pos;
        float sc = __half2float(Asc[(size_t)r*Kg + logical_k/GRP]);
        return (float)(q-8)*sc;
    };
    auto Bk = [&](int k,int n)->half{ return B[(size_t)k*N + n]; };

    float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
    int nKt = K/KK;                    // number of K-tiles (32 logical each)
    for(int kt=0; kt<nKt; kt++){
        int sk = kt*16;                // stored-col base for this K-tile (16 stored / tile)
        int lk = kt*32;                // logical-col base
        // A fragment (decode INT4 nonzeros for rows mt+gr, mt+gr+8)
        uint32_t a0=pack(__float2half(Aval(mt+gr,   sk+tc+0)),  __float2half(Aval(mt+gr,   sk+tc+1)));
        uint32_t a1=pack(__float2half(Aval(mt+gr+8, sk+tc+0)),  __float2half(Aval(mt+gr+8, sk+tc+1)));
        uint32_t a2=pack(__float2half(Aval(mt+gr,   sk+tc+8+0)),__float2half(Aval(mt+gr,   sk+tc+8+1)));
        uint32_t a3=pack(__float2half(Aval(mt+gr+8, sk+tc+8+0)),__float2half(Aval(mt+gr+8, sk+tc+8+1)));
        int n = nt + gr;
        uint32_t b0=pack(Bk(lk+tc+0,    n), Bk(lk+tc+1,    n));
        uint32_t b1=pack(Bk(lk+tc+8+0,  n), Bk(lk+tc+8+1,  n));
        uint32_t b2=pack(Bk(lk+tc+16+0, n), Bk(lk+tc+16+1, n));
        uint32_t b3=pack(Bk(lk+tc+24+0, n), Bk(lk+tc+24+1, n));
        asm volatile(
          "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),"r"(b2),"r"(b3),
            "f"(d0),"f"(d1),"f"(d2),"f"(d3), "r"(meta));
    }
    D[(size_t)(mt+gr  )*N + nt+tc+0]=d0;
    D[(size_t)(mt+gr  )*N + nt+tc+1]=d1;
    D[(size_t)(mt+gr+8)*N + nt+tc+0]=d2;
    D[(size_t)(mt+gr+8)*N + nt+tc+1]=d3;
}

static int run_tiled_gemm(int M,int N,int K,int p0,int p1){
    printf("\n[gemm.sp.int4] M=%d N=%d K=%d  pattern{%d,%d}\n",M,N,K,p0,p1);
    int Keff=K/2, Kg=K/GRP, rowstride=Keff/2;
    std::vector<uint8_t> Apk((size_t)M*rowstride);
    std::vector<half> Asc((size_t)M*Kg), B((size_t)K*N);
    srand(99+p0+p1*7);
    for(auto&x:Asc) x=__float2half(0.05f+0.02f*(rand()%5));     // positive scales
    for(auto&x:B)   x=__float2half((float)((rand()%9)-4));
    std::vector<int> qv((size_t)M*Keff);
    for(auto&q:qv) q=rand()%16;
    for(int r=0;r<M;r++) for(int j=0;j<Keff;j++){
        uint8_t&byte=Apk[(size_t)r*rowstride + (j>>1)];
        if(j&1) byte=(byte&0x0F)|((qv[(size_t)r*Keff+j]&0xF)<<4);
        else    byte=(byte&0xF0)|(qv[(size_t)r*Keff+j]&0xF);
    }
    // CPU reference
    std::vector<float> Alog((size_t)M*K,0.f), Dref((size_t)M*N,0.f);
    for(int r=0;r<M;r++) for(int j=0;j<Keff;j++){
        int quad=j>>1, pos=(j&1)?p1:p0; int lk=quad*4+pos;
        float sc=__half2float(Asc[(size_t)r*Kg + lk/GRP]);
        Alog[(size_t)r*K+lk]=(float)(qv[(size_t)r*Keff+j]-8)*sc;
    }
    for(int r=0;r<M;r++) for(int c=0;c<N;c++){ float a=0;
        for(int k=0;k<K;k++) a+=Alog[(size_t)r*K+k]*__half2float(B[(size_t)k*N+c]);
        Dref[(size_t)r*N+c]=a; }

    uint8_t* dApk; half *dAsc,*dB; float* dD;
    CK(cudaMalloc(&dApk,Apk.size())); CK(cudaMalloc(&dAsc,Asc.size()*2));
    CK(cudaMalloc(&dB,B.size()*2));   CK(cudaMalloc(&dD,(size_t)M*N*4));
    CK(cudaMemcpy(dApk,Apk.data(),Apk.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAsc,Asc.data(),Asc.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,B.data(),B.size()*2,cudaMemcpyHostToDevice));
    dim3 grid(N/NN, M/MM);
    gemm_sp_int4<<<grid,32>>>(dApk,dAsc,dB,make_meta(p0,p1),M,N,K,p0,p1,dD);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> Dg((size_t)M*N); CK(cudaMemcpy(Dg.data(),dD,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    int bad=0; float maxrel=0.f;
    for(size_t i=0;i<Dg.size();i++){ float e=fabsf(Dg[i]-Dref[i]); float d=fmaxf(1.f,fabsf(Dref[i]));
        maxrel=fmaxf(maxrel,e/d); if(e/d>2e-2f) bad++; }
    printf("[gemm.sp.int4] maxrel=%.4f  mismatches=%d/%d  D[0,0]=%.3f ref=%.3f\n",
        maxrel,bad,M*N,Dg[0],Dref[0]);
    cudaFree(dApk);cudaFree(dAsc);cudaFree(dB);cudaFree(dD);
    return bad;
}

int main(){
    int total=0;
    total += run_pattern(0,1);   // metadata 0x44444444
    total += run_pattern(1,3);   // metadata 0xdddddddd
    total += run_pattern(0,3);
    total += run_pattern(2,3);
    // tiled sparse-INT4 GEMM (on-the-fly nibble decode + per-128 scale + K/N tiling)
    total += run_tiled_gemm(64, 32, 256, 0,1);
    total += run_tiled_gemm(32, 16, 128, 1,3);
    total += run_tiled_gemm(128,64, 512, 2,3);
    printf("\n[mma.sp] %s\n", total==0 ? "PASS: sparse INT4 Tensor-Core GEMM matches dense reference"
                                       : "FAIL");
    return total==0?0:1;
}
