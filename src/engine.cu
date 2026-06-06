// LFM2.5-8B-A1B pure C/CUDA inference engine (accuracy path: dense INT4 group-wise weights).
// See kb/ for architecture + decisions. Single GPU, single stream, token-by-token + prefill.
//
// Author: g023 (https://github.com/g023/)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "tokenizer.h"

// ---- FP8 E4M3 helpers (intelligent KV cache). E4M3 max normal magnitude = 448. ----
#define E4M3_MAX 448.0f
#define KVGROUP 64               // KV cache quant group: 64 tokens along sequence, shared fp16 scale
static __device__ __forceinline__ uint8_t f2e4m3(float x){
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
static __device__ __forceinline__ float e4m3f(uint8_t b){
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    half hh; memcpy(&hh,&h,2); return __half2float(hh);
}

// ---------------- config (fixed for this model) ----------------
#define H 2048
#define NL 24
#define HEAD_DIM 64
#define NH 32
#define NKV 8
#define GQA (NH / NKV)            // 4
#define VOCAB 128000
#define NDENSE 2
#define NEXP 32
#define TOPK 4
#define MOE_INTER 1792
#define DENSE_INTER 7168
#define INPROJ (3 * H)            // 6144
#define GROUP 128
#define ROPE_THETA 5000000.0f
#define EPS 1e-5f

static const int ATTN_LAYER[NL] = {0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0};

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---------------- weight index ----------------
struct TDesc { uint8_t kind; uint64_t offset, packed, scales; int64_t d0, d1; };

struct Q4 { const uint8_t* packed; const half* scales; int N, K; };
struct F16 { const half* data; int d0, d1; };

struct Model {
    uint8_t* d_blob = nullptr;
    std::unordered_map<std::string, TDesc> idx;

    Q4 q4(const std::string& n) {
        auto it = idx.find(n);
        if (it == idx.end()) { fprintf(stderr,"missing q4 %s\n", n.c_str()); exit(1);}
        TDesc& t = it->second;
        Q4 w; w.packed = d_blob + t.offset; w.scales = (const half*)(d_blob + t.offset + t.packed);
        w.N = (int)t.d0; w.K = (int)t.d1; return w;
    }
    F16 f16(const std::string& n) {
        auto it = idx.find(n);
        if (it == idx.end()) { fprintf(stderr,"missing f16 %s\n", n.c_str()); exit(1);}
        TDesc& t = it->second;
        F16 w; w.data = (const half*)(d_blob + t.offset); w.d0=(int)t.d0; w.d1=(int)t.d1; return w;
    }
};

// ---------------- kernels ----------------

// y[m,n] = sum_k x[m,k] * dequant(W[n,k]).  grid=(ceil(N/8), M). block=256 (8 warps). smem = K floats.
__global__ void gemv_int4(const half* __restrict__ x, const uint8_t* __restrict__ packed,
                          const half* __restrict__ scales, float* __restrict__ y,
                          int N, int K, int Kg) {
    extern __shared__ float xs[];
    int m = blockIdx.y;
    const half* xm = x + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = __half2float(xm[i]);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const uint8_t* prow = packed + (size_t)n * (K / 2);
    const half* srow = scales + (size_t)n * Kg;
    float acc = 0.f;
    int half_k = K / 2;
    for (int j = lane; j < half_k; j += 32) {
        uint8_t b = prow[j];
        int k = 2 * j;
        float sc = __half2float(srow[k >> 7]);   // group = k/128
        float w0 = (float)((int)(b & 0xF) - 8) * sc;
        float w1 = (float)((int)(b >> 4) - 8) * sc;
        acc += w0 * xs[k] + w1 * xs[k + 1];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)m * N + n] = acc;
}

// fp16-weight GEMV (for lm_head / embed^T): y[m,n] = sum_k x[m,k]*W[n,k]. grid=(ceil(N/8),M) block256 smem K floats.
__global__ void gemv_fp16(const half* __restrict__ x, const half* __restrict__ W,
                          float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    int m = blockIdx.y;
    const half* xm = x + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = __half2float(xm[i]);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const half* wrow = W + (size_t)n * K;
    float acc = 0.f;
    for (int k = lane; k < K; k += 32) acc += __half2float(wrow[k]) * xs[k];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)m * N + n] = acc;
}

// RMSNorm over last dim D. in[M,D] fp16 -> out[M,D] fp16. weight[D] fp16. one block per row.
__global__ void rmsnorm(const half* __restrict__ in, const half* __restrict__ w,
                        half* __restrict__ out, int M, int D, float eps) {
    int m = blockIdx.x; if (m >= M) return;
    const half* row = in + (size_t)m * D;
    __shared__ float red[256];
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = __half2float(row[i]); ss += v * v; }
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / D + eps);
    half* o = out + (size_t)m * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        o[i] = __float2half(__half2float(row[i]) * inv * __half2float(w[i]));
}

// per-head RMSNorm: x[M, nHeads, HEAD_DIM] in place-ish (out separate). weight[HEAD_DIM].
// grid = (M*nHeads). block = HEAD_DIM.
__global__ void rmsnorm_head(const half* __restrict__ in, const half* __restrict__ w,
                             half* __restrict__ out, int rows, int D, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    const half* row = in + (size_t)r * D;
    __shared__ float red[HEAD_DIM];
    float v = __half2float(row[threadIdx.x]);
    red[threadIdx.x] = v * v; __syncthreads();
    for (int s = D >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / D + eps);
    out[(size_t)r * D + threadIdx.x] = __float2half(v * inv * __half2float(w[threadIdx.x]));
}

// RoPE in place on q[S,NH,HEAD_DIM] and k[S,NKV,HEAD_DIM]. positions[S] absolute. one thread per (token,head,d<32).
__global__ void rope(half* __restrict__ q, half* __restrict__ k, const int* __restrict__ pos,
                     int S, int nq_heads, int nk_heads) {
    int s = blockIdx.x;
    int d = threadIdx.x;             // 0..31
    if (d >= HEAD_DIM/2) return;
    float invf = powf(ROPE_THETA, -(float)(2*d) / HEAD_DIM);
    float ang = (float)pos[s] * invf;
    float cs = cosf(ang), sn = sinf(ang);
    // q heads
    for (int h = 0; h < nq_heads; h++) {
        half* base = q + ((size_t)s * nq_heads + h) * HEAD_DIM;
        float x0 = __half2float(base[d]); float x1 = __half2float(base[d + HEAD_DIM/2]);
        base[d]            = __float2half(x0 * cs - x1 * sn);
        base[d + HEAD_DIM/2] = __float2half(x1 * cs + x0 * sn);
    }
    for (int h = 0; h < nk_heads; h++) {
        half* base = k + ((size_t)s * nk_heads + h) * HEAD_DIM;
        float x0 = __half2float(base[d]); float x1 = __half2float(base[d + HEAD_DIM/2]);
        base[d]            = __float2half(x0 * cs - x1 * sn);
        base[d + HEAD_DIM/2] = __float2half(x1 * cs + x0 * sn);
    }
}

// ---- Intelligent KV cache: FP8 E4M3 committed groups + fp16 tail (current partial group) ----
// Layouts (per attention layer): committed FP8 kc8/vc8 [maxT,NKV,HEAD_DIM] uint8;
// per-group fp16 scales ksc/vsc [maxGroups,NKV]; fp16 tail ktail/vtail [KVGROUP,NKV,HEAD_DIM].
// The "tail" holds tokens [committed..total) of the in-progress group, kept fp16 until full.

// write `cnt` fresh tokens (k,v are offset to the first token) into the tail at slot..slot+cnt-1.
__global__ void kv_write_tail(const half* __restrict__ k, const half* __restrict__ v,
                              half* __restrict__ ktail, half* __restrict__ vtail,
                              int slot, int cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cnt * NKV * HEAD_DIM; if (idx >= total) return;
    int t = idx / (NKV * HEAD_DIM), rest = idx % (NKV * HEAD_DIM);
    size_t dst = (size_t)(slot + t) * NKV * HEAD_DIM + rest;
    ktail[dst] = k[idx]; vtail[dst] = v[idx];
}

// commit a full 64-token group from the tail into FP8, computing one fp16 scale per kv head
// (shared over the 64 tokens x HEAD_DIM block). grid = NKV, block = 256.
__global__ void kv_commit_group(const half* __restrict__ ktail, const half* __restrict__ vtail,
                                uint8_t* __restrict__ kc8, uint8_t* __restrict__ vc8,
                                half* __restrict__ ksc, half* __restrict__ vsc,
                                int dst_tok_base, int grp) {
    int h = blockIdx.x, tid = threadIdx.x;
    __shared__ float redk[256], redv[256];
    float mk = 0.f, mv = 0.f;
    for (int e = tid; e < KVGROUP * HEAD_DIM; e += blockDim.x) {
        int t = e / HEAD_DIM, d = e % HEAD_DIM;
        size_t off = ((size_t)t * NKV + h) * HEAD_DIM + d;
        mk = fmaxf(mk, fabsf(__half2float(ktail[off])));
        mv = fmaxf(mv, fabsf(__half2float(vtail[off])));
    }
    redk[tid] = mk; redv[tid] = mv; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) { redk[tid]=fmaxf(redk[tid],redk[tid+s]);
        redv[tid]=fmaxf(redv[tid],redv[tid+s]); } __syncthreads(); }
    float sck = redk[0] / E4M3_MAX + 1e-12f;
    float scv = redv[0] / E4M3_MAX + 1e-12f;
    if (tid == 0) { ksc[grp*NKV + h] = __float2half(sck); vsc[grp*NKV + h] = __float2half(scv); }
    for (int e = tid; e < KVGROUP * HEAD_DIM; e += blockDim.x) {
        int t = e / HEAD_DIM, d = e % HEAD_DIM;
        size_t off = ((size_t)t * NKV + h) * HEAD_DIM + d;
        size_t dst = ((size_t)(dst_tok_base + t) * NKV + h) * HEAD_DIM + d;
        kc8[dst] = f2e4m3(__half2float(ktail[off]) / sck);
        vc8[dst] = f2e4m3(__half2float(vtail[off]) / scv);
    }
}

// Fused FlashAttention over the FP8 cache + fp16 tail, online softmax, one thread per (query,head).
// `committed` = number of tokens stored in FP8 (a multiple of KVGROUP); keys [committed..qpos] live
// in the fp16 tail at index t-committed. When attn_sum != null, a second pass accumulates each key's
// normalized attention probability (summed over query heads) for H2O eviction (no extra mem traffic
// beyond re-reading K). `slot2tok` maps cache slots -> logical token positions (identity unless the
// circular eviction buffer is active).
__global__ void attention_fused(const half* __restrict__ q,
                                const uint8_t* __restrict__ kc8, const uint8_t* __restrict__ vc8,
                                const half* __restrict__ ksc, const half* __restrict__ vsc,
                                const half* __restrict__ ktail, const half* __restrict__ vtail,
                                int committed, half* __restrict__ out, float* __restrict__ attn_sum,
                                int S, int past, float scale) {
    int i = blockIdx.x;            // query token
    int h = threadIdx.x;           // head 0..NH-1
    if (h >= NH) return;
    int kvh = h / GQA;
    int qpos = past + i;
    const half* qv = q + ((size_t)i * NH + h) * HEAD_DIM;
    float qreg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) qreg[d] = __half2float(qv[d]);
    float o[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o[d] = 0.f;
    float m = -1e30f, l = 0.f;
    int cend = committed < qpos+1 ? committed : qpos+1;   // FP8 keys [0,cend)
    // segment 1: committed keys (FP8 E4M3, branch-free inner loops)
    for (int t = 0; t < cend; t++) {
        float kscl = __half2float(ksc[(t/KVGROUP)*NKV + kvh]);
        float vscl = __half2float(vsc[(t/KVGROUP)*NKV + kvh]);
        const uint8_t* kp8 = kc8 + ((size_t)t*NKV + kvh)*HEAD_DIM;
        const uint8_t* vp8 = vc8 + ((size_t)t*NKV + kvh)*HEAD_DIM;
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * (e4m3f(kp8[d])*kscl);
        s *= scale;
        float mnew = fmaxf(m, s), c = __expf(m - mnew), p = __expf(s - mnew);
        l = l * c + p;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) o[d] = o[d]*c + p*(e4m3f(vp8[d])*vscl);
        m = mnew;
    }
    // segment 2: tail keys (fp16)
    for (int t = cend; t <= qpos; t++) {
        const half* kpf = ktail + ((size_t)(t-committed)*NKV + kvh)*HEAD_DIM;
        const half* vpf = vtail + ((size_t)(t-committed)*NKV + kvh)*HEAD_DIM;
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * __half2float(kpf[d]);
        s *= scale;
        float mnew = fmaxf(m, s), c = __expf(m - mnew), p = __expf(s - mnew);
        l = l * c + p;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) o[d] = o[d]*c + p*__half2float(vpf[d]);
        m = mnew;
    }
    float inv = 1.f / l;
    half* ov = out + ((size_t)i * NH + h) * HEAD_DIM;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) ov[d] = __float2half(o[d] * inv);

    if (attn_sum) {                // H2O: accumulate normalized attention mass per key
        for (int t = 0; t < cend; t++) {
            float kscl = __half2float(ksc[(t/KVGROUP)*NKV+kvh]);
            const uint8_t* kp8 = kc8 + ((size_t)t*NKV+kvh)*HEAD_DIM;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * (e4m3f(kp8[d])*kscl);
            atomicAdd(&attn_sum[t], __expf(s*scale - m) * inv);
        }
        for (int t = cend; t <= qpos; t++) {
            const half* kpf = ktail + ((size_t)(t-committed)*NKV + kvh)*HEAD_DIM;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * __half2float(kpf[d]);
            atomicAdd(&attn_sum[t], __expf(s*scale - m) * inv);
        }
    }
}

// ---- H2O (Heavy-Hitter + local window) eviction mode: per-token-scaled FP8 circular buffer ----
// Each physical slot holds one token's KV in FP8 with its own per-head fp16 scale, so a slot can be
// reused (evicted) wholesale without disturbing any shared group scale. slot_pos[] gives each slot's
// logical token position (for causal masking + eviction policy); attn_sum[] is its cumulative
// attention mass. Used only when --kv-budget is set; the default group-64 path is unchanged.

// quantize S fresh tokens into chosen physical slots (one per-head scale per token). grid=S*NKV, block=HEAD_DIM.
__global__ void kv_quant_write(const half* __restrict__ k, const half* __restrict__ v,
                               uint8_t* __restrict__ kf, uint8_t* __restrict__ vf,
                               half* __restrict__ ks, half* __restrict__ vs,
                               const int* __restrict__ dst_slots, int S) {
    int jh = blockIdx.x, j = jh / NKV, h = jh % NKV, d = threadIdx.x;
    size_t src = ((size_t)j*NKV + h)*HEAD_DIM + d;
    float kv = __half2float(k[src]), vv = __half2float(v[src]);
    __shared__ float rk[HEAD_DIM], rv[HEAD_DIM];
    rk[d] = fabsf(kv); rv[d] = fabsf(vv); __syncthreads();
    for (int s = HEAD_DIM>>1; s > 0; s >>= 1) { if (d < s) { rk[d]=fmaxf(rk[d],rk[d+s]);
        rv[d]=fmaxf(rv[d],rv[d+s]); } __syncthreads(); }
    float sck = rk[0]/E4M3_MAX + 1e-12f, scv = rv[0]/E4M3_MAX + 1e-12f;
    int slot = dst_slots[j];
    size_t dst = ((size_t)slot*NKV + h)*HEAD_DIM + d;
    kf[dst] = f2e4m3(kv / sck); vf[dst] = f2e4m3(vv / scv);
    if (d == 0) { ks[slot*NKV + h] = __float2half(sck); vs[slot*NKV + h] = __float2half(scv); }
}

// fused FlashAttention over the H2O circular buffer (per-slot FP8 + scale), causal mask via slot_pos,
// accumulating cumulative attention mass per slot. one thread per (query, head).
__global__ void attention_h2o(const half* __restrict__ q,
                              const uint8_t* __restrict__ kf, const uint8_t* __restrict__ vf,
                              const half* __restrict__ ks, const half* __restrict__ vs,
                              const int* __restrict__ slot_pos, int n_live,
                              half* __restrict__ out, float* __restrict__ attn_sum,
                              int S, int past, float scale) {
    int i = blockIdx.x, h = threadIdx.x; if (h >= NH) return;
    int kvh = h / GQA, qpos = past + i;
    const half* qv = q + ((size_t)i*NH + h)*HEAD_DIM;
    float qreg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) qreg[d] = __half2float(qv[d]);
    float o[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o[d] = 0.f;
    float m = -1e30f, l = 0.f;
    for (int t = 0; t < n_live; t++) {
        if (slot_pos[t] > qpos) continue;                 // causal
        float sck = __half2float(ks[t*NKV + kvh]);
        float scv = __half2float(vs[t*NKV + kvh]);
        const uint8_t* kp = kf + ((size_t)t*NKV + kvh)*HEAD_DIM;
        const uint8_t* vp = vf + ((size_t)t*NKV + kvh)*HEAD_DIM;
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * (e4m3f(kp[d])*sck);
        s *= scale;
        float mnew = fmaxf(m, s), c = __expf(m - mnew), p = __expf(s - mnew);
        l = l*c + p;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) o[d] = o[d]*c + p*(e4m3f(vp[d])*scv);
        m = mnew;
    }
    float inv = 1.f / l;
    half* ov = out + ((size_t)i*NH + h)*HEAD_DIM;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) ov[d] = __float2half(o[d] * inv);
    if (attn_sum) {
        for (int t = 0; t < n_live; t++) {
            if (slot_pos[t] > qpos) continue;
            float sck = __half2float(ks[t*NKV + kvh]);
            const uint8_t* kp = kf + ((size_t)t*NKV + kvh)*HEAD_DIM;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * (e4m3f(kp[d])*sck);
            atomicAdd(&attn_sum[t], __expf(s*scale - m) * inv);
        }
    }
}

// ---- Flash-Decoding: parallel single-token attention (default group-64 FP8 + fp16 tail path) ----
// The original attention_fused launches grid=S, block=NH. For decode S==1 that is ONE block of 32
// threads (a single warp) serially scanning the whole context -> the long-context bottleneck (cost
// grows linearly with context on a single SM). Flash-Decoding splits the key range across NSPLIT
// blocks per head so the whole GPU is saturated; a tiny combine kernel merges the partial softmaxes.
// Numerically identical to attention_fused (online softmax, fp32 accumulate) -> coherence unchanged.
#define NSPLIT 16
#define ADEC_THREADS 128

// One block per (head, split): scans its slice of keys, writes partial (m,l,o) for that head/split.
__global__ void attn_decode_split(const half* __restrict__ q,
        const uint8_t* __restrict__ kc8, const uint8_t* __restrict__ vc8,
        const half* __restrict__ ksc, const half* __restrict__ vsc,
        const half* __restrict__ ktail, const half* __restrict__ vtail,
        int committed, int qpos, float scale,
        float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ po) {
    int h = blockIdx.x, sp = blockIdx.y, kvh = h / GQA;
    int tid = threadIdx.x, nthr = blockDim.x;
    __shared__ float qs[HEAD_DIM];
    if (tid < HEAD_DIM) qs[tid] = __half2float(q[(size_t)h*HEAD_DIM + tid]);
    __syncthreads();

    int total = qpos + 1;
    int chunk = (total + NSPLIT - 1) / NSPLIT;
    int kstart = sp * chunk;
    int kend = total < kstart + chunk ? total : kstart + chunk;
    int cend = committed < total ? committed : total;     // FP8 region [0,cend)

    float m = -1e30f, l = 0.f, o[HEAD_DIM];
    #pragma unroll
    for (int d=0; d<HEAD_DIM; d++) o[d]=0.f;
    for (int t = kstart + tid; t < kend; t += nthr) {
        float s = 0.f;
        if (t < cend) {
            float kscl = __half2float(ksc[(t/KVGROUP)*NKV + kvh]);
            float vscl = __half2float(vsc[(t/KVGROUP)*NKV + kvh]);
            const uint8_t* kp = kc8 + ((size_t)t*NKV + kvh)*HEAD_DIM;
            const uint8_t* vp = vc8 + ((size_t)t*NKV + kvh)*HEAD_DIM;
            #pragma unroll
            for (int d=0; d<HEAD_DIM; d++) s += qs[d]*(e4m3f(kp[d])*kscl);
            s *= scale;
            float mnew=fmaxf(m,s), c=__expf(m-mnew), p=__expf(s-mnew);
            l = l*c + p;
            #pragma unroll
            for (int d=0; d<HEAD_DIM; d++) o[d]=o[d]*c + p*(e4m3f(vp[d])*vscl);
            m = mnew;
        } else {
            const half* kp = ktail + ((size_t)(t-committed)*NKV + kvh)*HEAD_DIM;
            const half* vp = vtail + ((size_t)(t-committed)*NKV + kvh)*HEAD_DIM;
            #pragma unroll
            for (int d=0; d<HEAD_DIM; d++) s += qs[d]*__half2float(kp[d]);
            s *= scale;
            float mnew=fmaxf(m,s), c=__expf(m-mnew), p=__expf(s-mnew);
            l = l*c + p;
            #pragma unroll
            for (int d=0; d<HEAD_DIM; d++) o[d]=o[d]*c + p*__half2float(vp[d]);
            m = mnew;
        }
    }
    // block-reduce the partial softmax state across threads
    extern __shared__ float red[];                 // [nthr] m, [nthr] l, [nthr*HEAD_DIM] o
    float* sm_m = red; float* sm_l = red + nthr; float* sm_o = red + 2*nthr;
    sm_m[tid] = m; __syncthreads();
    for (int s=nthr>>1; s>0; s>>=1){ if(tid<s) sm_m[tid]=fmaxf(sm_m[tid],sm_m[tid+s]); __syncthreads(); }
    float bm = sm_m[0];
    float alpha = __expf(m - bm);
    sm_l[tid] = l*alpha;
    #pragma unroll
    for (int d=0; d<HEAD_DIM; d++) sm_o[tid*HEAD_DIM+d] = o[d]*alpha;
    __syncthreads();
    for (int s=nthr>>1; s>0; s>>=1){
        if(tid<s){ sm_l[tid]+=sm_l[tid+s];
            #pragma unroll
            for(int d=0; d<HEAD_DIM; d++) sm_o[tid*HEAD_DIM+d]+=sm_o[(tid+s)*HEAD_DIM+d]; }
        __syncthreads();
    }
    if (tid==0){ int pi=h*NSPLIT+sp; pm[pi]=bm; pl[pi]=sm_l[0];
        for(int d=0; d<HEAD_DIM; d++) po[(size_t)pi*HEAD_DIM+d]=sm_o[d]; }
}

// Combine NSPLIT partials per head into the final attention output. grid=NH, block=HEAD_DIM.
__global__ void attn_decode_combine(const float* __restrict__ pm, const float* __restrict__ pl,
                                    const float* __restrict__ po, half* __restrict__ out) {
    int h = blockIdx.x, d = threadIdx.x;
    float gm = -1e30f;
    #pragma unroll
    for (int s=0;s<NSPLIT;s++) gm = fmaxf(gm, pm[h*NSPLIT+s]);
    float l = 0.f, acc = 0.f;
    for (int s=0;s<NSPLIT;s++){ int pi=h*NSPLIT+s;
        if (pl[pi] <= 0.f) continue;
        float w = __expf(pm[pi]-gm);
        l += pl[pi]*w; acc += po[(size_t)pi*HEAD_DIM+d]*w;
    }
    out[(size_t)h*HEAD_DIM + d] = __float2half(acc / l);
}

// elementwise: Bx[s,c] = B[s,c]*xg[s,c]. in is inproj output [S, 3H] (B|C|xg). writes B,C,Bx buffers [S,H].
__global__ void conv_pre(const float* __restrict__ inproj, half* __restrict__ Cbuf,
                         half* __restrict__ Bx, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * H) return;
    int s = idx / H, c = idx % H;
    const float* row = inproj + (size_t)s * INPROJ;
    float B = row[c];
    float Cv = row[H + c];
    float xg = row[2*H + c];
    Cbuf[idx] = __float2half(Cv);
    Bx[idx] = __float2half(B * xg);
}

// causal depthwise conv k=3 with left context from state[2*H]. y[s,c]=C[s,c]*sum_j Bx[s-2+j,c]*w[c,j].
// updates state to last 2 Bx. grid over (S*H). w fp16 [H,3].
__global__ void conv_apply(const half* __restrict__ Bx, const half* __restrict__ Cbuf,
                           const half* __restrict__ w, half* __restrict__ state,
                           half* __restrict__ y, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * H) return;
    int s = idx / H, c = idx % H;
    auto bx = [&](int ss)->float{
        if (ss >= 0) return __half2float(Bx[(size_t)ss * H + c]);
        int si = ss + 2;               // -2 -> state0, -1 -> state1
        return __half2float(state[si * H + c]);
    };
    float w0 = __half2float(w[c*3+0]), w1 = __half2float(w[c*3+1]), w2 = __half2float(w[c*3+2]);
    float co = bx(s-2)*w0 + bx(s-1)*w1 + bx(s)*w2;
    y[idx] = __float2half(__half2float(Cbuf[idx]) * co);
}

__global__ void conv_update_state(const half* __restrict__ Bx, half* __restrict__ state, int S) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= H) return;
    // new last two Bx (positions S-2, S-1) with state carryover if S==1
    float prev1 = __half2float(state[1*H + c]);
    if (S == 1) {
        state[0*H + c] = __float2half(prev1);
        state[1*H + c] = Bx[c];
    } else {
        state[0*H + c] = Bx[(size_t)(S-2)*H + c];
        state[1*H + c] = Bx[(size_t)(S-1)*H + c];
    }
}

__global__ void add_inplace(half* __restrict__ a, const half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}
// add float src into half dst (residual from gemv float output): a += src
__global__ void add_f32_inplace(half* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + b[i]);
}
// copy float -> half
__global__ void f32_to_f16(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    dst[i] = __float2half(src[i]);
}
// SwiGLU: out[i] = silu(g[i]) * u[i]   (g,u float [M,D]) -> half
__global__ void swiglu(const float* __restrict__ g, const float* __restrict__ u, half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float x = g[i];
    float s = x / (1.f + __expf(-x));
    out[i] = __float2half(s * u[i]);
}
// embedding gather: out[s,:] = embed[ids[s],:]
__global__ void embed_gather(const half* __restrict__ embed, const int* __restrict__ ids,
                             half* __restrict__ out, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= S * H) return;
    int s = idx / H, c = idx % H;
    out[idx] = embed[(size_t)ids[s] * H + c];
}
// scale-accumulate: dst[c] += w * src[c]  (src float, dst half)
__global__ void acc_scaled(half* __restrict__ dst, const float* __restrict__ src, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + w * src[i]);
}

// argmax over logits[n] -> *out (single block). Avoids a 512KB D2H copy + CPU scan every token.
__global__ void argmax_kernel(const float* __restrict__ logits, int n, int* __restrict__ out){
    __shared__ float bv[256]; __shared__ int bi[256];
    int tid = threadIdx.x; float best = -1e30f; int bidx = 0;
    for (int i = tid; i < n; i += blockDim.x){ float v = logits[i]; if (v > best){ best = v; bidx = i; } }
    bv[tid] = best; bi[tid] = bidx; __syncthreads();
    for (int s = blockDim.x>>1; s > 0; s >>= 1){
        if (tid < s && bv[tid+s] > bv[tid]){ bv[tid] = bv[tid+s]; bi[tid] = bi[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) *out = bi[0];
}

// ---------------- host helpers ----------------
static int divup(int a, int b){ return (a + b - 1) / b; }

static const char* g_dbgdir = nullptr;
static bool g_dump_layers = false;
static bool g_evict = false;            // H2O KV eviction enabled (task 3)
static int  g_kv_budget = 1<<30;        // max live KV tokens (cache slots) before eviction
static int  g_kv_window = 0;            // local window W: last W tokens never evicted (0 = budget/2)
static void dump_hidden(const half* d, int S, const char* tag){
    if(!g_dbgdir) return;
    std::vector<half> hh((size_t)S*H); cudaMemcpy(hh.data(), d, (size_t)S*H*2, cudaMemcpyDeviceToHost);
    std::vector<float> ff((size_t)S*H); for(size_t i=0;i<hh.size();i++) ff[i]=__half2float(hh[i]);
    char path[512]; snprintf(path,512,"%s/%s.f32", g_dbgdir, tag);
    FILE* f=fopen(path,"wb"); fwrite(ff.data(),4,ff.size(),f); fclose(f);
}

struct Buffers {
    int maxS;
    half *h_in;         // hidden [maxS,H]
    half *norm;         // normed [maxS,H]
    float *gA, *gB;     // generic gemv outputs (sized for largest N among uses)
    half  *hbuf, *hbuf2;// half temporaries [maxS, up to 3H]
    half  *q, *k, *v, *attn_out;
    int   *d_pos;
    // Intelligent KV cache (attention layers): FP8 committed + fp16 tail + per-group scales + attn_sum
    uint8_t *kc8[NL], *vc8[NL];     // committed FP8 E4M3 [maxT,NKV,HEAD_DIM]
    half  *ksc[NL], *vsc[NL];       // per-group fp16 scales [maxGroups,NKV]
    half  *ktail[NL], *vtail[NL];   // current partial group, fp16 [KVGROUP,NKV,HEAD_DIM]
    float *attn_sum[NL];            // H2O cumulative attention per cache slot [maxT]
    int    committed[NL];           // # tokens committed to FP8 (multiple of KVGROUP)
    half  *conv_state[NL];          // conv layers only
    int   maxT;
    // H2O eviction-mode circular buffer (per attn layer), allocated only when g_evict
    uint8_t *hk[NL], *hv[NL];       // [C,NKV,HEAD_DIM] per-token FP8
    half  *hks[NL], *hvs[NL];       // [C,NKV] per-token,per-head scales
    float *hattn[NL];               // [C] cumulative attention per slot
    int   *d_slotpos[NL];           // [C] device slot->logical-position map
    std::vector<int> slotpos_h[NL]; // host mirror of slot->position
    int    n_live[NL];              // occupied slots
    int   *d_dst;                   // [maxS] scratch: target slots for fresh tokens
    float *d_router;    // [maxS,32]
    float *moe_g, *moe_u; // [1, MOE_INTER]
    half  *moe_h;       // [1, MOE_INTER]
    half  *expert_out_acc; // [maxS,H]
    float *d_logits;    // [VOCAB]
    // Flash-Decoding partials (single-token decode): per (head,split) m,l and o
    float *pm, *pl, *po;   // pm/pl [NH*NSPLIT], po [NH*NSPLIT*HEAD_DIM]
    int   *d_argmax;       // device scratch: argmax(logits)
    int   *d_tok;          // persistent embed-id buffer [maxS] (no per-token malloc)
};

// run y = gemv(int4 W) for M rows. W.K is inner dim.
static void run_gemv_int4(Q4 w, const half* x, float* y, int M){
    dim3 grid(divup(w.N, 8), M);
    size_t sm = (size_t)w.K * sizeof(float);
    gemv_int4<<<grid, 256, sm>>>(x, w.packed, w.scales, y, w.N, w.K, w.K / GROUP);
}
static void run_gemv_fp16(F16 w, const half* x, float* y, int M, int N, int K){
    dim3 grid(divup(N, 8), M);
    size_t sm = (size_t)K * sizeof(float);
    gemv_fp16<<<grid, 256, sm>>>(x, w.data, y, N, K);
}

int main(int argc, char** argv) {
    const char* dir = "scratch/engine_weights";  // override with --dir
    const char* ids_path = nullptr;
    const char* prompt = nullptr;      // chat-wrapped user message
    const char* raw = nullptr;         // raw text, no chat template
    int n_new = 48;
    bool stream = true;
    const char* dump_path = nullptr;   // if set, dump per-layer hidden of prefill last token
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--ids") && i+1<argc) ids_path = argv[++i];
        else if (!strcmp(argv[i],"--prompt") && i+1<argc) prompt = argv[++i];
        else if (!strcmp(argv[i],"--raw") && i+1<argc) raw = argv[++i];
        else if (!strcmp(argv[i],"--n") && i+1<argc) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--no-stream")) stream = false;
        else if (!strcmp(argv[i],"--dump") && i+1<argc) dump_path = argv[++i];
        else if (!strcmp(argv[i],"--dbgdir") && i+1<argc) g_dbgdir = argv[++i];
        else if (!strcmp(argv[i],"--dir") && i+1<argc) dir = argv[++i];
        else if (!strcmp(argv[i],"--kv-budget") && i+1<argc) { g_kv_budget = atoi(argv[++i]); g_evict = true; }
        else if (!strcmp(argv[i],"--kv-window") && i+1<argc) g_kv_window = atoi(argv[++i]);
    }
    if (!ids_path && !prompt && !raw) ids_path = "scratch/oracle/input_ids.i32";

    // tokenizer
    Tokenizer tk;
    std::string tokdir = std::string(dir) + "/tok";
    bool have_tok = tk.load(tokdir);

    // ---- load index ----
    std::string ip = std::string(dir) + "/index.bin";
    FILE* f = fopen(ip.c_str(), "rb"); if(!f){perror("index.bin");return 1;}
    uint32_t nt; fread(&nt,4,1,f);
    Model M;
    for (uint32_t i=0;i<nt;i++){
        uint8_t nl; fread(&nl,1,1,f); std::string nm(nl,0); fread(&nm[0],1,nl,f);
        TDesc t; fread(&t.kind,1,1,f); fread(&t.offset,8,1,f); fread(&t.packed,8,1,f);
        fread(&t.scales,8,1,f); fread(&t.d0,8,1,f); fread(&t.d1,8,1,f);
        M.idx[nm]=t;
    }
    fclose(f);
    printf("[engine] index: %u tensors\n", nt);

    // ---- load weights blob ----
    std::string wp = std::string(dir) + "/weights.bin";
    f = fopen(wp.c_str(),"rb"); if(!f){perror("weights.bin");return 1;}
    fseek(f,0,SEEK_END); size_t blob_sz = ftell(f); fseek(f,0,SEEK_SET);
    printf("[engine] weights.bin = %.2f GB, uploading...\n", blob_sz/1e9);
    CK(cudaMalloc(&M.d_blob, blob_sz));
    {
        size_t chunk = 256ull<<20; std::vector<uint8_t> buf(chunk);
        size_t done=0;
        while(done<blob_sz){ size_t r=fread(buf.data(),1,std::min(chunk,blob_sz-done),f);
            CK(cudaMemcpy(M.d_blob+done, buf.data(), r, cudaMemcpyHostToDevice)); done+=r;
            printf("\r[engine] upload %.1f%%", 100.0*done/blob_sz); fflush(stdout);}
        printf("\n");
    }
    fclose(f);

    // ---- build input ids (from --prompt/--raw via tokenizer, or --ids file) ----
    std::vector<int> ids;
    if (prompt || raw) {
        if (!have_tok) { fprintf(stderr,"tokenizer needed for --prompt/--raw\n"); return 1; }
        std::string text;
        if (prompt) text = std::string("<|startoftext|><|im_start|>user\n") + prompt +
                           "<|im_end|>\n<|im_start|>assistant\n";
        else text = raw;
        ids = tk.encode(text);
    } else {
        f = fopen(ids_path,"rb"); if(!f){perror("ids");return 1;}
        fseek(f,0,SEEK_END); long isz=ftell(f); fseek(f,0,SEEK_SET);
        int n=isz/4; ids.resize(n); fread(ids.data(),4,n,f); fclose(f);
    }
    int S0 = ids.size();
    printf("[engine] prompt %d tokens:", S0); for(int x:ids) printf(" %d",x); printf("\n");

    // ---- alloc buffers ----
    Buffers B; B.maxS = S0 + 4; B.maxT = S0 + n_new + 4;
    int maxS = B.maxS, maxT = B.maxT;
    CK(cudaMalloc(&B.h_in,  (size_t)maxS*H*2));
    CK(cudaMalloc(&B.norm,  (size_t)maxS*H*2));
    CK(cudaMalloc(&B.hbuf,  (size_t)maxS*INPROJ*2));
    CK(cudaMalloc(&B.hbuf2, (size_t)maxS*INPROJ*2));
    CK(cudaMalloc(&B.gA, (size_t)maxS*DENSE_INTER*4));
    CK(cudaMalloc(&B.gB, (size_t)maxS*DENSE_INTER*4));
    CK(cudaMalloc(&B.q, (size_t)maxS*NH*HEAD_DIM*2));
    CK(cudaMalloc(&B.k, (size_t)maxS*NKV*HEAD_DIM*2));
    CK(cudaMalloc(&B.v, (size_t)maxS*NKV*HEAD_DIM*2));
    CK(cudaMalloc(&B.attn_out, (size_t)maxS*NH*HEAD_DIM*2));
    CK(cudaMalloc(&B.d_pos, maxS*4));
    CK(cudaMalloc(&B.d_router, (size_t)maxS*NEXP*4));
    CK(cudaMalloc(&B.moe_g, (size_t)MOE_INTER*4));
    CK(cudaMalloc(&B.moe_u, (size_t)MOE_INTER*4));
    CK(cudaMalloc(&B.moe_h, (size_t)MOE_INTER*2));
    CK(cudaMalloc(&B.expert_out_acc, (size_t)maxS*H*2));
    CK(cudaMalloc(&B.d_logits, (size_t)VOCAB*4));
    CK(cudaMalloc(&B.pm, (size_t)NH*NSPLIT*4));
    CK(cudaMalloc(&B.pl, (size_t)NH*NSPLIT*4));
    CK(cudaMalloc(&B.po, (size_t)NH*NSPLIT*HEAD_DIM*4));
    CK(cudaMalloc(&B.d_argmax, 4));
    CK(cudaMalloc(&B.d_tok, (size_t)maxS*4));
    int maxGroups = maxT / KVGROUP + 2;
    for(int l=0;l<NL;l++){
        if(ATTN_LAYER[l]){
            CK(cudaMalloc(&B.kc8[l],(size_t)maxT*NKV*HEAD_DIM));
            CK(cudaMalloc(&B.vc8[l],(size_t)maxT*NKV*HEAD_DIM));
            CK(cudaMalloc(&B.ksc[l],(size_t)maxGroups*NKV*2));
            CK(cudaMalloc(&B.vsc[l],(size_t)maxGroups*NKV*2));
            CK(cudaMalloc(&B.ktail[l],(size_t)KVGROUP*NKV*HEAD_DIM*2));
            CK(cudaMalloc(&B.vtail[l],(size_t)KVGROUP*NKV*HEAD_DIM*2));
            CK(cudaMalloc(&B.attn_sum[l],(size_t)maxT*4));
            CK(cudaMemset(B.attn_sum[l],0,(size_t)maxT*4));
            B.committed[l] = 0;
        }
        else { CK(cudaMalloc(&B.conv_state[l],(size_t)2*H*2));
               CK(cudaMemset(B.conv_state[l],0,(size_t)2*H*2)); }
    }
    if (g_evict) {
        if (g_kv_budget < S0) { g_kv_budget = S0; }            // budget must hold the prompt
        if (g_kv_budget > maxT) g_kv_budget = maxT;            // no point exceeding total context
        if (g_kv_window <= 0) g_kv_window = g_kv_budget / 2;    // default local window
        if (g_kv_window >= g_kv_budget) g_kv_window = g_kv_budget - 1;
        int C = g_kv_budget;
        printf("[engine] H2O eviction ON: budget=%d slots, local window W=%d\n", C, g_kv_window);
        CK(cudaMalloc(&B.d_dst, (size_t)maxS*4));
        for (int l=0;l<NL;l++) if (ATTN_LAYER[l]) {
            CK(cudaMalloc(&B.hk[l],(size_t)C*NKV*HEAD_DIM));
            CK(cudaMalloc(&B.hv[l],(size_t)C*NKV*HEAD_DIM));
            CK(cudaMalloc(&B.hks[l],(size_t)C*NKV*2));
            CK(cudaMalloc(&B.hvs[l],(size_t)C*NKV*2));
            CK(cudaMalloc(&B.hattn[l],(size_t)C*4)); CK(cudaMemset(B.hattn[l],0,(size_t)C*4));
            CK(cudaMalloc(&B.d_slotpos[l],(size_t)C*4));
            B.slotpos_h[l].assign(C, -1);
            B.n_live[l] = 0;
        }
    }
    F16 embed = M.f16("model.embed_tokens.weight");
    F16 emb_norm = M.f16("model.embedding_norm.weight");

    // Expert bias is constant across tokens -> fetch to host once (was a D2H copy + sync per MoE
    // layer per token: 22 needless syncs/token on the decode critical path).
    std::vector<std::vector<float>> ebias(NL);
    for (int l = NDENSE; l < NL; l++) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        F16 ebt = M.f16(p+"feed_forward.expert_bias");
        std::vector<half> ebh(NEXP); CK(cudaMemcpy(ebh.data(), ebt.data, NEXP*2, cudaMemcpyDeviceToHost));
        ebias[l].resize(NEXP); for(int e=0;e<NEXP;e++) ebias[l][e]=__half2float(ebh[e]);
    }

    auto layer_forward = [&](int S, int past, int* h_pos){
        CK(cudaMemcpy(B.d_pos, h_pos, S*4, cudaMemcpyHostToDevice));
        for(int l=0;l<NL;l++){
            std::string p = "model.layers." + std::to_string(l) + ".";
            // operator_norm
            rmsnorm<<<S,256>>>(B.h_in, M.f16(p+"operator_norm.weight").data, B.norm, S, H, EPS);
            if(ATTN_LAYER[l]){
                // q,k,v
                run_gemv_int4(M.q4(p+"self_attn.q_proj.weight"), B.norm, B.gA, S); // [S,2048]
                run_gemv_int4(M.q4(p+"self_attn.k_proj.weight"), B.norm, B.gB, S); // [S,512]
                f32_to_f16<<<divup(S*NH*HEAD_DIM,256),256>>>(B.gA, B.q, S*NH*HEAD_DIM);
                f32_to_f16<<<divup(S*NKV*HEAD_DIM,256),256>>>(B.gB, B.k, S*NKV*HEAD_DIM);
                run_gemv_int4(M.q4(p+"self_attn.v_proj.weight"), B.norm, B.gB, S);
                f32_to_f16<<<divup(S*NKV*HEAD_DIM,256),256>>>(B.gB, B.v, S*NKV*HEAD_DIM);
                // qk layernorm (per head)
                rmsnorm_head<<<S*NH,HEAD_DIM>>>(B.q, M.f16(p+"self_attn.q_layernorm.weight").data, B.q, S*NH, HEAD_DIM, EPS);
                rmsnorm_head<<<S*NKV,HEAD_DIM>>>(B.k, M.f16(p+"self_attn.k_layernorm.weight").data, B.k, S*NKV, HEAD_DIM, EPS);
                rope<<<S, HEAD_DIM/2>>>(B.q, B.k, B.d_pos, S, NH, NKV);
                float scale = 1.0f / sqrtf((float)HEAD_DIM);
                if (g_evict) {
                    // H2O circular buffer: pick a physical slot per fresh token (evict heavy-hitter
                    // loser outside the local window once full), quantize into it, then attend.
                    int C = g_kv_budget, W = g_kv_window;
                    std::vector<int> dst(S);
                    std::vector<float> as;             // host attn_sum, fetched lazily on eviction
                    bool fetched = false;
                    for (int j = 0; j < S; j++) {
                        int posj = past + j, slot;
                        if (B.n_live[l] < C) { slot = B.n_live[l]++; }
                        else {
                            if (!fetched) { as.resize(C); CK(cudaMemcpy(as.data(), B.hattn[l], C*4, cudaMemcpyDeviceToHost)); fetched = true; }
                            int best = -1; float bv = 1e30f;
                            for (int t = 0; t < C; t++)                 // evict min attn outside window
                                if (B.slotpos_h[l][t] <= posj - W && as[t] < bv) { bv = as[t]; best = t; }
                            if (best < 0) for (int t = 0; t < C; t++) if (as[t] < bv) { bv = as[t]; best = t; }
                            slot = best; as[slot] = 1e30f;             // don't pick same slot twice this batch
                            CK(cudaMemset(B.hattn[l] + slot, 0, 4));    // reused slot starts fresh
                        }
                        B.slotpos_h[l][slot] = posj; dst[j] = slot;
                    }
                    CK(cudaMemcpy(B.d_dst, dst.data(), S*4, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(B.d_slotpos[l], B.slotpos_h[l].data(), C*4, cudaMemcpyHostToDevice));
                    kv_quant_write<<<S*NKV, HEAD_DIM>>>(B.k, B.v, B.hk[l], B.hv[l], B.hks[l], B.hvs[l], B.d_dst, S);
                    attention_h2o<<<S, NH>>>(B.q, B.hk[l], B.hv[l], B.hks[l], B.hvs[l],
                        B.d_slotpos[l], B.n_live[l], B.attn_out, B.hattn[l], S, past, scale);
                } else {
                    // default: blueprint group-64 FP8 + fp16 tail
                    int placed = 0;
                    while (placed < S) {
                        int gpos = past + placed;
                        int slot = gpos - B.committed[l];          // 0..KVGROUP-1
                        int can  = std::min(S - placed, KVGROUP - slot);
                        const half* ksrc = B.k + (size_t)placed*NKV*HEAD_DIM;
                        const half* vsrc = B.v + (size_t)placed*NKV*HEAD_DIM;
                        kv_write_tail<<<divup(can*NKV*HEAD_DIM,256),256>>>(ksrc, vsrc, B.ktail[l], B.vtail[l], slot, can);
                        placed += can;
                        if (slot + can == KVGROUP) {               // group full -> commit to FP8
                            kv_commit_group<<<NKV,256>>>(B.ktail[l], B.vtail[l], B.kc8[l], B.vc8[l],
                                B.ksc[l], B.vsc[l], B.committed[l], B.committed[l]/KVGROUP);
                            B.committed[l] += KVGROUP;
                        }
                    }
                    if (S == 1) {
                        // Flash-Decoding: saturate the GPU for single-token attention (long-context win)
                        dim3 g(NH, NSPLIT);
                        size_t sm = (size_t)(2*ADEC_THREADS + ADEC_THREADS*HEAD_DIM)*sizeof(float);
                        attn_decode_split<<<g, ADEC_THREADS, sm>>>(B.q, B.kc8[l], B.vc8[l], B.ksc[l],
                            B.vsc[l], B.ktail[l], B.vtail[l], B.committed[l], past, scale, B.pm, B.pl, B.po);
                        attn_decode_combine<<<NH, HEAD_DIM>>>(B.pm, B.pl, B.po, B.attn_out);
                    } else {
                        attention_fused<<<S, NH>>>(B.q, B.kc8[l], B.vc8[l], B.ksc[l], B.vsc[l],
                            B.ktail[l], B.vtail[l], B.committed[l], B.attn_out, nullptr, S, past, scale);
                    }
                }
                run_gemv_int4(M.q4(p+"self_attn.out_proj.weight"), B.attn_out, B.gA, S); // [S,H]
                add_f32_inplace<<<divup(S*H,256),256>>>(B.h_in, B.gA, S*H);
            } else {
                run_gemv_int4(M.q4(p+"conv.in_proj.weight"), B.norm, B.gA, S);  // [S,3H] float
                conv_pre<<<divup(S*H,256),256>>>(B.gA, B.hbuf /*C*/, B.hbuf2 /*Bx*/, S);
                conv_apply<<<divup(S*H,256),256>>>(B.hbuf2, B.hbuf, M.f16(p+"conv.conv.weight").data,
                                                   B.conv_state[l], B.norm /*reuse as y*/, S);
                conv_update_state<<<divup(H,256),256>>>(B.hbuf2, B.conv_state[l], S);
                run_gemv_int4(M.q4(p+"conv.out_proj.weight"), B.norm, B.gA, S);  // [S,H]
                add_f32_inplace<<<divup(S*H,256),256>>>(B.h_in, B.gA, S*H);
            }
            // FFN
            rmsnorm<<<S,256>>>(B.h_in, M.f16(p+"ffn_norm.weight").data, B.norm, S, H, EPS);
            if(l < NDENSE){
                run_gemv_int4(M.q4(p+"feed_forward.w1.weight"), B.norm, B.gA, S); // [S,DI]
                run_gemv_int4(M.q4(p+"feed_forward.w3.weight"), B.norm, B.gB, S);
                swiglu<<<divup(S*DENSE_INTER,256),256>>>(B.gA, B.gB, B.hbuf, S*DENSE_INTER);
                run_gemv_int4(M.q4(p+"feed_forward.w2.weight"), B.hbuf, B.gA, S); // [S,H]
                add_f32_inplace<<<divup(S*H,256),256>>>(B.h_in, B.gA, S*H);
            } else {
                // router
                run_gemv_fp16(M.f16(p+"feed_forward.gate.weight"), B.norm, B.d_router, S, NEXP, H);
                std::vector<float> router(S*NEXP); CK(cudaMemcpy(router.data(), B.d_router, S*NEXP*4, cudaMemcpyDeviceToHost));
                const std::vector<float>& eb = ebias[l];   // cached host-side (constant)
                CK(cudaMemset(B.expert_out_acc, 0, (size_t)S*H*2));
                for(int s=0;s<S;s++){
                    float sig[NEXP], score[NEXP];
                    for(int e=0;e<NEXP;e++){ sig[e]=1.f/(1.f+expf(-router[s*NEXP+e])); score[e]=sig[e]+eb[e]; }
                    int sel[TOPK]; float selw[TOPK];
                    bool used[NEXP]={false};
                    for(int t=0;t<TOPK;t++){ int best=-1; float bv=-1e30f;
                        for(int e=0;e<NEXP;e++) if(!used[e]&&score[e]>bv){bv=score[e];best=e;}
                        used[best]=true; sel[t]=best; selw[t]=sig[best]; }
                    float wsum=0; for(int t=0;t<TOPK;t++) wsum+=selw[t];
                    for(int t=0;t<TOPK;t++) selw[t]/=(wsum+1e-6f);
                    const half* xs = B.norm + (size_t)s*H;
                    for(int t=0;t<TOPK;t++){
                        int e=sel[t];
                        std::string ep = p+"feed_forward.experts."+std::to_string(e)+".";
                        run_gemv_int4(M.q4(ep+"w1.weight"), xs, B.moe_g, 1); // [MOE_INTER]
                        run_gemv_int4(M.q4(ep+"w3.weight"), xs, B.moe_u, 1);
                        swiglu<<<divup(MOE_INTER,256),256>>>(B.moe_g, B.moe_u, B.moe_h, MOE_INTER);
                        run_gemv_int4(M.q4(ep+"w2.weight"), B.moe_h, B.moe_g, 1); // [H]
                        acc_scaled<<<divup(H,256),256>>>(B.expert_out_acc+(size_t)s*H, B.moe_g, selw[t], H);
                    }
                }
                add_inplace<<<divup(S*H,256),256>>>(B.h_in, B.expert_out_acc, S*H);
            }
            if(g_dump_layers){ char tag[32]; snprintf(tag,32,"L%02d",l+1); dump_hidden(B.h_in,S,tag); }
        }
    };

    // ---- prefill ----
    {
        CK(cudaMemcpy(B.d_tok, ids.data(), S0*4, cudaMemcpyHostToDevice));
        embed_gather<<<divup(S0*H,256),256>>>(embed.data, B.d_tok, B.h_in, S0);
    }
    if(g_dbgdir){ g_dump_layers=true; cudaDeviceSynchronize(); dump_hidden(B.h_in,S0,"L00"); }
    { std::vector<int> pos(S0); for(int i=0;i<S0;i++) pos[i]=i; layer_forward(S0, 0, pos.data()); }
    g_dump_layers=false;
    CK(cudaDeviceSynchronize());

    if(dump_path){
        rmsnorm<<<S0,256>>>(B.h_in, emb_norm.data, B.norm, S0, H, EPS);
        std::vector<half> hh(S0*H); CK(cudaMemcpy(hh.data(),B.norm,(size_t)S0*H*2,cudaMemcpyDeviceToHost));
        std::vector<float> ff(S0*H); for(size_t i=0;i<hh.size();i++) ff[i]=__half2float(hh[i]);
        FILE* df=fopen(dump_path,"wb"); fwrite(ff.data(),4,ff.size(),df); fclose(df);
        printf("[engine] dumped final_normed [%d,%d] to %s\n",S0,H,dump_path);
    }

    // ---- decode loop ----
    std::vector<int> out_ids;
    int cur = S0;
    // logits for last prefill token
    auto compute_logits_last = [&](int S){
        rmsnorm<<<S,256>>>(B.h_in, emb_norm.data, B.norm, S, H, EPS);
        // lm_head on last row only
        const half* lastrow = B.norm + (size_t)(S-1)*H;
        run_gemv_fp16(embed, lastrow, B.d_logits, 1, VOCAB, H);
        argmax_kernel<<<1,256>>>(B.d_logits, VOCAB, B.d_argmax);   // GPU argmax: copy 1 int, not 512KB
        int am; CK(cudaMemcpy(&am, B.d_argmax, 4, cudaMemcpyDeviceToHost));
        return am;
    };
    CK(cudaDeviceSynchronize());
    auto t_dec0 = std::chrono::high_resolution_clock::now();
    int next = compute_logits_last(S0);
    out_ids.push_back(next);
    std::string shown;
    auto stream_print = [&](){
        if(!stream || !have_tok) return;
        std::string full = tk.decode(out_ids, /*skip_special=*/false);
        if(full.size() > shown.size()){ fputs(full.c_str()+shown.size(), stdout); fflush(stdout); shown=full; }
    };
    if(stream && have_tok){ printf("\n[engine] ===== output =====\n"); stream_print(); }

    const int EOS = 124900;
    for(int step=0; step<n_new-1 && next!=EOS; step++){
        // embed single token into h_in row0 (persistent d_tok, no per-token malloc/free)
        CK(cudaMemcpy(B.d_tok, &next, 4, cudaMemcpyHostToDevice));
        embed_gather<<<divup(H,256),256>>>(embed.data, B.d_tok, B.h_in, 1);
        int pos = cur; layer_forward(1, cur, &pos); cur++;
        next = compute_logits_last(1);
        out_ids.push_back(next);
        stream_print();
        if(next==EOS) break;
    }
    CK(cudaDeviceSynchronize());
    auto t_dec1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    if(stream && have_tok) printf("\n[engine] ===== end =====\n");
    printf("[engine] generated %zu tokens in %.2fs (%.1f tok/s)\n",
           out_ids.size(), secs, out_ids.size()/secs);
    if(have_tok && !stream) printf("[engine] text: %s\n", tk.decode(out_ids).c_str());
    // write ids to file for python decode
    FILE* of=fopen("scratch/engine_out_ids.i32","wb");
    fwrite(out_ids.data(),4,out_ids.size(),of); fclose(of);
    return 0;
}
