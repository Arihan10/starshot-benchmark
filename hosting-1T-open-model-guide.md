# Hosting a 1-Trillion-Parameter Open Model — A Complete Serving Guide

**Profile for this guide:** optimize for **throughput / cost-per-token**, **rent** GPUs from the cloud, **no quantization**. Written mid-2026. All figures are sourced inline; a consolidated source list is at the end.

---

## 0. TL;DR — the decisions

1. **Every open "~1T" model is a sparse Mixture-of-Experts (MoE).** Kimi K2 (1T total / 32B active), DeepSeek-V3 (671B/37B), DeepSeek-V4-Pro (1.6T/49B), GLM-5.2 (~745B/44B). This one fact drives everything: you are **memory-capacity, memory-bandwidth, and all-to-all-communication bound — not compute bound.** You pay to *hold* a 1T model but only *compute* like a ~35B one.

2. **"No quantization" needs a precise definition (Section 2).** The flagship open 1T models (Kimi K2, DeepSeek) are **natively trained and shipped in FP8**. Serving them in FP8 is **not** quantization — it is their native precision, lossless. Upcasting them to BF16 doubles your memory and halves your GEMM throughput **for zero quality gain**. True "no post-training quantization" means: *serve at the model's native precision, never downcast below it.* Take that stance and FP8-native models stay FP8; BF16-native models (Qwen3, Llama) stay BF16.

3. **Rent B200 on a neocloud** for the best bandwidth-and-FLOPs per dollar (~$3–6/GPU-hr), or **MI300X** for the cheapest way to *hold* the weights (~$2/GPU-hr, ~$1.66 reserved) if your stack is ROCm-validated. **Avoid hyperscalers** — they run 2–4× the neocloud price on identical silicon. A **GB200 NVL72** rack fits the entire model in one NVLink domain (simplest sharding, best decode) but is priciest and allocation-gated.

4. **Stack: SGLang or vLLM (Wide-EP) on Hopper; TensorRT-LLM on Blackwell.** The non-negotiable recipe is **Prefill/Decode disaggregation + Wide Expert Parallelism (EP) + Data-Parallel attention (DP-attention) + EPLB load balancing + dual-microbatch compute/comms overlap + DeepEP + DeepGEMM + CUDA graphs.** Omit any one and you leave large idle bubbles.

5. **Realistic cost at scale:** published reproductions land near **$0.20–0.21 per 1M output tokens** for DeepSeek-V3 / Kimi-K2 at large deployment scale. That number *requires* scale (big aggregate batch, large EP); a tiny 8–16-GPU deployment will not reach it.

**Fast picker:**

| If you want… | Rent | Precision | Engine | Model |
|---|---|---|---|---|
| Cheapest proven $/token, BF16-friendly stack | 2× 8×**H200** or 96×**H100** | native FP8 GEMM + BF16 MLA | **SGLang** | DeepSeek-V3 / Kimi K2 |
| Best per-GPU throughput, newest silicon | 2× 8×**B200** (neocloud) | native FP8/FP4 | **TensorRT-LLM** | DeepSeek-V4 / R1 |
| Simplest sharding, whole model in one memory space | **GB200 NVL72** rack | native | SGLang / TRT-LLM | any |
| Cheapest HBM to just hold weights | 16×**MI300X** | native FP8 | **vLLM**/SGLang (ROCm) | DeepSeek / Kimi |
| Strictly all-BF16 everything (see §2 caveat) | +2× the GPUs | BF16 upcast | SGLang/vLLM | **Qwen3-235B/480B** (BF16-native) |

---

## 1. Step 0 — Understand the workload before you buy anything

### 1.1 It's MoE, and that changes the whole problem

| Model | Total | Active/token | Experts (routed/active/shared) | Attention | Context | Native precision | Open weights? |
|---|---|---|---|---|---|---|---|
| **Kimi K2.6** (Moonshot) | **1T** | 32B | 384 / 8 / 1 | MLA | 256K | FP8 | ✅ |
| **DeepSeek-V4-Pro** | **1.6T** | 49B | 384 / 6 / 1 | CSA+HCA (MLA-derived) | 1M | FP8 base, **FP4 experts on disk** | ✅ |
| **DeepSeek-V3 / V3.1 / V3.2** | 671B | 37B | 256 / 8 / 1 | MLA (V3.2 adds sparse DSA) | 128K | FP8 | ✅ |
| **GLM-5.2** (Zhipu) | ~745B | 44B | 256 / 8 | DSA (DeepSeek-style) | 200K | — | ✅ |
| Qwen3-Coder-480B | 480B | 35B | 128 / 8 | **GQA** | up to 1M | **BF16** | ✅ |
| Qwen3-Max (1T+) | 1T+ | — | MoE | — | 262K | — | ❌ API-only |
| Llama 4 Behemoth (~2T) | ~2T | 288B | 16 | GQA | — | BF16 | ❌ unreleased |

The load-bearing consequence of MoE sparsity:

- **Compute per token is small** — Kimi K2 fires 32B of 1T params (~3.2%); DeepSeek-V3 fires 37B of 671B. Arithmetically it runs like a ~35B dense model.
- **But routing is data-dependent and changes every token**, so **all 256–384 experts must be resident in HBM simultaneously.** You pay the full 1–2 TB memory-capacity cost to serve something that only *computes* like a 35B model.
- Therefore, at **decode** (one token at a time, low arithmetic intensity), the bottleneck is **streaming the selected experts' weights out of HBM** and the **cross-GPU all-to-all** that ships tokens to their experts — **not** the matmuls. This is why memory bandwidth and interconnect dominate your hardware choice, and why the entire software stack is built around *keeping the pipeline full* rather than *going faster per FLOP*.

### 1.2 The memory arithmetic

Weights alone, at 1T parameters:

- **BF16** (2 bytes/param) → **2 TB** (≈1.82 TiB)
- **Native FP8** (1 byte/param) → **1 TB** (≈0.91 TiB)
- **FP4 experts** (~0.5 byte/param) → DeepSeek-V4-Pro's 1.6T lands at ~862 GB on disk

Then add, on top of weights:

- **KV cache** — scales with concurrent sequences × context length. MLA makes this small (Section 7.3), which is the only reason 1M-token contexts are feasible.
- **Activations + runtime/workspace overhead** — budget **+30–50%** over raw weights.

**Rule of thumb: provision ~1.5× the weight size in aggregate HBM.** So plan ~3 TB HBM for a BF16 1T model, ~1.5 TB for FP8.

### 1.3 Two phases, two different machines

Prefill and decode have opposite bottlenecks, which is *why you disaggregate them* (Section 8):

| Phase | What it does | Bottleneck | Wants |
|---|---|---|---|
| **Prefill** | Process the whole prompt in parallel | **Compute-bound** (large batched GEMMs) | High FLOPs, big chunked batches, "normal" all-to-all |
| **Decode** | Generate one token at a time | **Memory-bandwidth + all-to-all bound** | High HBM bandwidth, huge aggregate batch, "low-latency" all-to-all, CUDA graphs |

Running both on the same GPUs means a long prefill stalls latency-sensitive decode, and the two phases fight over the same all-to-all communication group. Separating them lets each run its optimal configuration.

---

## 2. The precision decision — what "no quantization" actually means

This is the most misunderstood knob at this scale, and you explicitly asked for no quantization, so it deserves its own section.

**Three different things get called "quantization":**

1. **Post-training quantization (PTQ)** — taking a model trained in BF16/FP8 and *downcasting* it (GPTQ, AWQ, INT4, INT8, or FP8-on-a-BF16-model) to save memory. This can degrade quality. **This is what you want to avoid, and you're right to.**
2. **Native low precision** — the model was *trained* in FP8 and its official weights *ship* in FP8. **Kimi K2 and the entire DeepSeek V3/V4 line are native FP8.** Serving them in FP8 is not quantization — it's the reference precision the authors validated. There is **no quality loss** because there is no downcast.
3. **Upcasting** — taking a native-FP8 model and running it in BF16. This is the *opposite* of quantization, and it is **wasteful**: it doubles weight memory (1 TB → 2 TB), roughly halves MoE-GEMM throughput, and recovers **zero** quality, because the information was never there at BF16 resolution — the weights were trained in FP8.

**The correct "no quantization" policy: serve at the model's native precision, and never downcast below it.**

- **FP8-native models** (Kimi K2, DeepSeek V3/V4) → serve **FP8**. This is lossless and is what every published deployment does.
- **BF16-native models** (Qwen3-235B/480B, Llama) → serve **BF16**.
- **DeepSeek-V4-Pro** ships **FP4 for the expert weights** — for a strict no-quantization stance that model is a poor fit unless you accept its shipped FP4 experts as "native." Kimi K2 (native FP8) or a BF16-native Qwen3 MoE is the cleaner choice if precision purity is the goal.

**Important nuance from real deployments:** even the configurations people *call* "BF16" keep **BF16 only for the numerically sensitive MLA / combine path** and use **FP8 for the big grouped MoE GEMMs** — because the MoE GEMM is where the FLOPs are and native FP8 there is effectively lossless with DeepGEMM's two-level accumulation. **Nobody at this scale runs the MoE GEMMs in pure BF16**, because it buys no quality and costs ~1.8–1.9× throughput plus double the memory.

**If you nonetheless mandate literal all-BF16 (MoE GEMMs included):** expect roughly **~1.8–1.9× lower GEMM throughput** than the FP8 path and **2× the weight memory** (multiply every GPU count below by ~2). Pick a **BF16-native model** (Qwen3) so you aren't upcasting for nothing. This guide sizes both the native-FP8 path (recommended) and flags the all-BF16 multiplier where it matters.

---

## 3. Hardware selection (rented)

### 3.1 Rank your constraints in this order

1. **HBM capacity** — the hard wall. You cannot start until the weights + KV + overhead fit.
2. **Memory bandwidth (TB/s)** — sets decode throughput (decode is bandwidth-bound).
3. **Interconnect** — NVLink domain size + inter-node fabric set how painful your required sharding is (all-to-all for EP, all-reduce for TP).
4. **BF16/FP8 FLOPs** — matters mostly for prefill; rarely your binding constraint.

### 3.2 GPU comparison (verified against vendor datasheets, mid-2026)

| GPU | HBM/GPU | Mem BW | BF16 dense TFLOPs¹ | Intra-node link | NVLink domain | GPUs to hold 2 TB (BF16)² | GPUs to hold 1 TB (FP8)² |
|---|---|---|---|---|---|---|---|
| **H100 SXM** | 80 GB | 3.35 TB/s | ~990 | NVLink4 900 GB/s | 8 | 25 → ~38 w/ overhead | 13 → ~19 |
| **H200 SXM** | 141 GB | 4.8 TB/s | ~990 | NVLink4 900 GB/s | 8 | 15 → ~23 | 8 → ~12 |
| **B200** | 192 GB | 8.0 TB/s | ~2,250 | NVLink5 1.8 TB/s | 8 (72 in NVL) | 11 → ~16 (2 nodes) | 6 → ~8 |
| **GB200** (in NVL72) | 192 GB | 8.0 TB/s | ~2,250 | NVLink5 1.8 TB/s | **72** (13.4 TB pool) | 11 (fits in 1 rack) | 6 |
| **B300 / GB300** | **288 GB** | 8.0 TB/s | ~3,750 | NVLink5 1.8 TB/s | 8 / 72 | 7 → ~11 | 4 → ~6 |
| **AMD MI300X** | 192 GB | 5.3 TB/s | 1,307 | IF 896 GB/s | 8 | 11 → ~16 | 6 → ~8 |
| **AMD MI325X** | **256 GB** | 6.0 TB/s | ~1,307 | IF 896 GB/s | 8 | 8 → ~12 | 4 → ~6 |
| **AMD MI355X** | 288 GB | 8.0 TB/s | ~2,510 | IF ~1,075 GB/s | 8 | 7 → ~11 | 4 → ~6 |

¹ NVIDIA datasheets quote the **sparse** (2:1) number; **dense BF16 is half** — this table already halves it. ² Weights only at 100% capacity → then with ~1.5× overhead for KV + activations.

**Reading the table:** capacity picks your minimum GPU count; then within candidates that fit, **bandwidth picks your decode throughput** (B200/MI355X at 8 TB/s ≫ H100 at 3.35). The NVLink domain column is why GB200 NVL72 is special — 72 GPUs in one coherent 13.4 TB NVLink pool means you can hold the whole model and tensor-parallelize decode without ever crossing the slow inter-node fabric.

### 3.3 Interconnect: why the fabric decides your topology

- **Tensor Parallelism (TP)** does an all-reduce every layer → needs the fattest link → keep it **inside** the NVLink domain (≤8 on HGX, ≤72 on NVL72).
- **Expert Parallelism (EP)** does an all-to-all dispatch/combine → tolerates crossing nodes over InfiniBand/RoCE, but the fabric speed caps your decode throughput. DeepEP needs only ~20 SMs to saturate a 400 Gb/s NIC, leaving the rest for compute.
- **NVL72 advantage:** the whole 1T model lives in one NVLink domain, so the expert all-to-all runs at 1.8 TB/s NVLink instead of 400–800 Gb/s InfiniBand. That's the single biggest decode-throughput lever available, and why Blackwell rack-scale posts the highest per-GPU numbers.

### 3.4 Rental pricing (mid-2026, neocloud vs hyperscaler)

| GPU | Cheapest on-demand (neocloud) | Reserved | Hyperscaler on-demand | 8-GPU node $/hr |
|---|---|---|---|---|
| **H100 SXM** | **$1.99–2.50** (RunPod, UpCloud, Hyperstack) | ~$1.80–3.22 | AWS $6.88, Azure $6.98 | ~$16–25 |
| **H200 SXM** | **$2.30–3.44** (FluidStack $2.30, Theta $2.49) | ~$3.44 | AWS/Azure ~$10.60+ | ~$27–34 |
| **B200** | **$3.40–5.89** (Lyceum/Packet ~$3, RunPod $5.89) | **$2.25** (36-mo) | AWS $14.24, GCP $16.11 | Lambda ~$53 |
| **GB200** (NVL72) | **$10.50** (CoreWeave slice) | custom | Azure $27.04, OCI $16.00 | rack-scale, custom |
| **B300 / GB300** | **~$3.13–3.80** | — | limited | DGX B300 |
| **AMD MI300X** | **$1.71–1.99** (TensorWave, DO, Hot Aisle) | **$1.66** (RunPod 12-mo) | OCI $6.00, Azure $7.86 | ~$16–28 |
| **AMD MI325X** | **$2.00–2.25** (DO, TensorWave, Vultr) | — | limited | — |
| **AMD MI355X** | **$2.95** (TensorWave) | **$2.29** (Vultr 36-mo) | OCI $8.60 | ~$18–21 |

Notes: **neoclouds beat hyperscalers 2–4× on identical silicon** — never rent big blocks from AWS/GCP/Azure/OCI for cost-per-token. B200 supply is tight (prices rose ~20% YoY) but promo neocloud rates ($3–3.40) show a wide spread — shop around. GB200 NVL72 and large B200 blocks are mostly **custom-contract**, not public on-demand; availability, not list price, is the real constraint on the newest parts. Prices are live-scraped and move daily — re-verify at provision time.

### 3.5 Hardware recommendation for throughput/$

1. **Default: B200 on a neocloud.** Best bandwidth (8 TB/s) and FLOPs per dollar. 8× B200 = 1.5 TB HBM/node; **2 nodes hold an FP8 1T model with huge KV headroom** (or a BF16 model tightly), linked by NVLink5 intra-node + InfiniBand across the two nodes.
2. **Cheapest to hold the weights: MI300X.** ~$2/GPU-hr on-demand, $1.66 reserved; 11× × 192 GB ≈ 2.1 TB. Best raw $/GB-HBM — *if* your SGLang/vLLM ROCm path is validated. Budget stack-maturity risk.
3. **Highest absolute throughput, simplest sharding: GB200 NVL72.** Whole model in one NVLink domain. Priciest per GPU-hr and allocation-gated, but the best decode numbers and least sharding pain.
4. **Legacy-cheap: H200.** At ~$2.30–2.50/GPU-hr it's the value pick on Hopper; ~16 GPUs (2 nodes) hold an FP8 model comfortably, and it's the most-proven, best-documented target (SGLang's reference reproductions run here).

---

## 4. Software stack — pick the inference engine

| Engine | Best for | Large-EP | DP-attention | P/D disagg | Verdict |
|---|---|---|---|---|---|
| **SGLang** | Hopper, BF16/FP8, DeepSeek/Kimi scale | ✅ DeepEP | ✅ (v0.4+) | ✅ Mooncake/NIXL | **Most proven** — published, reproducible recipe matching DeepSeek's own numbers; native Kimi K2 support |
| **vLLM (Wide-EP, V1)** | Broadest ecosystem | ✅ `--enable-expert-parallel` | ✅ | ✅ llm-d / Dynamo / Ray | **At parity**, widest deployment tooling |
| **TensorRT-LLM** | Blackwell, max per-GPU | ✅ (up to 6.17× w/ large-EP+EPLB+MTP) | ✅ | ✅ | **Best on B200/GB200**, deepest low-level tuning, steepest learning curve |
| **DeepSpeed** | — | — | — | — | **Not competitive** for this workload today |

**Recommendation:** **SGLang** or **vLLM Wide-EP** on Hopper (H100/H200) for a BF16/native-FP8 deployment — SGLang if you want the exact published DeepSeek/Kimi recipe, vLLM if you want the richest K8s/Dynamo/Ray deployment ecosystem. **TensorRT-LLM** on Blackwell when you want the last drop of per-GPU throughput (accepting that its headline numbers lean on FP4/FP8, which is native for these models anyway).

---

## 5. Parallelism strategy — the heart of throughput

### 5.1 The four axes

- **TP (Tensor Parallel):** split each layer's matmuls across GPUs; all-reduce every layer. High bandwidth → **keep inside NVLink**.
- **PP (Pipeline Parallel):** split layers across GPUs; cheap comms but introduces pipeline **bubbles**. Used sparingly here.
- **EP (Expert Parallel):** distribute the MoE experts across GPUs; all-to-all dispatch/combine. **The primary axis for MoE.** Larger EP → fewer experts per GPU → more VRAM freed for KV cache → bigger batch → higher throughput.
- **DP (Data Parallel):** replicate a component across GPUs, each handling different sequences.

### 5.2 The winning recipe: Wide-EP + DP-attention

The configuration every serious stack converges on:

- **Experts → EP** (spread the 256–384 experts across the whole fleet).
- **Attention → DP, not TP** (**DP-attention**). Because MLA compresses KV to a tiny latent, TP would *duplicate* that latent across ranks and waste memory; DP-attention gives each rank its own small KV, so you push batch size — and therefore throughput — far higher. **This is the single highest-leverage memory decision for MLA models.**
- **Dense FFN / LM-head → DP** as well (e.g. SGLang `--moe-dense-tp-size=1`), which cuts FFN communication ~50% (two all-reduces → one reduce-scatter + one all-gather).

Why so much EP: MoE is extremely sparse (8 of 256 experts/layer), so each expert needs a **large aggregate batch** to make its GEMM compute-efficient. Wide EP is what assembles that aggregate batch across many GPUs.

### 5.3 Reference topology (DeepSeek's own production, H800)

Precision consistent with training — **FP8 for matmuls/dispatch, BF16 for the sensitive MLA/combine path**:

- **Prefill unit:** Routed-Expert **EP32**, MLA/Shared-Expert **DP32**, 4 nodes, **+32 redundant experts** (9 routed + 1 shared per GPU).
- **Decode unit:** Routed-Expert **EP144**, MLA/Shared **DP144**, 18 nodes, **+32 redundant experts** (2 routed + 1 shared per GPU).
- Measured: **~73.7k input tok/s/node** (prefill, incl. cache hits) and **~14.8k output tok/s/node** (decode) per 8-GPU H800 node (~1.85k tok/s/GPU decode). Average user-visible output 20–22 tok/s; 56.3% of input tokens hit an on-disk KV cache.

### 5.4 Mapping to your rented hardware

- **2× 8×B200 (16 GPUs):** EP16 across both nodes for experts, DP-attention per rank. NVLink5 intra-node, InfiniBand inter-node. Good starting throughput deployment.
- **16–24× H200:** EP16–EP24 + DP-attention; the well-trodden SGLang path.
- **GB200 NVL72:** EP up to 72 in a single NVLink domain — the all-to-all runs at NVLink speed. Highest per-GPU decode throughput.
- **Scale note:** per-GPU throughput and $/token *improve with deployment size* (bigger aggregate batch, fuller experts). If you're optimizing cost-per-token, run **fewer, larger** deployments rather than many small ones. An 8–16-GPU deployment will not reach the $0.20/1M headline number — that needs DeepSeek-scale EP.

---

## 6. Kernel optimizations

| Kernel | Role | Precision notes |
|---|---|---|
| **FlashAttention-3** | Attention on Hopper; warp-specialized, TMA async | **BF16/FP16 ~740 TFLOPS (~75% H100 util), 1.5–2× over FA-2**. Your BF16 attention baseline. |
| **FlashMLA** | DeepSeek's MLA decode kernel (Hopper) | **Supports BF16/FP16**; paged KV, block 64; **~3000 GB/s (83% of BW), up to 660 TFLOPS** on H800; ~1450 TFLOPS on B200. The go-to non-quantized MLA decode kernel. |
| **FlashInfer** | Kernel layer under vLLM & SGLang; JIT attention/GEMM/MoE | 29–69% ITL reduction vs compiler baselines; FA-2/FA-3, cuDNN, CUTLASS, TRT-LLM backends. |
| **DeepGEMM** | Grouped MoE GEMM | **FP8-in, BF16-out** (`gemm_fp8_fp8_bf16_nt`); ~1350–1550 FP8 TFLOPS. Two layouts: **contiguous** (prefill), **masked** (decode, CUDA-graph-friendly). Default-on in vLLM. |
| **CUTLASS / Triton grouped GEMM** | Generic MoE-GEMM fallbacks | Underlie FlashInfer's Blackwell GEMMs; SGLang bridges DeepEP dispatch → DeepGEMM via a custom Triton permutation kernel. |
| **PagedAttention** | Block-paged KV cache | Foundational; eliminates KV fragmentation. Default in vLLM/TRT-LLM. |
| **Blackwell kernels** | CuTe-DSL GEMM (tcgen05.mma / 2CTA), CUTLASS NVFP4, Flash-Attention-CuTe | Includes **BF16 KV-cache prefill** paths. |

**CUDA graphs** are the cross-cutting win for decode: they eliminate per-kernel launch overhead. DeepEP's low-latency dispatch and DeepGEMM's masked-layout GEMM are specifically built to be CUDA-graph-compatible so the whole decode step captures cleanly. Turn them on (`FULL_AND_PIECEWISE` in vLLM; piecewise CUDA-graph + `torch.compile` in TRT-LLM).

---

## 7. Data structures & memory management

### 7.1 PagedAttention (block paging)
KV cache is stored in fixed-size blocks with a virtual→physical mapping (like OS paging), eliminating fragmentation and enabling block sharing. This is the baseline KV layout — it's what lets you pack many concurrent sequences into HBM without wasting space to padding.

### 7.2 RadixAttention / prefix caching
SGLang keeps KV activations in a **radix tree with LRU eviction**. Any request sharing a prefix with a cached one (system prompts, few-shot preambles, multi-turn history, agent scaffolds) **reuses the cached KV instead of recomputing it** → lower TTFT and, because you're not re-spending prefill compute, higher throughput. Reported up to **5× throughput**; workloads with 60%+ prefix overlap see 75–95% cache-hit rates. This is close to free money for any workload with repeated prefixes.

### 7.3 MLA — why this scale is even feasible
Multi-head Latent Attention compresses K and V **jointly into one low-rank latent per token** and reconstructs per-head K/V on the fly. DeepSeek-V3 caches just **512 + 64 (decoupled RoPE) = 576 elements/token ≈ 1.1 KB/token at BF16**, versus thousands of elements for MHA — a **~93% KV reduction with no quality loss** (unlike GQA, MLA matches or beats full MHA quality). Smaller KV means longer context in the same VRAM *and* faster decode (less HBM traffic per step), and it's what makes DP-attention affordable. This is **architectural, not quantization** — no precision is lost.

### 7.4 Hierarchical / cross-fleet KV cache
Push KV reuse beyond a single node:
- **DeepSeek** backs prefix caching with an **on-disk** KV store (their 3FS distributed filesystem) and hits **56.3% input-token cache hits** in production.
- **Mooncake** (Kimi/Moonshot) pools CPU DRAM, SSD, and RDMA into a disaggregated KVCache with a global "Conductor" scheduler — lifted Kimi's capacity **+107% (H800)**.
- **LMCache** / **NVIDIA Dynamo KVBM** provide hierarchical KV offload (GPU → CPU → SSD) embeddable in vLLM/TRT-LLM.

### 7.5 Expert placement (weight layout)
EPLB (Section 8) doesn't just balance load — it decides the **logical→physical expert placement**, duplicating hot experts and co-locating cold ones, and enables non-power-of-two EP sizes (e.g. 12, 72). Treat expert placement as a first-class data-structure decision, refreshed from live routing statistics.

---

## 8. Reducing GPU idle time (your central concern)

### 8.1 Why GPUs sit idle in large-EP MoE serving

| Cause | What happens |
|---|---|
| **(a) Blocking all-to-all** | MoE dispatch/combine is a collective; with small decode compute, comms dominate and GPUs busy-wait. |
| **(b) Expert imbalance** | Hot experts overload some ranks while others idle; the whole EP group waits on the slowest rank. |
| **(c) Prefill stalls decode** | One compute-heavy prefill request freezes the EP group's forward pass. |
| **(d) CPU / Python (GIL)** | At large aggregate batch, the host can't schedule kernels fast enough to keep the GPU fed. |
| **(e) Kernel-launch overhead** | Thousands of tiny decode kernels, each with launch latency. |

Each technique below targets a specific cause.

### 8.2 The techniques (with measured impact)

1. **Continuous / in-flight batching + chunked prefill** *(table stakes; a, c)* — never wait to fill a batch; slice long prefills into chunks so they interleave with decode instead of blocking it.

2. **Prefill/Decode disaggregation** *(a, c)* — separate GPU pools so compute-bound prefill never interrupts latency-bound decode, and each phase uses its **own** DeepEP dispatch mode (which can't coexist in one comm group). Implementations: SGLang PD (non-blocking RDMA transfer via **Mooncake** / **NIXL**), vLLM (**llm-d / NVIDIA Dynamo / Ray**). SGLang measured **3.8× prefill / 4.8× decode** from disaggregation on GB200.

3. **Dual-microbatch / "two-batch" overlap** *(a)* — split a batch into two microbatches so one's all-to-all comms hide behind the other's compute. SGLang **TBO: +27–35% decode, +40.5% prefill** (also enables 16k-token/device batches that otherwise OOM); vLLM **DBO** (`--enable-dbo`). On very-high-bandwidth NVLink (GB200), switch to *finer-grained* overlap (combine overlapped with down-GEMM + shared experts).

4. **EPLB — expert load balancing** *(b; the single biggest lever)* — add **redundant/replicated experts** (e.g. 256 → 288, ~+12.5%) and compute a placement that duplicates hot experts and co-locates cold ones to minimize the max per-GPU load. **Measured: 1.49× prefill / 2.54× decode.** Use **online/dynamic** mode (sliding-window routing stats → periodic live weight-shuffle, no restart). DeepSeek runs three separate balancers (prefill, decode, expert-parallel). Budget the redundant-expert memory and expect real systems pain — NUMA placement, 512 MB huge pages to stop TLB thrashing, and a documented cudaMemcpyAsync-vs-cudaGraphLaunch deadlock to avoid.

5. **DeepEP all-to-all** *(a)* — purpose-built MoE comms: **normal kernels** (high-throughput NVLink↔RDMA forwarding) for prefill, **low-latency kernels** (pure-RDMA, CUDA-graph-compatible) for decode. Saturates a 400 Gb/s NIC with only ~20 SMs, leaving the rest for compute. FP8 (NVFP4 on Blackwell) dispatch halves network traffic.

6. **CUDA graphs** *(e)* — capture the whole decode step to remove launch overhead; relies on graph-compatible DeepEP low-latency + DeepGEMM masked kernels.

7. **Zero-overhead / overlap scheduler** *(d)* — overlap CPU scheduling with GPU compute. SGLang's overlap scheduler alone: **+20%** (51 → 60.4 tok/s/rank).

8. **`stream_interval` / async scheduling** *(d)* — at large aggregate batch the Python/GIL host path becomes the bottleneck; emit responses every N iterations and cut C++/Python IPC (TRT-LLM's documented fix).

9. **MTP speculative decoding** *(throughput multiplier)* — Multi-Token Prediction: a lightweight draft head proposes n tokens, the full model verifies them in one parallel pass, so **quality is identical**. SGLang: **+14% to +60%** depending on concurrency (largest at low/moderate load, smaller once kernels already run large batches). Start `draft_token_num=2`, raise to 4 only with headroom, watch acceptance length in logs.

10. **Prefix-aware routing** *(cross-node reuse)* — NVIDIA Dynamo's Smart Router hashes requests into a global radix tree to route each to the GPU that already holds the matching prefix, minimizing recompute across the fleet.

### 8.3 The non-negotiable combination

**PD-disaggregation + Wide-EP (EP experts, DP attention/dense/LM-head) + EPLB + dual-microbatch overlap + DeepEP + DeepGEMM + CUDA graphs (decode).** Omitting any one leaves a large idle bubble somewhere in the pipeline. Add MTP and prefix caching on top as throughput multipliers.

### 8.4 Diagnose what you're bound by (roofline)

- **Decode slow, GPUs <100% busy, network hot** → all-to-all bound → more overlap, faster fabric, or bigger NVLink domain (GB200).
- **Decode slow, GPUs busy, bandwidth saturated** → memory-bandwidth bound → higher-BW GPU (B200/MI355X) or native FP8 to move fewer bytes.
- **Uneven per-GPU utilization** → expert imbalance → EPLB + more redundant experts.
- **GPUs starved, CPU pegged** → host/GIL bound → async scheduler, `stream_interval`.
- **Targets:** FlashMLA hits ~83% of theoretical bandwidth (MBU) and ~91% peak FLOPs on H800; FA-3 ~75% util in BF16. If you're far below these, you have idle to reclaim.

---

## 9. Reference deployment blueprints (throughput-optimized, rented, native precision)

All numbers are from published reproductions; treat as order-of-magnitude planning figures, not guarantees. "Native precision" = FP8 MoE GEMM + BF16 MLA/combine unless noted.

### Blueprint A — Hopper value (recommended starting point)
- **Model:** DeepSeek-V3 (671B) or Kimi K2 (1T), native FP8
- **Hardware:** 12× 8×**H100** (96 GPUs) *or* 16× 8×**H200** (128 GPUs) on a neocloud
- **Engine:** SGLang, PD-disaggregated, Wide-EP + DP-attention + EPLB + TBO + DeepEP + DeepGEMM
- **Measured:** DeepSeek-V3 on 96×H100 → **52.3k input / 22.3k output tok/s per node**, **~$0.20 / 1M output tokens**. Kimi K2 on 128×H200 → **224k prefill / 288k decode tok/s** fleet-wide, **~$0.21 / 1M output tokens**.

### Blueprint B — Blackwell max per-GPU
- **Model:** DeepSeek-R1 / V4, native FP8 (FP4 experts where shipped)
- **Hardware:** 2× 8×**B200** (neocloud) up to a **GB200 NVL72** rack
- **Engine:** TensorRT-LLM (or SGLang) with large-EP + EPLB + MTP
- **Measured:** SGLang on GB200 NVL72 → **18,471 input / 9,087 output tok/s per GPU** on the BF16-attention + FP8-MoE path; up to **26,156 / 13,386** with NVFP4 MoE (3.8× / 4.8× vs H100). TensorRT-LLM large-EP: **up to 6.17× per-GPU** vs small-EP baselines.

### Blueprint C — Cheapest HBM (AMD)
- **Model:** DeepSeek / Kimi, native FP8
- **Hardware:** 16× **MI300X** (~$2/GPU-hr) or 12× **MI325X** (256 GB each)
- **Engine:** vLLM or SGLang on ROCm
- **Trade-off:** lowest $/GB-HBM to hold the weights; validate the ROCm kernel path (FlashMLA/DeepGEMM equivalents) before committing — stack maturity is the risk, not the hardware.

---

## 10. Cost model

**Cost per 1M output tokens = (GPU-hr rate × number of GPUs) ÷ (output tokens per hour).**

Worked example — Blueprint A on 96×H100 at $2.00/GPU-hr on-demand:
- GPU cost/hr = 96 × $2.00 = **$192/hr**
- Output = 22.3k tok/s/node × 12 nodes = 267.6k tok/s → × 3600 = **963M output tok/hr**
- **$192 / 963 = ~$0.20 per 1M output tokens** ✓ (matches the published figure)

Three levers move this number:
1. **Sustained utilization / aggregate batch.** The $0.20 assumes you keep the fleet busy at large batch. At 30% utilization your effective cost triples. **Cost-per-token is an *occupancy* game** — a smaller, always-full deployment beats a larger, half-idle one.
2. **GPU rate.** Reserved/committed and spot can undercut on-demand 20–70% (H100 spot floors near $0.61); if your load is steady, commit.
3. **Deployment scale.** Per-GPU throughput rises with EP size, so consolidating into fewer large deployments lowers $/token — the opposite of what latency-optimized serving would do.

**Rent-vs-buy sanity check (you chose rent, so briefly):** an 8×H200 node runs ~$27–34/hr rented; buying the equivalent HGX is ~$315k. Break-even is roughly **12–18 months of 24/7 use** *before* power, cooling, colocation, networking, and ops staff — which for GPUs depreciating this fast, and at neocloud spot/reserved rates, usually favors renting unless you run at very high sustained utilization for years. Renting also sidesteps the availability problem for the newest parts.

---

## 11. Benchmarking & tuning checklist

**Metrics to track:** TTFT (time-to-first-token), TPOT/ITL (inter-token latency), **throughput (tok/s/GPU)** ← your north star, **goodput** (throughput within an SLO), and **MBU/MFU** (bandwidth/FLOP utilization — how close to the roofline).

**Load-test** with realistic input/output length distributions and concurrency sweeps; the $/token optimum is at high concurrency, so test there, not at batch-size-1.

**Tuning order (highest leverage first):**
1. DP-attention on (not TP-attention).
2. Enable PD-disaggregation.
3. Turn on EPLB, add redundant experts, enable online rebalancing.
4. Enable dual-microbatch overlap (TBO/DBO).
5. Enable CUDA graphs for decode.
6. Grow aggregate batch / EP size until bandwidth or fabric saturates.
7. Add MTP; tune `draft_token_num` to acceptance length.
8. Turn on prefix caching / radix cache; add hierarchical KV if prefixes repeat across the fleet.
9. Fix the host path (`stream_interval`, async scheduler) once GPUs show starvation.

---

## 12. Pitfalls & gotchas

- **Datasheet TFLOPs are sparse (2:1).** Halve them for dense BF16. (Table in §3.2 already did.)
- **"1T" is often nominal.** Kimi K2 is a true 1T; DeepSeek V3 is 671B (the *architecture*, not a 1T model); only DeepSeek-V4-Pro (1.6T) and Kimi K2 are open and actually ≥1T. Qwen3-Max (1T+) and Llama 4 Behemoth (~2T) are **not open**.
- **FP4 on-disk param confusion.** DeepSeek-V4-Pro shows ~862B "params" on disk because experts ship in FP4 — the logical count is 1.6T. Don't size memory off the on-disk number.
- **MLA needs DP-attention, not TP-attention.** TP duplicates the latent KV and wastes your biggest memory advantage.
- **Upcasting FP8→BF16 buys nothing.** Double memory, ~half GEMM throughput, zero quality gain. Serve native.
- **MTP gains shrink at high concurrency.** It's a low/mid-load win; don't expect the +60% in a batch-saturated deployment.
- **EPLB has real systems pain.** NUMA placement, 512 MB huge pages, and a specific online-rebalance deadlock. Budget engineering time.
- **Hyperscalers cost 2–4×.** Use neoclouds for cost-per-token.
- **Newest-GPU availability, not price, is the constraint.** GB200/B300 per-GPU-hr rates assume you can actually get allocation.
- **The host/Python path can bottleneck the GPU** at large batch. Watch CPU; use async scheduling.
- **Cost-per-token is an occupancy game.** The headline $/token numbers require large, full deployments — not many small idle ones.

---

## 13. Sources

**Models & architecture**
- Kimi K2 technical report — https://arxiv.org/abs/2507.20534 ; K2.6 release — https://siliconangle.com/2026/04/20/moonshot-ai-releases-kimi-k2-6-model-1t-parameters-attention-optimizations/
- DeepSeek-V3 technical report (MoE, MLA, FP8, MTP) — https://arxiv.org/abs/2412.19437
- DeepSeek-V3.2 (sparse attention) — https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html ; DeepSeek-V4-Pro card — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V2 (MLA / KV reduction) — https://arxiv.org/abs/2405.04434
- GLM-5 analysis — https://www.digitalapplied.com/blog/zhipu-ai-glm-5-release-744b-moe-model-analysis
- Qwen3-Coder-480B — https://arxiv.org/abs/2505.09388

**Hardware & pricing** (live tables, re-verify at provision time)
- getdeploying GPU pages: H100 https://getdeploying.com/gpus/nvidia-h100 · H200 https://getdeploying.com/gpus/nvidia-h200 · B200 https://getdeploying.com/gpus/nvidia-b200 · GB200 https://getdeploying.com/gpus/nvidia-gb200 · MI300X https://getdeploying.com/gpus/amd-mi300x · MI355X https://getdeploying.com/gpus/amd-mi355x
- GB200 NVL72 — https://www.nvidia.com/en-us/data-center/gb200-nvl72/ ; Blackwell Ultra (B300) — https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- AMD MI350 series — https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html
- GB200 NVL72 cost — https://newsletter.semianalysis.com/p/h100-vs-gb200-nvl72-training-benchmarks

**Serving stack, kernels, idle-time**
- DeepSeek inference system overview (production topology, throughput) — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- DeepSeek repos: FlashMLA https://github.com/deepseek-ai/FlashMLA · DeepGEMM https://github.com/deepseek-ai/DeepGEMM · DeepEP https://github.com/deepseek-ai/DeepEP · EPLB https://github.com/deepseek-ai/EPLB · profile-data https://github.com/deepseek-ai/profile-data
- SGLang large-scale EP (DeepSeek reproduction) — https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ ; Kimi K2 EP — https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep/ ; MTP — https://www.lmsys.org/blog/2025-07-17-mtp/ ; GB200 — https://www.lmsys.org/blog/2025-09-25-gb200-part1/
- vLLM large-scale serving (Wide-EP) — https://blog.vllm.ai/2025/12/17/large-scale-serving.html ; GB200 — https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html
- TensorRT-LLM large-EP — https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html
- FlashAttention-3 — https://arxiv.org/abs/2407.08608 ; FlashInfer — https://arxiv.org/abs/2501.01005
- RadixAttention / SGLang — https://www.lmsys.org/blog/2024-01-17-sglang/
- Mooncake (KV-centric disaggregation) — https://arxiv.org/abs/2407.00079 ; NVIDIA Dynamo — https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- DistServe (P/D disaggregation) — https://hao-ai-lab.github.io/blogs/distserve-retro/

*Compiled mid-2026. Pricing and model availability change frequently — re-verify the live pricing tables and model cards before committing budget.*
