# Secant: Weight-Space Programming & Subspace-Locked LoRA

**Secant** is a standalone Rust workspace (`secant-core`, `secant-cli`, `secant-ffi`) and Python library (`secant.nn.ProjectThenHealLinear`, `secant.capture`) for **deterministic Weight-Space Programming (WSP)**, **in-place quantized block surgery (`Q8_0`)**, and **Subspace-Locked LoRA Healing (`Project-Then-Heal`)** on `.safetensors` and `.gguf` (v2/v3) checkpoints.

Motivated by MIT CSAIL's [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, and Khattab, Nov 2026) [@rikiyat2026wsp] alongside representation-engineering and null-space editing research [@zou2023representation; @zou2024circuitbreakers; @arditi2024refusal], Secant turns direct weight surgery into a **reproducible, hash-bracketed engineering discipline** exercised across **17 model checkpoints (`0.6B` to `72.7B`, 9 model families)**.

```{figure} assets/secant-pipeline.svg
:name: fig-secant-pipeline
:alt: Secant four-stage weight-space programming pipeline
:width: 100%

The four stages of Secant: (1) zero-eval `float64` weight reading and Boundary-3 payload sanitization, (2) two-pass Gram-Schmidt composition with Target-Shift Calibration, (3) zero-copy `mmap` and 34-byte `Q8_0` quantized block surgery, and (4) `ProjectThenHealLinear` subspace-locked LoRA training.
```

---

## What Secant Does (Ordered by Core Importance)

1. **Direct In-Place `mmap` & 34-Byte `Q8_0` Quantized Block Surgery on Disk (`secant patch`, `secant quant patch`, `secant quant witness`):**
   - Opens `.safetensors` and quantized `Q8_0` `.gguf` files via `mmap`, decodes only the touched 34-byte `Q8_0` blocks (2-byte `f16` scale + 32 `int8` quants), applies the surgical edit, and re-packs them in milliseconds while leaving `98%+` of the file (`240,640` untouched blocks) bit-for-bit identical. Also proves the $\Pi_{128}$ head-grid commutation boundary (`Q8_0` 32-weight blocks commute with 128-column heads, whereas 256-weight `Q4_K` superblocks straddle two heads).
2. **`Project-Then-Heal`: Why Gradient Projection Fails Under `AdamW` & How Forward-Graph Locking Fixes It (`secant.nn.ProjectThenHealLinear`, `secant heal`):**
   - Even if you project raw gradients out of $\mathrm{span}(D)$ before `AdamW.step()`, `AdamW` divides each coordinate by its running variance $\sqrt{\hat{v}_t} + \epsilon$, rotating updates right back into $\mathrm{span}(D)$ on **`100%` of steps (`108/108` runs)**. `ProjectThenHealLinear` locks the LoRA output factor inside the forward pass via $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$, bounding writer leakage to **`5.21e-18 – 9.87e-18`** in `float64` (`1.78e-10 – 8.67e-10` in `float32`).
3. **Target-Shift Calibration Across Model Scales (`secant compose --target-shift`):**
   - Copying a fixed Frobenius norm ratio ($\gamma = 0.30$) from a `0.8B` model to a `4B` model closes **`0/30`** `<think>` loops because wider layers have different activation scales. Solving analytically for the step size $\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$ closes **`21/30` (`70%`)** at $1.0\times$ and **`30/30` (`100%`)** (`23/30` exact culprit frame, $p = 1.86 \times 10^{-9}$) at $1.5\times$ on `Qwen3.5-4B`, and holds to within `<= 7.99e-15` across 15 architectures (`0.6B` to `32.8B`).
4. **Sparse Rank-1 Surgery & Multi-Direction Orthogonal Composition (`secant.capture`, `secant compose`):**
   - Instead of modifying every column of a weight matrix, Secant edits only the top 64 or 128 input columns (`< 2%` of the layer) selected by held-out $\text{SNR}_{\text{proj}}$ and orthonormalizes multiple behavioral directions (`D D^T = I_k`, `< 1e-15` residual) via two-pass `float64` Gram-Schmidt so sequential edits never overwrite one another.
5. **Downstream C/C++ Sanitizer Vulnerability Uplift (`2/30 -> 12/30`, `6x`):**
   - Rank-1 security-invariant surgery lifts `Qwen2.5-Coder-32B-Instruct` from `6.7%` (`2/30`) to **`40.0%` (`12/30`, `+10` crashes, $p = 0.00195$)** on our 30-case C/C++ sanitizer vulnerability suite without increasing reasoning token length (`125.9 -> 122.4` mean tokens).
6. **Zero-Eval Weight Reading & Malicious/Corrupted Payload Auditing (`secant inspect`, `secant read --vocab-tensor`, `secant sanitize`, `secant probe`):**
   - Memory-maps `.safetensors` and `.gguf` files directly on disk, computes deterministic non-FMA `float64` singular directions, projects them onto the vocabulary unembedding matrix without a GPU, and scans tensor payloads for `NaN`/`Inf`/subnormal floats, zero-norm rows, and out-of-bounds MoE router IDs that can crash C++ inference kernels.
7. **Methodology Guardrail: The `93.3%` Copy-the-Input Benchmark Control:**
   - Feeding the **raw user prompt itself** (with zero model inference) into a naive whole-output keyword matcher scores **`93.3%` (`28/30`)** on crash-triage benchmarks because every stack-trace function is named in the prompt. Strict post-`</think>` single-commitment scoring eliminates that mirage (`0/30` on raw prompt copy).

---

## Headline Empirical Results at a Glance

| Experiment / Regime | Models Tested | Primary Metric | Unconstrained / Baseline | Secant Result |
| :--- | :--- | :--- | :--- | :--- |
| **6-Arm Optimizer Ablation** (`648` runs) | Synthetic + Layer-16 `down_proj` (`0.8B`, `4B`, `7B`) | Relative subspace leakage $\|D \Delta W\|_F / \|\Delta W\|_F$ | Std LoRA: `1.5e-1`<br>Grad-Proj AdamW: `1.56% – 2.98%` | **`5.21e-18 – 9.87e-18`** (`float64` module) |
| **Regime A: Live LM Healing** | `Qwen3.5-0.8B`, `SmolLM2-1.7B` | Post-surgery causal-LM healing leakage $\max \lvert D \Delta W\rvert$ | `4.64e-3 – 1.05e-2` | **`5.21e-18 – 9.87e-18`** (`float64`) / **`<= 2.49e-10`** (merged `F32`) |
| **15-Model H100 Stage-1 Sweep** | `15` models (`0.6B` to `32.8B`, 7 families) | Target-shift error & `float32` locked LoRA leakage | Grad-Proj AdamW: `0.33% – 1.67%` | Shift error **`<= 7.99e-15`**; leakage **`1.78e-10 – 8.67e-10`** |
| **Regime B: `<think>` Calibration** | `Qwen3.5-4B` (`n = 30` crash traces) | Loop closure & exact culprit attribution | `0/30` (`0%`) at baseline & fixed-ratio | **`30/30` (`100%`)** closed; **`23/30` (`76.7%`)** culprit ($p = 1.86 \times 10^{-9}$) |
| **30-Case C/C++ Sanitizer Suite** | `Qwen2.5-Coder-32B-Instruct` (`n = 30`) | Paired sanitizer vulnerability detection | `2/30` (`6.7%`) | **`12/30` (`40.0%`, `+10` findings, $p = 0.00195$)** |

```{figure} assets/qwen4b-phase-transition.svg
:name: fig-qwen4b-phase-transition
:alt: Qwen3.5-4B alpha dose-response chart
:width: 100%

Target-Shift Calibration dose-response on `Qwen3.5-4B` (`n = 30` C++ sanitizer crash-triage benchmark, strict post-`</think>` scoring). Uncalibrated fixed-ratio ($\gamma = 0.30$) and equal-norm random controls remain at `0/30`; calibrated composition reaches `21/30` at $1.0\times$ (`cal738`) and `30/30` loop closure with `23/30` exact culprit frame at $1.5\times$ (`a15`).
```

---

## Documentation Map (Where to Go Next)

- **[Quickstart & Live CLI Walkthrough](quickstart.md)**: Start here. Build `secant` in 60 seconds and run all 5 core workflows (reading weight concepts into vocabulary words, composing calibrated plans, patching `Q8_0` `.gguf` blocks via `mmap`, and locking LoRA adapters).
- **[Mathematical Theory & Scale Calibration](theory-and-calibration.md)**: How contrastive capture (`SNR_proj`), two-pass `float64` Gram-Schmidt, and Target-Shift Calibration ($\beta = |\Delta_{\text{target}}| / \|h_S\|_2^2$) work.
- **[Quantized Block Surgery (`Q8_0` & `Q4_K`)](quantized-surgery.md)**: How 34-byte `Q8_0` in-place block editing works and why `Q8_0` commutes with 128-column head permutations ($\Pi_{128}$) while `Q4_K` superblocks straddle two heads.
- **[Project-Then-Heal Subspace-Locked LoRA](project-then-heal.md)**: Why `AdamW`'s diagonal preconditioner rotates projected gradients back into $\mathrm{span}(D)$ and how $B_{\text{locked}} = P_\perp(P_\perp(\tilde{B}))$ prevents it.
- **[CLI, Python & C-ABI Reference](cli-and-python-reference.md)**: Complete flag tables, exit codes, JSON receipt schemas, `secant.nn` / `secant.capture` Python APIs, and `secant.h` C-ABI signatures.
- **[Container Geometry & Payload Security](payload-security.md)**: Boundary-3 static tensor value scanning (`secant sanitize`) and deterministic test fixture generation (`secant probe`).
- **[Full Scientific Paper](PAPER.md)**: Self-contained technical paper with formal propositions, proofs, complete 17-checkpoint tables, and experimental caveats.
