<div align="center">

<img src="docs/assets/secant-logo.svg" alt="Secant Logo" width="84" height="84" />

# Secant

**Deterministic Weight-Space Programming, In-Place Quantized Block Surgery (`Q8_0`), Static Checkpoint Auditing, and Subspace-Locked LoRA (`Project-Then-Heal`)**

[![Documentation](https://img.shields.io/badge/Docs-professor--moody.github.io%2Fsecant-0A66C2?style=flat-square)](https://professor-moody.github.io/secant/)
[![Technical Paper](https://img.shields.io/badge/Paper-MyST_%2F_Quarto-8A2BE2?style=flat-square)](https://professor-moody.github.io/secant/paper)
[![Blog Series](https://img.shields.io/badge/Blog_Series-m00dy.sh-12704f?style=flat-square)](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?style=flat-square)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

**[Documentation](https://professor-moody.github.io/secant/)** · **[4-Part Blog Series](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)** · **[Quickstart](https://professor-moody.github.io/secant/quickstart)** · **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal](https://professor-moody.github.io/secant/project-then-heal)** · **[Payload Security](https://professor-moody.github.io/secant/payload-security)** · **[Technical Paper](https://professor-moody.github.io/secant/paper)**

<br/>

<img src="docs/assets/secant-pipeline.svg" alt="Secant Four-Stage Architecture" width="100%" />

</div>

---

## Overview

In [*"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang, and Khattab showed that an LLM agent (`gpt-6-astra`) could inspect activations and edit `0.016%` of `Qwen3.5-0.8B`'s weights to lift its `MATH-500` score from `7.9%` to `48.1%`, while calling for *"turning what currently seems like black magic into a rigorous, reproducible engineering discipline"* and *"auditing without evaluation."*

**Secant** (`v0.3.0`) is a Rust CLI/library (`secant-core`, `secant-cli`, `secant-ffi`) and PyTorch package (`secant.nn.ProjectThenHealLinear`, `secant.capture`) built to test and productionize weight-space programming across **`17` models (`0.8B` to `32.8B`)** and **`59,440` H100 benchmark evaluations** (plus a `17`-checkpoint Project-Then-Heal study reaching `72.7B` on single shards). Every operation runs directly on `.safetensors` and `.gguf` files via `mmap` and is bracketed by pre/post SHA-256 receipts.

---

## Core Capabilities

### 1. Pre-Load Value & Companion Sanitization (`secant sanitize`)
- Scans `.safetensors` (`F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`) and `.gguf` (`Q8_0`, `Q4_K`, and all 10 standard block formats) via `mmap` without deserializing weights into a Python runtime.
- Detects `NaN`/`Inf` injection (`Rule B3-1`, `B3-2A`), sub-floor norms and divisor scales (`Rule B3-2B`), out-of-range MoE routing and permutation indices (`Rule B3-3`), and cross-file companion JSON (`config.json`, `tokenizer_config.json`, `tokenizer.json`, `generation_config.json`) vs. tensor-shape mismatches (`Rule S-0`, including `added_tokens_decoder` bounds and disabled-modality `0` sentinels).
- Skips benign non-index auxiliary masks and counters (`BOOL`, `U8` causal masks/`*.bias`, scalar `I64` `num_batches_tracked`) cleanly while failing closed (exit `2`) on malformed companion JSON or unimplemented float/quantized formats. Verified **`0.00%` weight/block FPR** across **`110` public Hub checkpoints** (`32.5B` float elements + `201.8M` GGUF blocks).

### 2. Spectral & Reference-Diff Checkpoint Auditing (`secant audit-trojan`, `secant read --reference`)
- **Reference-free screening:** Iterative Marchenko-Pastur bulk peeling, rank-$k$ singular energy (`--rank-k`), native PEFT LoRA `B * A` sibling outlier Z-scores, and unembedding vocabulary projection (`E * u_1`) to surface concentrated weight tampering without running inference.
- **Exact reference-diff (`--reference`):** Streams $\Delta W = W - W_{\text{ref}}$ row-by-row in `float64` across SafeTensors and GGUF (`F32`/`F16`/`BF16`/`FP8`/`Q8_0`/`Q4_K`), isolating surgical rank-$k$ and column-sparse weight edits (`100%` precision/recall on Takehi et al.'s `p50c` checkpoint and sub-$g^*$ late-layer edits).

### 3. Calibrated `mmap` & Quantized `Q8_0` Weight Surgery (`secant analyze`, `compose`, `patch`, `quant patch`)
- **Activation-calibrated composition:** Orthogonalizes contrastive directions in `float64` (`R^-1` modified Gram-Schmidt) and sizes edits from the target model's own reference activations rather than transferring raw edit ratios across scales.
- **Hash-bracketed disk surgery:** Applies sub-tensor byte-range edits on `.safetensors` and 34-byte block-aligned edits on `Q8_0` `.gguf` files (copy-by-default or atomic `--in-place`), while refusing `Q4_K` superblock surgery where 256-weight blocks cross 128-column panel boundaries.

### 4. Subspace-Locked LoRA Healing (`ProjectThenHealLinear`, `secant heal`, `verify-subspace`)
- **Forward-pass subspace projection:** Locks LoRA factor $B$ orthogonal to the edited subspace ($P_\perp(P_\perp B)$ in `float64`) inside the forward pass so AdamW coordinate rescaling cannot re-introduce an ablated direction, holding post-edit leakage below **`10^-10` (`F32`)** and **`10^-17` (`float64`)**.

<p align="center">
  <img src="docs/assets/secant-patch.gif" alt="secant patch live terminal recording" width="49%" />
  <img src="docs/assets/secant-probe.gif" alt="secant probe live terminal recording" width="49%" />
</p>

---

## Quickstart

```sh
# 1. Build and install the Rust CLI
cargo install --path crates/secant-cli

# 2. Install the Python package (PyTorch ProjectThenHealLinear + contrastive capture)
pip install -e python/
```

```sh
# Audit a checkpoint on disk for Boundary-3 kernel hazards and spectral Trojans
secant sanitize --model model-Q4_K_M.gguf --receipt /tmp/sanitize.json --attempt-id quickstart --num-experts 8 --dry-run --json
secant audit-trojan --model model.safetensors --vocab-tensor model.embed_tokens.weight --top 16 --json

# Compose an activation-calibrated plan and patch a Q8_0 GGUF checkpoint in place
secant compose --profile profile.json --vectors vec/ --target-shift -4.0 --out plan.json --json
secant quant patch --base model-q8_0.gguf --plan plan.json --out model-patched-q8_0.gguf --json

# Lock a LoRA adapter away from the excised subspace and verify orthogonality
secant heal --lora lora_adapter.safetensors --vectors vec/ --out lora_locked.safetensors --base model.safetensors --merge-out model_healed.safetensors --json
secant verify-subspace --base model.safetensors --healed model_healed.safetensors --vectors vec/ --tensors "model.layers.15.mlp.down_proj.weight" --json
```

### Exit Codes (`0` / `1` / `2`)

Every `secant` subcommand follows a strict, fail-closed three-state exit-code contract suitable for CI and automated pre-load gates. Supported SafeTensors payload dtypes are `F64`, `F32`, `BF16`, `F16`, `F8_E4M3`, and `F8_E5M2` (plus bounded integer routing/index tensors and skipped `BOOL`/`U8` masks and scalar `I64` `num_batches_tracked` counters):

| Exit Code | Meaning | Typical Conditions |
| :---: | --- | --- |
| **`0`** | **Clean / Passed / Remediated** | Inspection or read completed; `sanitize` found zero violations (or `--out` published a clean remediated copy); `verify-patch` / `verify-subspace` / `quant witness` passed all byte-identity and orthogonality tolerances. |
| **`1`** | **Policy or Verification Gate Failed** | `sanitize` detected `Rule S-0` or `Boundary-3` (`B3-1..B3-3`) security findings that remain unremediated; `verify-patch` detected unexpected tensor/header mutation; `verify-subspace` exceeded the subspace leakage bound. |
| **`2`** | **Usage, I/O, or Fail-Closed `UNKNOWN`** | Invalid CLI flags; missing or unreadable file; existing `--out` / `--receipt` path refusal; malformed or non-object sibling companion JSON (`config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`); exotic or unimplemented float/quantized format (e.g., `F6_E2M3`/`F4` or unreviewed GGUF `Q8_K`/`IQ*`/`TQ*`); unbounded integer index tensor; or `pre_sha256 != post_sha256` hash-bracket violation. |

---

## Documentation, Blog Series & Paper

- **[Full Technical Paper](https://professor-moody.github.io/secant/paper)**: Complete statistical benchmarks, multi-seed random controls, dose-response curves, derivations, and SHA-256 reproducibility receipts.
- **[4-Part Blog Series ("From Sparks to Secant")](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)**: Plain-English walkthrough and interactive visualizers for Parts I to IV.
- **[Documentation Home](https://professor-moody.github.io/secant/)** · **[Quickstart & Live CLI Captures](https://professor-moody.github.io/secant/quickstart)** · **[CLI & Python Reference](https://professor-moody.github.io/secant/cli-and-python-reference)**
- **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery (`Q8_0` vs. `Q4_K`)](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal LoRA](https://professor-moody.github.io/secant/project-then-heal)** · **[Payload Security](https://professor-moody.github.io/secant/payload-security)**

## Citation

```bibtex
@software{keys2026secant,
  title   = {Secant: Deterministic Weight-Space Programming, In-Place Quantized Block Surgery, and Subspace-Locked LoRA},
  author  = {Keys, Nathan},
  year    = {2026},
  url     = {https://github.com/professor-moody/secant}
}
```
