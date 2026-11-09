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

## Key Findings

1. **Random Controls & `<think>` Loops:** Simply setting `enable_thinking=False` on unedited `Qwen3.5-0.8B` lifts `MATH-500` from `1.0%` to **`48.0%`** (close to Takehi et al.'s `48.1%` edited score, measured under a different protocol: `32,768` tokens, stochastic decoding), and on `Qwen3-8B` `GSM8K`, where nearly every completion hits the token cap, a same-size random edit gives the exact same boost as a targeted edit (`18% -> 30%`). However, when `<think>` is kept on in severe loops (`Qwen3.5-0.8B` `MATH-500`), our `0.016%` `B1` edit closes **`64%`** of `<think>` traces (**`24.0%`** accuracy at `4,096` tokens) while three independent orthogonal random direction seeds (`seed_1..seed_3`, `20261108..20261110`) of the exact same size close **`2.0%`, `6.0%`, and `1.0%`** (mean **`3.0% ± 2.6%`**, `0/100` closed by $\ge 2/3$ majority vote) and score **`0.0%`, `2.0%`, and `1.0%`** (mean **`1.0% ± 1.0%`**, Holm-adjusted $p \le 8.39 \times 10^{-5}$ across all 3 seeds).
2. **Target-Shift Calibration Across Scales:** Copying a `0.8B` edit ratio onto `Qwen3.5-4B` closes **`0 / 30`** reasoning loops, while sizing the edit from the `4B` model's own reference activations (`secant analyze` + `secant compose`) closes **`21 / 30`** at `1.0x` and **`30 / 30`** with the loop edit at `1.5x` (`19 / 30` strict root cause, $p = 1.86 \times 10^{-9}$ vs. base; two random controls `0 / 30`; `n = 30`, one run per arm). Pooled across `14` models, unhealed edits move public benchmarks by `2.1` points or less (single 100-item cells `-12` to `+19`; one significant pooled cost, `MMLU-Pro` `-2.0`).
3. **`mmap` & Quantized `Q8_0` Surgery:** `secant patch` and `secant quant patch` modify `.safetensors` byte ranges and 34-byte `Q8_0` `.gguf` blocks on disk (copy by default, `--in-place` optional), removing **`98.5%`** of one measured weight component on `Qwen3.5-0.8B` `ssm_out` and modifying **`0.20%`** of the bytes of the `8.67 GB` `Qwen3.5-4B` file (`256`-weight `Q4_K` superblocks can span two 128-column panels and retained `67%` in that test, so `secant quant patch` refuses them).
4. **Project-Then-Heal (`ProjectThenHealLinear`):** Standard LoRA drifts back into a surgical edit's subspace within 60 steps (`0.82%` to `8.51%` relative leakage unconstrained; `0.33%` to `1.67%` even with gradient projection) because AdamW rescales each coordinate. Locking the LoRA factor inside the forward pass holds leakage to **`10^-10` (`F32`) / `10^-17` (`float64`)** (15 models). In the separate 6-model RMS study a healing pass lifted `Qwen2.5-Math-7B` `GSM8K` **`87% -> 93%`** (`n = 100`, unadjusted `p = 0.031`) and `Qwen2.5-Coder-7B` `HumanEval` **`36% -> 42%`** (`p = 0.238`, neither surviving Holm correction); direct 60-step `heal-the-base` (`base_lora_1p0x`) and `healed-random` (`random_healed_rms_1p0x`) control arms show that on `Qwen2.5-Math-7B` both controls also reach **`93%`** on `GSM8K` (`p = 1.000`, proving the `87% -> 93%` lift comes from the 60-step LoRA pass while `ProjectThenHealLinear` cuts domain leakage from `4.75%` to `5.08e-17`), whereas on `Qwen2.5-Coder-7B` `HumanEval` `heal-the-base` (`37%`) and `healed-random` (`35%`) do not reproduce `42%` (`p = 0.332` and `0.189`).
5. **Paired C/C++ Vulnerability Triage (`ASAN-120`):** Across `60` constructed, sanitizer-validated C/C++ functions paired with `60` patched twins (`9` models x `7` arms), small non-reasoning models collapse to one-sided guessing (`93%+` false positives on `Llama-3.1-8B` and `SmolLM2-1.7B`), whereas on reasoning models the `composed_rms_1p0x` edit beats its random control on **`4 / 4` models** (sign test across all 9 models `p = 0.125`) and lifts `Qwen3-14B` strict root-cause recall from **`45%` to `55%`** (`+10` pp, 95% CI `0` to `+21.7`; no comparison survives Holm correction).
6. **Pre-Load Payload & Trojan Auditing:** `secant sanitize` scans `186.8M` GGUF blocks across 10 GGUF and SafeTensors checkpoints with **`0` observed false positives** while catching planted `NaN`/`Inf` scales; its divisor-scale and routing-index rules are validated on synthetic fixtures. `secant audit-trojan` flags a planted rank-one edit once it reaches roughly the layer's top singular value (`8%` to `12%` of weight norm in our tests); below that, single-file inspection shows no signal.

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

---

## Documentation, Blog Series & Paper

- **[4-Part Blog Series ("From Sparks to Secant")](https://m00dy.sh/notes/sparks-to-secant-weight-space-programming-series)**: Plain-English walkthrough and interactive visualizers for Parts I to IV.
- **[Documentation Home](https://professor-moody.github.io/secant/)** · **[Quickstart & Live CLI Captures](https://professor-moody.github.io/secant/quickstart)** · **[CLI & Python Reference](https://professor-moody.github.io/secant/cli-and-python-reference)**
- **[Theory & Calibration](https://professor-moody.github.io/secant/theory-and-calibration)** · **[Quantized Surgery (`Q8_0` vs. `Q4_K`)](https://professor-moody.github.io/secant/quantized-surgery)** · **[Project-Then-Heal LoRA](https://professor-moody.github.io/secant/project-then-heal)** · **[Payload Security](https://professor-moody.github.io/secant/payload-security)**
- **[Full Technical Paper](https://professor-moody.github.io/secant/paper)**: Complete derivations, proofs, and reproducibility receipts.

## Citation

```bibtex
@software{keys2026secant,
  title   = {Secant: Deterministic Weight-Space Programming, In-Place Quantized Block Surgery, and Subspace-Locked LoRA},
  author  = {Keys, Nathan},
  year    = {2026},
  url     = {https://github.com/professor-moody/secant}
}
```
