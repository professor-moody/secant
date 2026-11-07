# Check Model Files and Weight Values

Because model weights (`SafeTensors` and `GGUF`) are frequently exchanged across untrusted hubs and fine-tuning pipelines, **Secant treats every container file as an untrusted binary input**. Before any weight-space read or edit, Secant validates payload extents in checked 64-bit arithmetic. Three commands then audit weight **values**: `secant sanitize` (static value scan and optional repaired copy), `secant probe` (boundary-value test fixtures), and `secant read --detect-anomalies` (row and column norm anomalies).

The **container** is the file structure that names tensors and locates their bytes. The **payload** is the stored tensor data. A valid container can still hold unusual or invalid numerical values, so Secant checks these separately. This page describes only the standalone `secant` CLI.

---

## 1. Check That Tensor Bytes Fit Inside the File

Before reading a tensor, Secant checks that its declared size and offset describe bytes inside the file. An invariant is a condition that must hold for the file to be accepted. These checks prevent the parser from trusting impossible sizes or locations.

1. **Checked payload extents (every command):** When a file is opened, each tensor's element count, byte size and payload offset are computed with overflow-checked arithmetic, and every payload interval $[\text{offset}_i, \text{offset}_i + \text{size}_i)$ must lie inside the file. GGUF payload offsets are taken relative to `data_offset`, the header end rounded up to `general.alignment` (default `32`). Any overflow or out-of-file extent is an error (exit code `2`).
2. **Overlap and hole check (`secant sanitize`):** Before scanning values, `sanitize` sorts the tensor intervals and rejects any overlap. For `SafeTensors` it additionally rejects gaps between consecutive payloads and unindexed trailing payload bytes.

---

## 2. Scan Values and Optionally Write a Repaired Copy

The scanner looks for the configured findings without running model inference. Repair is explicit: a scan alone does not change the input. Some findings can be replaced under the chosen policy; others cause refusal.

`secant sanitize --model <file> --receipt <new.json> --attempt-id <id>` scans supported static payload values and writes a `secant.wsp.sanitize.v1` receipt. `Q8_0` and `Q4_K` GGUF tensors are scanned at the scale-header level (next subsection). A GGUF containing any other quantized type (`Q4_0`, `Q6_K`, `IQ*`, ...) is refused with exit `2`, even if other tensors are unquantized (`crates/secant-core/src/security.rs`). A successful `inspect` or `quant witness` does not imply `sanitize` supports the same file.

| Finding | Criterion | Repair available |
|---|---|---|
| NaN | any NaN element | `--clamp-nonfinite`: NaN becomes `+0` |
| Infinity | any $\pm\infty$ element | `--clamp-nonfinite`: becomes the signed largest representable value not above `--max-abs` |
| Extreme finite | $\lvert x\rvert >$ `--max-abs` (default `10000`; a policy threshold, not an overflow proof) | opt-in `--clamp-extreme` (Secant extension) |
| Subnormal | subnormal float element | opt-in `--flush-subnormal` (Secant extension) |
| Zero or tiny-norm vector | L2 norm of a selected contiguous vector (default norm-tensor globs, plus `--norm-tensor`) below `--min-norm` (default $10^{-12}$) | none: always refused |
| Routing id out of range | integer routing tables (`tid2eid` / `expert_id`, plus `--routing-tensor`): negative id, or id $\ge$ `--num-experts` | none: always refused |
| Duplicate routing id | repeated id within a routing row | `--dedup-routing-rows`: later duplicates become the lowest unused valid expert id |

`--out <new-file>` publishes a repaired copy only when a remediation flag is given **and** a rescan of the copy is clean; outputs and receipts never overwrite existing files.

Exit codes: `0` clean or remediated under this policy, `1` findings remain, `2` usage, I/O, unsupported input, or incomplete scan. A clean receipt does not prove runtime inference safety.

### Quantized scale headers (`Q8_0`, `Q4_K`)

A `Q8_0` block is `{f16 d; i8 qs[32]}` and decodes to $d \cdot q_i$; a `Q4_K` super-block is `{f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}` and decodes to $d \cdot sc_j \cdot q - dmin \cdot m_j$. One hostile f16 scale poisons all 32 or 256 weights of its block and is the value a consumer divides by or converts (the `QUANT_SCALE_DIV_OR_OVERFLOW` sink class of `secant scan-sinks`). Each scale field is checked against the same `--max-abs` policy, using $M$, the largest multiplier the block actually applies to that field ($\max \lvert qs\rvert$ for `Q8_0` `d`; $\max_j sc_j \cdot \max q$ for `Q4_K` `d`; $\max_j m_j$ for `Q4_K` `dmin`). Arithmetic is exact (f16 widened to `float64`, multipliers at most 945). Code: `crates/secant-core/src/quant_scale.rs`.

| Finding | Criterion | Repair available |
|---|---|---|
| `quant_scale_nan` | the f16 scale is NaN | `--clamp-nonfinite`: `+0` (that term of the block decodes to zero) |
| `quant_scale_inf` | the f16 scale is $\pm\infty$ | `--clamp-nonfinite`: signed largest f16 $v$ with $v \cdot M \le$ `--max-abs` |
| `quant_scale_extreme` | finite and $\lvert scale\rvert \cdot M >$ `--max-abs` | `--clamp-extreme`: the same bounded value |
| `quant_scale_subnormal` | the f16 scale is subnormal | `--flush-subnormal`: signed zero (zeroes up to 256 weights, a larger edit than flushing one float) |
| `quant_block_extreme` | no field is NaN, Inf or extreme, yet a decoded element exceeds `--max-abs` (only when the two `Q4_K` terms have opposite signs; ggml's quantizer writes $d, dmin \ge 0$) | none: always refused |

For these findings `index` is the block index, `row` the GGUF row (`ne[0]` elements), `file_offset` points at the f16 field, `bits` is its encoding, and the receipt adds `field` (`d` or `dmin`). The scan record adds `quant_tensors` and `quant_blocks`, and `policy_version` becomes `unquantized-static-values-v1+quant-scale-headers-v1`. Files without quantized tensors produce exactly the Go oracle's receipt layout. A quantized tensor matched by a norm glob is `unknown` (exit `2`): norm-row checks are implemented for unquantized floats only. The Go oracle refuses every quantized file; this is a recorded divergence (`crates/secant-core/tests/security_parity.rs`, `sanitize_quantized_q4_0_parity_and_q8_q4k_scale_divergence_json`).

(sec-quant-sanitize-controls)=
#### Measured on real quantizer output, with planted controls

This run used `Qwen3.5-4B` BF16 GGUF, quantized by `llama-quantize` (llama.cpp `f39148a95`) to `Q8_0` and to `--pure` `Q4_K`. Those two files are quantizer output that the experimenter produced, not downloaded checkpoints. On a copy of each, three f16 scales in `token_embd.weight` blocks 0, 1 and 2 were overwritten with `dd conv=notrunc`: NaN `0x7e00`, Inf `0x7c00`, and `0x7bff` (65504). In the `Q4_K` copy the Inf went into `dmin` of block 1. Receipts and digests are in the lab evidence bank (receipt `trojan-curve-receipt.json`, SHA-256 `4bd9bc90...c88f1`).

| Input | Exit | Status | Quant blocks | Findings | Kinds |
|---|---:|---|---:|---:|---|
| `Q8_0` clean | `1` | `findings` | `135,168,000` | `272,730` | `quant_scale_subnormal` `272,730` |
| `Q4_K` pure, clean | `1` | `findings` | `16,896,000` | `8,992,791` | `quant_scale_subnormal` `8,992,791` |
| `Q8_0` planted | `1` | `findings` | `135,168,000` | `272,733` | plus exactly one each of `quant_scale_nan`, `quant_scale_inf`, `quant_scale_extreme` |
| `Q4_K` pure, planted | `1` | `findings` | `16,896,000` | `8,992,794` | plus exactly one each of `quant_scale_nan`, `quant_scale_inf`, `quant_scale_extreme` |
| `Q8_0` planted, `--clamp-nonfinite --clamp-extreme --flush-subnormal --out` | `0` | `remediated` | | `0` remaining | changes `1` / `1` / `1` / `272,730` |
| `Q4_K` planted, same flags | `0` | `remediated` | | `0` remaining | changes `1` / `1` / `1` / `8,992,791` |
| Real `Q4_K_M` (`Qwen2.5-7B` shard 1, mixes in `Q6_K`) | `2` | `unknown` | | | `unsupported ggml type 14`: refused, not guessed |

What this shows:

- **The planted scales are found exactly.** Each planted file adds exactly `+1` NaN, `+1` Inf and `+1` extreme finding, with the right `field` (`d`, `d`, `d` for `Q8_0`; `d`, `dmin`, `d` for `Q4_K`) and the planted `file_offset` (for example `10979232` for the NaN). The subnormal count does not change, because the overwritten scales were ordinary values.
- **Ordinary quantizer output has a large subnormal background.** Subnormal f16 scales are `0.20%` of `Q8_0` blocks (`272,730 / 135,168,000`). For `Q4_K` there are `0.53` findings per super-block (`8,992,791 / 16,896,000`). Typical values such as `0x03f8` (about `6.06e-5`) sit just under the f16 normal floor (`6.10e-5`), so they come from blocks of very small weights.
- **So under the default policy, the gate fails every quantized file.** Both clean files exit `1`. A hostile NaN, Inf or extreme scale shows up as a count of 1 among hundreds of thousands of findings. The status alone (`findings`) cannot tell a hostile file from a clean one, but the `counts` map can: read `quant_scale_nan`, `quant_scale_inf` and `quant_scale_extreme` and ignore `quant_scale_subnormal`.
- **`--flush-subnormal` changes real weights.** On these clean files it rewrote `272,730` and `8,992,791` scales to zero, and each rewrite zeroes 32 or 256 weights. Nobody measured how that changes model output, so use `--clamp-nonfinite` and `--clamp-extreme` alone unless you intend that edit.

"Clean" here means "ordinary quantizer output". Nobody ran inference on these files. The per-tensor distribution of the subnormal findings is `NOT_RUN`: the receipt keeps 128 examples. For `Q8_0` they fall in `token_embd.weight` (56) and `blk.0.attn_gate.weight` (72); for `Q4_K` all 128 are in `token_embd.weight`.

---

## 3. Generate Small Test Files (`secant probe`)

`secant probe` does **not** scan a model. It generates header-valid test fixtures containing boundary values, for exercising how an inference runtime handles them:

| Class | Content |
|---|---|
| `class1` | NaN and $-\infty$ router / gate floats (sentinel comparisons) |
| `class2` | NaN, $+10^{38}$, $-10^{38}$ scale / bias floats (float-to-int saturation vs. overflow) |
| `class3` | NaN arrays: sign, payload and signalling NaN variants, ties, $\pm\infty$ mixes |
| `class4` | `I32` / `I64` routing tables: `-1`, `INT_MIN`, `num_experts`, `INT_MAX`, $2^{32}$, duplicate ids; plus all-zero and subnormal norm vectors |

`secant probe --class <class1..class4|all> --format <safetensors|gguf|both> --out-dir <new-dir> --receipt <new.json> --attempt-id <id>`. With `--base <model>`, probe cells replace selected values in a copy of an existing model. Each fixture is self-verified before publication (container inspection, and `sanitize` must report the intended finding kinds). Fixtures are byte-deterministic for a given class, format and `--num-experts`. Exit codes: `0` generated and self-verified, `2` usage, I/O, or could not verify.

---

## 4. Flag Unusual Row and Column Sizes

The reader can flag empty rows, unusually small rows, and unusually large columns. A row or column's L2 norm is its length. The median gives a typical size; median absolute deviation (MAD) measures spread without letting a few extreme values dominate. A flag is a diagnostic clue, not proof of a malicious or broken model.

During `secant read --detect-anomalies`, each selected 2-D tensor is scanned in one `float64` pass. With row L2 norms $\rho_i = \|W_{i,:}\|_2$ and column L2 norms $c_j = \|W_{:,j}\|_2$, the `anomalies` record (`crates/secant-core/src/read.rs`) holds three counts:

| Field | Criterion |
|---|---|
| **`zero_rows`** | $\rho_i = 0$ |
| **`collapsed_rows`** | $0 < \rho_i < 10^{-3} \cdot \mathrm{median}_r \rho_r$ |
| **`spike_cols`** | $c_j > \mathrm{median}(c) + 10 \cdot 1.4826 \cdot \mathrm{MAD}(c)$ **and** $c_j > 2.0 \cdot \mathrm{median}(c)$, where $\mathrm{MAD}(c) = \mathrm{median}_j \lvert c_j - \mathrm{median}(c)\rvert$ |

The record also carries up to 32 example indices per class, `median_row_l2`, the rule text, and `flagged = (zero_rows + collapsed_rows + spike_cols > 0)`. Detection is informational: `read` exits `0` whether or not anomalies are flagged.

Non-finite values are handled separately from the anomaly record: a tensor holding NaN or $\pm\infty$ gets `status = "nonfinite"` and a `nonfinite_elements` count, and its Frobenius norm and spectrum are reported as undefined (use `secant sanitize` to locate the elements). There is no rank-collapse detector.

---

## 5. Spectral Spikes Outside the Marchenko-Pastur Bulk (`secant audit-trojan`)

A `d_out x d_in` matrix of i.i.d. entries with standard deviation $s$ has its singular values inside a bulk whose upper edge is

$$\sigma_{MP} = \sigma_{bulk}\left(1 + \sqrt{d_{in}/d_{out}}\right), \qquad \sigma_{bulk} = s\sqrt{d_{out}}.$$

A rank-1 edit $\theta\, u v^\top$ large enough to change behaviour, and many trojan insertions, detach a singular value from that bulk, so $\sigma_1/\sigma_{MP} \gg 1$ is a spike. `secant read --marchenko-pastur` (and `secant audit-trojan`, which also turns on `--detect-anomalies`) estimates $s^2 = (\lVert W\rVert_F^2 - \sum_{outliers}\sigma_i^2)/(d_{out} d_{in})$, removing computed singular values above the edge until the set is stable, and writes a per-tensor `marchenko_pastur` block (`sigma_entry`, `sigma_bulk`, `sigma_mp`, `sigma1_over_mp`, `outliers`, `outlier_ratios`, `spectrum_exhausted`, `bulk_energy_fraction`, `threshold`, `flagged`, rule text) plus a report-level `spectral_outliers` ranking by ratio. With `--vocab-tensor`, each ranking row carries the top promoted vocabulary tokens of $u_1$. `--mp-threshold` (default `1.5`) sets `flagged`, which means only "outside the i.i.d. bulk". Code: `crates/secant-core/src/mp.rs`.

Trained weights already sit above the i.i.d. edge, so each ranking row also carries a `sibling` record: the tensor's layer family (every all-digit name component becomes `*`, e.g. `blk.*.ffn_down.weight`), the family size and median ratio, the Iglewicz-Hoaglin modified z-score $0.6745\,(x - \mathrm{median})/\mathrm{MAD}$, and `sibling_outlier = z > 3.5`. Families with fewer than 4 members get `sibling_outlier: null` (not judged). Measured on a genuine, unedited `bge-large` quantized to `Q8_0` by `llama-quantize --pure`: all 48 `ffn_up` / `ffn_down` tensors have $\sigma_1/\sigma_{MP}$ between 2.44 and 8.65 (so all 48 are `flagged`), and 5 of 48 are sibling outliers (late `ffn_down` layers 20 to 22, early `ffn_up` layers 0 and 1). Treat a sibling outlier as a lead, and confirm an edit by comparing against the base checkpoint.

Limits stated plainly. Only the top-k computed singular values can be removed, so with more than k outliers the edge is over-estimated and `outliers` is a lower bound (`spectrum_exhausted: true`, shown as `k+`). When the computed spectrum explains all but $10^{-8}$ of the energy (an exactly low-rank matrix, a vector), no bulk remains: `status` is `undefined: ...`, no ratio is reported and nothing is flagged. Trained weights are not i.i.d., and ratios above 1 occur in genuine checkpoints: compare a tensor against its sibling layers or against the base checkpoint before an edit, not against 1. Detection is informational: both commands exit `0` whether or not anything is flagged. Without `--marchenko-pastur`, the `read` receipt is unchanged (no new keys).

```bash
secant audit-trojan --model edited.gguf --tensor 'blk.*.ffn_down.weight' --vocab-tensor output.weight
secant read --model edited.gguf --marchenko-pastur --mp-threshold 2.0 --json
```

(sec-trojan-dose-curve)=
### 5.1 How Large Must a Planted Edit Be Before the Audit Sees It?

This experiment planted a rank-1 edit of known size in two real checkpoints, then ran `secant audit-trojan` on every edited copy. It measures the edit size at which each signal appears. Edit: `rank1-columns` on the layer-12 MLP down projection. The direction is the unit-normalized embedding row of a trigger token T. The edit touches 16 evenly spaced input columns with equal coefficients. The **dose** is $g = \lVert\Delta W\rVert_F / \lVert W\rVert_F$, measured from the patch receipt after `BF16` rounding.

| Arm | Checkpoint | Layer family audited | Trigger token | $\lVert W\rVert_F$ |
|---|---|---|---|---:|
| A | `SmolLM2-1.7B-Instruct` SafeTensors (tied embeddings) | `model.layers.*.mlp.down_proj.weight` (24) | `41401` (" giraffe") | `458.666` |
| C | `Qwen3.5-4B` BF16 GGUF | `blk.*.ffn_down.weight` (33) | `16657` (" cheese") | `41.581` |

All 24 audits (12 doses on each model) exit `0`. Each run brackets its input: the patched file's SHA-256 equals the audit's `pre_sha256` and `post_sha256`, and `input_unchanged` is true. That is a non-adversarial host-filesystem check, not descriptor-bound execution. Doses `0` to `4%` were specified before the run. They showed nothing, so `6`, `8`, `10`, `12` and `16%` were **added afterwards**. That is a recorded deviation: every threshold below is post-hoc, and since the true threshold lies somewhere between two tested doses, each first-detection dose is an upper bound.

```{figure} assets/trojan-detection-phase-transition.svg
:name: fig-trojan-phase-transition
:alt: Two panels against relative Frobenius dose from 0 to 16 percent for SmolLM2-1.7B and Qwen3.5-4B. Panel A shows the sibling modified z-score crossing 3.5 at 12 percent, sigma_1 over sigma_MP rising from 8 percent, and sigma_2 jumping at 6 percent. Panel B shows the cosine between the planted direction and u_1 jumping from below 0.2 to above 0.98 between 6 and 10 percent, and the trigger token entering the promoted top-16 at rank 1 or 2 at 8 percent.
:width: 100%

The planted edit stays hidden until its singular value $\sigma_{trojan} = g\,\lVert W\rVert_F$ reaches the unedited layer's $\sigma_1$. That happens at exactly $g^* = \sigma_1/\lVert W\rVert_F$: `6.57%` for `Qwen3.5-4B` and `8.65%` for `SmolLM2-1.7B` (dashed lines). **Panel A** shows the automatic family statistics. The sibling z-score crosses `3.5` only at `12%`, and $\sigma_2$ is the first quantity to move, at `6%`. **Panel B** shows the vocabulary projection of $u_1$. The trigger token is rank 1 (Qwen) or rank 2 (SmolLM2) at `8%`, one tested dose after the crossing, while the sibling flag is still silent.
```

**Arm A, `SmolLM2-1.7B`, layer 12 of 24.** $\sigma_{trojan}/\sigma_1$ is the realized edit norm divided by the unedited layer's $\sigma_1$.

| Dose | $\sigma_{trojan}/\sigma_1$ | $\sigma_1$ | $\sigma_2$ | $\sigma_1/\sigma_{MP}$ | Sibling z | Outlier | Family rank | $\cos(u, u_1)$ | Rank of T in top 16 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `0.00%` | `0` | `39.697` | `25.31` | `2.632` | `1.42` | no | `4` | `0.059` | none |
| `0.10%` | `0.012` | `39.697` | `25.31` | `2.632` | `1.42` | no | `4` | `0.059` | none |
| `0.25%` | `0.029` | `39.697` | `25.31` | `2.632` | `1.42` | no | `4` | `0.059` | none |
| `0.50%` | `0.058` | `39.698` | `25.31` | `2.632` | `1.42` | no | `4` | `0.060` | none |
| `1.00%` | `0.116` | `39.699` | `25.31` | `2.632` | `1.42` | no | `4` | `0.061` | none |
| `2.00%` | `0.231` | `39.704` | `25.31` | `2.632` | `1.42` | no | `4` | `0.064` | none |
| `4.00%` | `0.462` | `39.725` | `25.31` | `2.632` | `1.42` | no | `4` | `0.081` | none |
| `6.00%` (post-hoc) | `0.693` | `39.789` | **`30.10`** | `2.636` | `1.43` | no | `4` | `0.133` | none |
| `8.00%` (post-hoc) | `0.924` | `40.450` | `37.93` | `2.679` | `1.58` | no | `3` | `0.569` | **`2`** |
| `10.00%` (post-hoc) | `1.155` | `47.700` | `39.47` | `3.160` | `3.12` | no | **`1`** | `0.984` | `1` |
| `12.00%` (post-hoc) | `1.386` | `56.533` | `39.56` | `3.745` | **`5.01`** | **yes** | `1` | `0.993` | `1` |
| `16.00%` (post-hoc) | `1.849` | `74.529` | `39.60` | `4.937` | `8.84` | yes | `1` | `0.997` | `1` |

**Arm C, `Qwen3.5-4B` GGUF, layer 12 of 33.**

| Dose | $\sigma_{trojan}/\sigma_1$ | $\sigma_1$ | $\sigma_2$ | $\sigma_1/\sigma_{MP}$ | Sibling z | Outlier | Family rank | $\cos(u, u_1)$ | Rank of T in top 16 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `0.00%` | `0` | `2.7311` | `2.112` | `2.189` | `0.33` | no | `10` | `0.008` | none |
| `0.10%` | `0.015` | `2.7311` | `2.112` | `2.189` | `0.33` | no | `10` | `0.008` | none |
| `0.25%` | `0.038` | `2.7311` | `2.112` | `2.189` | `0.33` | no | `10` | `0.008` | none |
| `0.50%` | `0.076` | `2.7311` | `2.112` | `2.189` | `0.33` | no | `10` | `0.008` | none |
| `1.00%` | `0.152` | `2.7311` | `2.112` | `2.188` | `0.33` | no | `10` | `0.008` | none |
| `2.00%` | `0.304` | `2.7311` | `2.112` | `2.188` | `0.33` | no | `10` | `0.009` | none |
| `4.00%` | `0.609` | `2.7312` | `2.115` | `2.187` | `0.33` | no | `10` | `0.014` | none |
| `6.00%` (post-hoc) | `0.913` | `2.7326` | **`2.678`** | `2.188` | `0.33` | no | `10` | `0.168` | none |
| `8.00%` (post-hoc) | `1.218` | `3.4657` | `2.731` | `2.775` | `1.28` | no | `3` | **`0.991`** | **`1`** |
| `10.00%` (post-hoc) | `1.523` | `4.2710` | `2.731` | `3.420` | `2.43` | no | **`1`** | `0.994` | `1` |
| `12.00%` (post-hoc) | `1.827` | `5.0850` | `2.731` | `4.071` | **`3.58`** | **yes** | `1` | `0.996` | `1` |
| `16.00%` (post-hoc) | `2.436` | `6.7265` | `2.731` | `5.386` | `5.92` | yes | `1` | `0.998` | `1` |

What the curve shows:

1. **Below $\sigma_1$, the edit is hidden.** Through `4%` (the whole pre-specified range) neither model shows anything a reviewer could act on. The sibling z moves by under `0.01`, $\sigma_1$ by under `0.1%`, and T never enters the top 16 of $u_1$. This follows from the arithmetic: at `4%`, $\sigma_{trojan}$ is `0.46` (SmolLM2) and `0.61` (Qwen) of the unedited $\sigma_1$, so the edit sits inside the trained spectrum.
2. **$\sigma_2$ moves first, but you need the base checkpoint to see it.** At `6%`, $\sigma_2$ rises from `2.112` to `2.678` on Qwen and from `25.31` to `30.10` on SmolLM2, while $\sigma_1$ and the ratio stay flat. `audit-trojan` ranks only $\sigma_1/\sigma_{MP}$, so this precursor appears only when the edited file is compared against its base. The data do not show that such a comparison would catch doses below `6%`: at `4%` on SmolLM2, $\sigma_{trojan}$ is below $\sigma_4$ (`18.3` vs. `23.45`).
3. **The vocabulary projection comes before the automatic flag.** At `8%` the trigger token is the top promoted token of $u_1$ on Qwen and the second on SmolLM2, with $\cos(u, u_1)$ of `0.991` and `0.569`. The sibling z is still `1.28` and `1.58`. Someone reading the `PROMOTED TOKENS` column would see it; the automatic flag would not. The jump in $\cos(u, u_1)$ straddles the exact crossing dose $g^*$ (`6.57%` Qwen, `8.65%` SmolLM2): this is the edit's singular value overtaking the trained $\sigma_1$, so the top singular vector switches to the planted direction.
4. **Family rank 1 at `10%`, sibling flag at `12%`.** At `10%` the edited layer becomes the family's largest $\sigma_1/\sigma_{MP}$ on both models. The automatic sibling outlier ($z > 3.5$) first fires at `12%`, and only just on Qwen ($z = 3.58$). Whether another layer, token or column placement would move that threshold is `UNKNOWN`.
5. **No false positives on the unedited bases.** The unedited bases have 0 sibling outliers among 24 SmolLM2 and 33 Qwen layers. At `12%` and `16%` the planted layer is the only sibling outlier in each family. Nearly every tensor is `flagged` as outside the i.i.d. bulk, as Section 5 predicts for trained weights, so `flagged` is not a detector.

Limits stated plainly: this is a static weight audit. It tested one layer, one token and one 16-column placement per model, and ran no inference, so it says nothing about whether the planted edit changes model behaviour. `secant audit-trojan` reports only the top-16 promoted and suppressed tokens, so a full-vocabulary rank of T is `NOT_RUN`. The rank-1 crossing holds for any rank-1 edit, but these thresholds are measured for this layer only. SmolLM2 layer 20, which has a much lower baseline ratio, is `NOT_RUN`.

---

## 6. Find Weight-Value Sinks in Consumer Source (`secant scan-sinks`)

`secant scan-sinks --source <dir>` scans C, C++, CUDA, Metal and Rust sources (`.c .h .cc .cpp .cxx .hpp .hh .hxx .inc .inl .mm .metal .cu .cuh .rs`) for four places where a format-valid but hostile weight value turns into a memory-safety, DoS or abort event:

| Sink class | Shape |
|---|---|
| `FLOAT_TO_INDEX_CAST` | a float, logit, score or weight cast to an integer type (C cast, `static_cast`, `lrint` family, Rust `as`) |
| `UNVALIDATED_ROUTING_OR_INDEX_TENSOR` | an expert-id, token-id, index or permutation tensor used as a subscript or pointer offset |
| `QUANT_SCALE_DIV_OR_OVERFLOW` | a reciprocal of, division by, or f16 conversion of a quantization scale (`d`, `dmin`, `scale`, `amax`, ...) |
| `WEIGHT_VALUE_RELEASE_ASSERT` | a release-mode abort (`GGML_ASSERT`, `CHECK*`, `TORCH_CHECK`, `ORT_ENFORCE`, ...) over loaded values; `assert`, `DCHECK*`, `OP_REQUIRES*` and `RET_CHECK*` are excluded |

Each hit records whether a guard-shaped expression (`isnan`/`isfinite`, a bound check, a scale-vs-zero test) appears on the sink line or earlier in the same function, and a `risk_score` (kernel and loader paths raise it, guards and test paths lower it, Rust float `as` casts saturate and are down-weighted). Comments and string literals are blanked first; hidden and VCS directories and symlinks are skipped; files over 4 MiB are counted in `files_skipped`. The `secant.wsp-sink-scan.v1` receipt (`--out <new.json>`, or `--json` to stdout) carries exact per-class and unguarded counts, the ranked list cut to `--top` (default 100), and `tree_digest_sha256` over every scanned `(path, sha256)` pair, so the same tree gives the same receipt. Exit `0` when the scan completes, `2` on usage or I/O errors.

The receipt is byte-identical to the Go oracle's `wsp scan-sinks` except for `schema`, the third `limitations` sentence, and the timestamps; that is verified on nine source trees (`crates/secant-cli/tests/scan_sinks_cli.rs`, `receipt_is_byte_identical_to_the_oracle`). One recorded divergence: a source line longer than 1 MiB stops the oracle's scan of that file silently, while Secant scans every line. A hit is a hypothesis: replay it and read the source at the site before calling it a bug.

The nine parity trees hold `9,612` hits in total. Seven are inference and ML source trees: llama.cpp `ggml/src` `2,328`, whisper.cpp `2,547`, vLLM `csrc` `572`, XNNPACK `3,613`, tract `204`, sentencepiece `131` and OpenCV `modules/dnn` `137`, which together account for `9,532`. The other two are a 12-hit synthetic fixture (CRLF, invalid UTF-8, hidden and symlinked files) and `68` hits in crucible's own native code. These are pattern matches used for oracle parity. They are not counts of bugs, and nobody has triaged them.

---

## 7. A Static Atlas of 46 Local Checkpoints

To see what genuine, unedited weights look like to these commands, `secant read --top-k-singular 3 --top-cols 8 --detect-anomalies [--vocab-tensor ...] --json` was run on 46 local checkpoints. It used the shipped release binary (SHA-256 `0060b2ee...a682`) and is evidence of stored weight values only. Every input was hashed before and after the whole sweep (46 of 46 identical). Every `read` receipt also carries `pre_sha256 == post_sha256`.

| Coverage | Files | Meaning |
|---|---:|---|
| Full | 14 | every tensor read |
| Scoped | 16 | 12 `Qwen3.5-4B` and 4 `SmolLM2` edited variants, read only where `verify-patch --base` reports a difference |
| None | 16 | vocab-only GGUF files: `read` exits `2` with `empty or invalid tensor layout` |

The sweep produced 429 non-empty `read` receipts. Across them it analyzed `2,619` tensors and skipped `253` whose quantization `read` cannot decode (`Q4_0`, `Q2_K`, `Q3_K`, others). It found `0` tensors with non-finite values, which agrees with the 26 `clean` results from `sanitize`. Of `5,101` computed singular triplets, `323` (`6.3%`) reached the 1000-iteration cap without converging. Treat $\sigma_2$ and $\sigma_3$ from those as less reliable.

- **Lossless conversion between formats.** For all 32 `Qwen3.5-4B` MLP down projections, the SafeTensors and GGUF copies have equal Frobenius norms and bit-equal $\sigma_1..\sigma_3$. The promoted token lists match on all 29 layers whose shard also holds the vocabulary tensor.
- **Watch the sign of singular vectors when comparing token lists.** For two edited variants of `blk.16.ffn_down.weight`, the promoted and suppressed lists swap places relative to the base (Jaccard `0` each, `1` when crossed). That comes from the sign convention of $v_1$ (largest component positive), not from a change in which tokens the direction touches. Compare the union of promoted and suppressed tokens.
- **Anomalies are stored-weight facts, not verdicts.** Examples: the `Qwen3.5-4B` vision tower has collapsed rows in `model.visual.blocks.{0..3}.mlp.linear_fc1.weight` (2, 2, 26 and 18). The dequantized `token_embd.weight` of `Qwen2.5-7B` `Q4_K_M` has `2,358` zero rows; whether those zeros come from the source weights or from the quantizer is `UNKNOWN`, because no BF16 copy was available.

The planted-trojan curve in [Section 5.1](#sec-trojan-dose-curve) uses two of these models. Its unedited layer-12 $\sigma_1$ values (`39.69683998584097` for SmolLM2, `2.731132194906345` for Qwen) are bit-identical to this atlas, even though the two runs used different `secant` builds.
