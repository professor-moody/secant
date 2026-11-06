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
