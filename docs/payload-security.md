# Container Geometry & Payload Security

Because model weights (`SafeTensors` and `GGUF`) are frequently exchanged across untrusted hubs and fine-tuning pipelines, **Secant treats every container file as an untrusted binary input**. Before any weight-space read or edit, Secant validates payload extents in checked 64-bit arithmetic. Three commands then audit weight **values**: `secant sanitize` (static value scan and optional repaired copy), `secant probe` (boundary-value test fixtures), and `secant read --detect-anomalies` (row and column norm anomalies).

This page describes only what the standalone `secant` CLI implements. The router-sink audit sometimes associated with this topic is `crucible wsp scan-sinks`, a command of the separate Crucible toolchain; the `secant` CLI has no `scan-sinks` subcommand.

---

## 1. Binary Container Invariants (`SafeTensors` & `GGUF` v2/v3)

1. **Checked payload extents (every command):** When a file is opened, each tensor's element count, byte size and payload offset are computed with overflow-checked arithmetic, and every payload interval $[\text{offset}_i, \text{offset}_i + \text{size}_i)$ must lie inside the file. GGUF payload offsets are taken relative to `data_offset`, the header end rounded up to `general.alignment` (default `32`). Any overflow or out-of-file extent is an error (exit code `2`).
2. **Overlap and hole check (`secant sanitize`):** Before scanning values, `sanitize` sorts the tensor intervals and rejects any overlap. For `SafeTensors` it additionally rejects gaps between consecutive payloads and unindexed trailing payload bytes.

---

## 2. Static Weight-Value Scan (`secant sanitize`)

`secant sanitize --model <file> --receipt <new.json> --attempt-id <id>` scans every selected tensor and writes a `secant.wsp.sanitize.v1` receipt.

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

---

## 3. Boundary-Value Fixtures (`secant probe`)

`secant probe` does **not** scan a model. It generates header-valid test fixtures containing boundary values, for exercising how an inference runtime handles them:

| Class | Content |
|---|---|
| `class1` | NaN and $-\infty$ router / gate floats (sentinel comparisons) |
| `class2` | NaN, $+10^{38}$, $-10^{38}$ scale / bias floats (float-to-int saturation vs. overflow) |
| `class3` | NaN arrays: sign, payload and signalling NaN variants, ties, $\pm\infty$ mixes |
| `class4` | `I32` / `I64` routing tables: `-1`, `INT_MIN`, `num_experts`, `INT_MAX`, $2^{32}$, duplicate ids; plus all-zero and subnormal norm vectors |

`secant probe --class <class1..class4|all> --format <safetensors|gguf|both> --out-dir <new-dir> --receipt <new.json> --attempt-id <id>`. With `--base <model>`, probe cells are written into a byte-identical copy of an existing model. Each fixture is self-verified before publication (container inspection, and `sanitize` must report the intended finding kinds). Fixtures are byte-deterministic for a given class, format and `--num-experts`. Exit codes: `0` generated and self-verified, `2` usage, I/O, or could not verify.

---

## 4. Weight-Space Anomaly Detection (`secant read --detect-anomalies`)

During `secant read --detect-anomalies`, each selected 2-D tensor is scanned in one `float64` pass. With row L2 norms $\rho_i = \|W_{i,:}\|_2$ and column L2 norms $c_j = \|W_{:,j}\|_2$, the `anomalies` record (`crates/secant-core/src/read.rs`) holds three counts:

| Field | Criterion |
|---|---|
| **`zero_rows`** | $\rho_i = 0$ |
| **`collapsed_rows`** | $0 < \rho_i < 10^{-3} \cdot \operatorname{median}_r \rho_r$ |
| **`spike_cols`** | $c_j > \operatorname{median}(c) + 10 \cdot 1.4826 \cdot \mathrm{MAD}(c)$ **and** $c_j > 2.0 \cdot \operatorname{median}(c)$, where $\mathrm{MAD}(c) = \operatorname{median}_j \lvert c_j - \operatorname{median}(c)\rvert$ |

The record also carries up to 32 example indices per class, `median_row_l2`, the rule text, and `flagged = (zero_rows + collapsed_rows + spike_cols > 0)`. Detection is informational: `read` exits `0` whether or not anomalies are flagged.

Non-finite values are handled separately from the anomaly record: a tensor holding NaN or $\pm\infty$ gets `status = "nonfinite"` and a `nonfinite_elements` count, and its Frobenius norm and spectrum are reported as undefined (use `secant sanitize` to locate the elements). There is no rank-collapse detector.
