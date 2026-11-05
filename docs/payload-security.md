# Container Geometry & Payload Security

Because model weights (`SafeTensors` and `GGUF`) are frequently exchanged across untrusted hubs and fine-tuning pipelines, **Secant treats every container file as an untrusted binary input**. Before any weight-space read or edit occurs, Secant validates structural bounds, alignment gaps, and spectral anomalies.

---

## 1. Binary Container Invariants (`SafeTensors` & `GGUF` v2/v3)

1. **No Overlapping or Out-of-Bounds Tensor Spans:**
   Every tensor descriptor $[offset_i, offset_i + byte\_len_i)$ is checked in 64-bit checked arithmetic against the file size and against all sibling tensor intervals. Any overlap, unaligned offset, or integer overflow immediately returns exit code `2`.
2. **GGUF Alignment-Gap Auditing (`secant probe` & `secant sanitize`):**
   In GGUF v2/v3, the header-to-data boundary and inter-tensor boundaries are padded to `general.alignment` (default `32` bytes). Malicious or non-compliant writers can smuggle arbitrary payload bytes inside these alignment gaps without affecting standard inference loaders.
   - `secant probe --model <file.gguf>` inspects every padding byte between the end of the metadata KV table and `data_offset`, as well as every inter-tensor alignment gap. If any padding byte is non-zero, `secant probe` flags `NON_ZERO_ALIGNMENT_GAP` and exits with code `1`.
   - `secant sanitize --model <in.gguf> --out <clean.gguf>` zeros all alignment padding regions while verifying that every tensor payload digest (`SHA-256`) remains 100% identical.

---

## 2. Weight-Space Anomaly & Poisoning Detection (`secant read --detect-anomalies`)

During `secant read`, each selected tensor is scanned in a single deterministic `float64` pass for four classes of structural anomalies:

| Anomaly Class | Mathematical Criterion | Security / Reliability Impact |
|---|---|---|
| **`NON_FINITE_VALUES`** | $\exists (i,j) : \mathrm{isnan}(W_{i,j}) \lor \mathrm{isinf}(W_{i,j})$ | Silently corrupts forward activations into `NaN` across all downstream layers. |
| **`COLLAPSED_ROW`** | $\|W_{i,:}\|_2 < 10^{-3} \cdot \mathrm{median}_r \|W_{r,:}\|_2$ | Dead output channel or pruned/zeroed residual coordinate. |
| **`ROW_NORM_SPIKE`** | $\|W_{i,:}\|_2 - \mathrm{median}_r \|W_{r,:}\|_2 > 10 \cdot \mathrm{MAD}$ and ratio $> 2.0$ | Uncalibrated backdoor trigger or extreme outlier feature dominating residual stream norm. |
| **`RANK_COLLAPSE`** | $\sigma_1(W) / \|W\|_F \approx 1$ or $\sigma_1(W) < 10^{-12}$ | Degenerate rank-1 matrix collapse from unstable fine-tuning or malicious weight overwrite. |

---

## 3. Mixture-of-Experts (MoE) Router Sink Auditing (`secant scan-sinks`)

In Mixture-of-Experts architectures, the router gate matrix $W_{\text{gate}} \in \mathbb{R}^{E \times d_{\text{model}}}$ dispatches tokens to the top-$k$ experts. An adversarial perturbation or quantization bias on a single row of $W_{\text{gate}}$ can create a **router sink** that starves other experts or forces tokens through a compromised expert.

`secant scan-sinks --model <path> --json` audits every router gate tensor for row-norm skew, dominant singular alignment, and expert load-balance asymmetry.
