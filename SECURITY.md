# Security Policy

Secant treats all model containers (`.safetensors` and `.gguf`), LoRA adapters, and companion configuration files (`config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`) as **untrusted binary and structured inputs**.

## Supported Versions

Security fixes are backported only to the current minor release series:

| Version | Supported |
| --- | :---: |
| `0.3.x` | Yes |
| `< 0.3.0` | No |

## Reporting a Vulnerability

**Please do not open public GitHub issues or pull requests for suspected security vulnerabilities.**

Report vulnerabilities privately through **GitHub Private Vulnerability Reporting**:

- **Submit an advisory:** <https://github.com/professor-moody/secant/security/advisories/new>

Include a minimal reproduction (command invocation, synthetic fixture or SHA-256 digest of any public artifact, platform/toolchain version, and observed vs. expected exit code or receipt output). We acknowledge private reports within **72 hours** and coordinate remediation and disclosure before any public commit or advisory is published.

## Security Scope

Reports in the following categories are explicitly in scope:

1. **Memory safety & integer arithmetic:** Panics, out-of-bounds reads/writes, unchecked integer overflow in container header or block-extent calculations, or soundness defects at the `secant-ffi` C ABI boundary (`*_v1` entry points).
2. **Path traversal & file clobbering:** Any path through which an untrusted container header, plan, or CLI invocation overwrites an existing output/receipt file without explicit opt-in or escapes the intended target path.
3. **Fail-open scanner bypasses (`secant sanitize`):** Crafted `.safetensors`, `.gguf`, or sibling companion JSON files (`config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`) that evade documented `Rule S-0` or `Boundary-3` (`B3-1`, `B3-2A`, `B3-2B`, `B3-3`) checks and produce `status: "clean"` (exit code `0`) instead of `status: "findings"` (exit code `1`) or fail-closed `status: "unknown"` (exit code `2`).
4. **Exit-code & hash-bracket contract violations:** Any condition where `secant` returns exit code `0` despite an incomplete scan, unsupported dtype (e.g., unimplemented `F8_E4M3` or unreviewed GGUF quant types), malformed companion config, failed subspace/patch verification gate, or mutated input file (`pre_sha256 != post_sha256` / `input_unchanged == false`).

### Out of Scope / Documented Boundaries

- `secant sanitize` is a **static pre-load container and numerical-domain gate**, not a proof of runtime kernel safety or behavioral alignment.
- Reference-free single-file spectral auditing (`secant audit-trojan` without `--reference`) has documented sub-$g^*$ detection limits in late layers; see [`docs/payload-security.md`](docs/payload-security.md) for measured thresholds.
