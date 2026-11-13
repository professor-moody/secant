# Security Policy

Secant treats all model containers (`.safetensors` and `.gguf`), LoRA adapters, edit plans, and companion configuration files (`config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`) as **untrusted binary and structured inputs**.

## Supported Versions

Security fixes are backported only to the current minor release series:

| Version | Supported |
| --- | :---: |
| `0.3.x` | Yes |
| `< 0.3.0` | No |

## Reporting a Vulnerability

**Please do not open public GitHub issues or pull requests for suspected security vulnerabilities.**

Report vulnerabilities privately through [GitHub Private Vulnerability Reporting](https://github.com/professor-moody/secant/security/advisories/new). If that channel is unavailable, email `professor.moody@pm.me` with the subject `[SECANT SECURITY]` and ask for an encrypted follow-up channel before sending sensitive artifacts.

Include the affected version or commit, exact command, platform and toolchain, observed vs. expected exit code or receipt, and a minimal synthetic reproducer. If a public model artifact is required, provide its URL and SHA-256 digest. Do not send private model weights without prior agreement.

## Security Scope

Reports in the following categories are explicitly in scope:

1. **Memory safety and integer arithmetic:** Panics, out-of-bounds reads/writes, unchecked integer overflow in container header or block-extent calculations, or soundness defects at the `secant-ffi` C ABI boundary (`*_v1` entry points).
2. **Path handling and file clobbering:** Any path through which an untrusted container header, plan, or CLI invocation overwrites an existing output or receipt file without explicit opt-in, escapes an intended target path, or publishes a partial transaction.
3. **Fail-open scanner bypasses (`secant sanitize`):** Crafted `.safetensors`, `.gguf`, or sibling companion JSON files that evade documented `Rule S-0` or `Boundary-3` checks and produce `status: "clean"` (exit code `0`) instead of `status: "findings"` (exit code `1`) or fail-closed `status: "unknown"` (exit code `2`).
4. **Exit-code, schema, and hash-bracket contract violations:** Any condition where `secant` returns exit code `0` despite an incomplete scan, unsupported dtype, malformed companion config, failed subspace or patch gate, incompatible JSON output, or mutated input file (`pre_sha256 != post_sha256` / `input_unchanged == false`).

## Out of Scope and Documented Boundaries

- `secant sanitize` is a **static pre-load container and numerical-domain gate**, not a proof of runtime kernel safety, model provenance, authenticity, malware absence, or behavioral alignment.
- Secant does not replace provenance signatures, malware scanning, sandboxing, inference-runtime testing, or behavioral evaluation.
- Reference-free single-file spectral auditing (`secant audit-trojan` without `--reference`) is heuristic and has documented sub-$g^*$ detection limits in late layers; see [`docs/payload-security.md`](docs/payload-security.md) for measured thresholds. A reproducible bypass of a documented guarantee remains in scope, but merely demonstrating a documented limit is not a vulnerability.
