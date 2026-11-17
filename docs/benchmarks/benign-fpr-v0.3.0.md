# Secant v0.3.0 benign-checkpoint FPR benchmark

Attempt `att-2eeb8901796a6d60`. Candidate changes on
`feat/v0.3.0-release-hygiene`, based on
`7d546d51712a674425dcf33add84512cd66a9ab4`, measured before they were committed.
That branch was rewritten before publication to remove private host names and
paths (see `data/REDACTIONS.md`), so the recorded base commit is not in the
public history; its published counterpart is
`1efff6c` ("chore: prepare v0.3.0 release packaging"). The
published `scripts/bench_benign_fpr.py` records a generic platform label instead
of the machine name, and one test comment in `crates/secant-core/src/security.rs`
lost a dash, so recomputing `source_content_sha256` on the published tree gives
a different digest from the one recorded here. The redaction changed no scanner code.

The observed default-policy FPR is **0 / 105 = 0.0% on each host**: 100 synthetic
configurations and five locally retained benign configurations. The five retained
configurations contain four distinct model payloads plus one companion-config
variant. This is a finite-corpus observation, not a population guarantee or a
claim of 105 independent trained models. No full historical trained checkpoint
was available within the authorized repository.

The read-only `data/sanitize_corpus_20261109` corpus contains historical receipts
for 105 full checkpoints whose inputs reside on a separate Linux x86_64 lab
host. They are **NOT_RUN** here.
Another 23 historical benign file/option rows lack local inputs. Those 128 rows
are explicitly inventoried and excluded from this measurement. All 32 local
GGUF/SafeTensors input files were inventoried, with their provenance labels kept
separate from current scan outcomes. No unclassified local files remain.

## Reproduce

Run from the repository root on macOS or Linux, using a new output directory:

```sh
cargo build --release --locked -p secant-cli -p secant-core --bins --examples
python3 -m unittest discover -s scripts -p 'test_bench_benign_fpr.py'
python3 scripts/bench_benign_fpr.py --run-dir data/benign_fpr_v030/new-run
# On a second host, copy the first host's results.json and add:
# --compare path/to/first-host-results.json
shasum -a 256 -c docs/benchmarks/SHA256SUMS
```

Python uses only the standard library and the native `/usr/bin/time` utility.
The runner writes its inventory, generated models, raw CLI stdout/stderr,
receipts, results, verified `SHA256SUMS`, and Markdown/JSON summaries into the run
directory. Native Secant performs every scan with `sanitize --dry-run --json`.
No remediation flags or threshold adjustments are used.

The generator is `crates/secant-core/examples/benign_fpr_fixtures.rs`; the runner
is `scripts/bench_benign_fpr.py`. Five runner tests check denominator handling,
incomplete/error cases, changed evidence, unclassified inputs, and RSS units and
independence from an earlier high-memory child. Generated payloads are ignored
by Git. Each crate has a symlink to the small canonical JSON summary so Cargo
packages that summary as an ordinary file; no model binaries belong in crates.

## Matrix and measurement

The matrix is four architecture-inspired layouts (Llama, GPT-2, Qwen2, Mixtral)
times five dtypes (F32, F16, BF16, Q8_0, Q4_K) times five deterministic variants.
Widths are 256/512, with one/two layers, embeddings, attention and MLP
projections, near-one norm gammas, and four-expert MoE gates. These are reduced
numerical fixtures with role-oriented tensor names, not loadable trained
architecture implementations. A fixed integer PRNG and CLT-12 normal
approximation generate projection distributions with standard deviations
0.01 to 0.0625, including a sparse wider component. Native Secant encoders
produce Q8_0 and Q4_K payloads; their norm vectors remain F32.

Each host runs 148 cases twice: 105 default benign cases, two optional benign
policy arms, 13 new mathematical positive controls, and 28 retained control
arms. Repeat scans do not inflate the FPR denominator. Incomplete scans or
integrity failures cannot produce a passing FPR. All 296 scans per host matched
their exit/status expectations; both repetitions agree. Cross-host comparison
matches all 148 cases on input/companion hashes, exit code, status, and complete
scan counters/examples. The generated fixture manifests are byte-identical.

Every invocation has timestamped SHA-256 brackets for the scanner binary, model,
and present/absent companion files. JSON stdout must equal the on-disk receipt,
which must retain `secant.wsp.sanitize.v1`, the attempt identity, unchanged input,
no published output, and policy `zero-false-positive-b3-v2`.

Throughput is total input-file MiB divided by total wall seconds. Latency includes
native `time`/CLI startup, Secant's own input hashing, and receipt/JSON output.
Outer hash bracketing is excluded, but it warms the file cache. RSS is the
scanner child's native `time` rusage, normalized from bytes on Darwin and KiB on
Linux. It excludes the Python launcher's pre-exec memory. The hosts are ordinary
research machines, not isolated performance appliances. These measurements do
not establish cold-storage or full-model inference throughput.

## Rule separation

There is no shared scalar S-0/B3 score. Distances are to actual rule boundaries,
computed from serialized tensors. The 13 new controls each produce exit 1 and
the expected finding kind in both repetitions on both hosts.

| Check | Benign boundary clearance | Mathematical positive control |
|---|---|---|
| Norm L2, floor `1e-12` | Minimum `15.945711941383223`, over `1.59e13` times the floor | L2 `0`, `1.599999949224836e-19`, `2.2420775429197073e-44`; `norm_all_zero` / `norm_below_min` |
| Divisor scale, F32 floor `9.999999960041972e-13` | Minimum `0.125`, over `1.25e11` times the floor | `1.401298464324817e-45` and `-1`; `divisor_scale_below_floor` / `divisor_scale_nonpositive` |
| Dense magnitude, ceiling `10000` | Maximum `1.077417254447937` | Retained mathematical probe controls exercise the value gate |
| Q8_0 / Q4_K scale headers | All finite; maximum absolute scale `0.004749298095703125` | One NaN/Inf scale field; `quant_scale_nan` / `quant_scale_inf` |
| S-0 token IDs | Minimum `vocab_size - 1 - eos_token_id = 253` | Headroom `-1`; `s0_companion_token_id_oob` |
| S-0 MoE metadata | `expert_count - expert_used_count = 2` | Headroom `-1`; GGUF and companion expert-used findings |

Dense subnormals and finite quantized multiplier scales are allowed by the
default policy. The subnormal positives above are collapsed norm vectors or
divisor scales; no runtime amplification or exploitation is claimed.

One retained historical control changes finding identity across source
revisions: `companion-hostile-count-zero` previously reported
`s0_companion_expert_count`, and currently reports
`s0_companion_expert_used_count`. Exit 1 is preserved. The current dense-model
zero-sentinel behavior is documented in `CHANGELOG.md:14` and implemented in
`security.rs` companion-config handling. The runner records this historical
difference without treating an old snapshot as a current-version rule contract.
Current synthetic expected kinds and cross-host equality remain mandatory.

## Release limitations and evidence

The measured corpus gate passes. The broader release lane remains **HOLD**:
Human Gate 1 prevents commits, full historical inputs are unavailable within
scope, strict Clippy finds existing warnings, and the optional live Crucible
oracle suite fails on a missing-file error-string difference (`lstat` versus
`open`). Standard `cargo fmt --all -- --check`, standard Clippy, and workspace
tests pass; both platforms report 187 tests. Optional baseline-binary and
external ggml-reference checks are NOT_RUN because their environment variables
are absent. These limits must accompany any release claim.

Raw final evidence is banked in the maintainers' private evidence store under
`w61r-astra-xhigh-secant-fmt-and-fpr-bench/att-2eeb8901796a6d60/`.
The adjacent JSON preserves exact numbers, platform/tool identities, source and
binary digests, manifest digests, coverage, controls, and comparison results.
The result is AI-assisted and awaits human review before submission.

## Exact throughput table

Values below retain the summary JSON precision. `retained-opt-in` is reported
separately and is excluded from `all-default`.

| Host | Cohort | Configurations | Scans | False positives | FPR (%) | MiB/s | ms/checkpoint | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| darwin | all-default | 105 | 210 | 0 | 0.0 | 133.92829544734266 | 62.18142662380952 | 6.671875 |
| darwin | retained-default | 5 | 10 | 0 | 0.0 | 7.906307007236914 | 13.6278459 | 5.125 |
| darwin | retained-opt-in | 2 | 4 | 0 | 0.0 | 15.61942803428776 | 13.80817725 | 5.09375 |
| darwin | synthetic-BF16 | 20 | 40 | 0 | 0.0 | 109.4324210579822 | 82.9550855 | 6.671875 |
| darwin | synthetic-F16 | 20 | 40 | 0 | 0.0 | 125.01320110854633 | 72.61602189999999 | 6.625 |
| darwin | synthetic-F32 | 20 | 40 | 0 | 0.0 | 168.3604857975499 | 107.816250025 | 6.5625 |
| darwin | synthetic-Q4_K | 20 | 40 | 0 | 0.0 | 105.5298823322357 | 24.2508886 | 6.578125 |
| darwin | synthetic-Q8_0 | 20 | 40 | 0 | 0.0 | 136.33246779345353 | 35.407282275 | 6.546875 |
| linux | all-default | 105 | 210 | 0 | 0.0 | 139.3982696275472 | 59.741433652380955 | 7.68359375 |
| linux | retained-default | 5 | 10 | 0 | 0.0 | 11.446708232486312 | 9.4128313 | 7.0078125 |
| linux | retained-opt-in | 2 | 4 | 0 | 0.0 | 20.99682584231417 | 10.27183025 | 6.796875 |
| linux | synthetic-BF16 | 20 | 40 | 0 | 0.0 | 118.44220431187453 | 76.64477285000001 | 7.609375 |
| linux | synthetic-F16 | 20 | 40 | 0 | 0.0 | 93.47218205088822 | 97.1193905 | 7.68359375 |
| linux | synthetic-F32 | 20 | 40 | 0 | 0.0 | 189.167099828142 | 95.9574696 | 7.65234375 |
| linux | synthetic-Q4_K | 20 | 40 | 0 | 0.0 | 174.331112336865 | 14.6800728 | 7.6015625 |
| linux | synthetic-Q8_0 | 20 | 40 | 0 | 0.0 | 179.53107821274608 | 26.8876131 | 7.55859375 |
