# Native CLI reports and WSP extensions

Secant prints labeled summaries, tables, byte sizes, singular spectra and explicit
gate results by default. `--json` retains the existing serialized reports and
exit semantics: 0 success/pass, 1 failed gate, 2 usage/I/O/unknown. An informational
measurement is not promoted to a gate. For example, `read` anomaly flags remain
informational, and `quant roundtrip` reports error without inventing a threshold.

## Deterministic optimizer rotation

```console
secant rotation-test --tensor-npy w.npy --directions-npy d.npy \
  --rank 4 --steps 60 --lr 0.01 --wd 0.01 --seed 0 --json
```

The tensor is a finite C-order F32/F64 matrix `[d_out, d_in]`; directions are a
vector or `[k, d_out]` rows. Two-pass Gram-Schmidt constructs an orthonormal basis.
The witness fits the normalized residual `T = W / rms(W)` with loss
`mean((B A - T)^2)`. A frozen base cancels out of this residual objective. This
experiment characterizes optimizer geometry, not downstream model quality or the
previous lane's token-level benchmark. The locked loss floor is
`||D T||_F^2 / numel(T)`; finite rank can make the attainable minimum higher.

All arms share SplitMix64-seeded uniform A, zero B, scale 1 and AdamW
`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`. Arm 1 is unconstrained. Arm 2 projects
the B gradient before AdamW; coordinate-wise preconditioning can rotate that
gradient out of the protected complement. Arm 3 implements the float64
`ProjectThenHealLinear` parameterization `B_used = P(P B_tilde)`, including its
projected backward gradient. Both A and B parameters are trained in every arm.

`secant.rotation-test.v1` includes every step, explicit step-1 and step-T
measurements, and relative leakage `||D dW||_F / ||dW||_F` with `dW = B_used A`.
Zero updates have `delta_is_zero=true` and ratio 0 by convention; they cannot
satisfy the structural lock gate. Exit 1 means that gate failed. Leakage in the
comparison arms is expected evidence, not a command failure. Input SHA-256
digests bracket the run. Same inputs, options and build produce identical JSON.
The independent Python test compares every step to PyTorch AdamW and the actual
`secant.nn.ProjectThenHealLinear` implementation.

## Single-tensor HTTP range fetch

```console
secant fetch-tensor --repo org/model --revision main \
  --tensor model.norm.weight --out norm.npy --receipt norm.receipt.json
secant fetch-tensor --repo org/model --tensor lm_head.weight \
  --row-start 0 --row-count 32 --out rows.npy --json
```

The Hub model API resolves the revision to an immutable commit. The fetcher
reads `model.safetensors.index.json` to locate the shard; only a 404 falls back
to the single-file `model.safetensors` layout. It fetches the eight-byte header
length, the JSON header, and the selected contiguous tensor payload. Each range
must return HTTP 206 with the exact requested `Content-Range` and a consistent
shard size. HTTP 200 is rejected without reading a whole shard. Shape, dtype,
offset bounds, response lengths and limits are checked before publication.

Metadata/header responses are limited to 16 MiB each. The default selected raw
payload limit is 512 MiB, configurable with `--max-bytes`. Optional first-axis
row slicing avoids retrieving the rest of a large tensor. F16/BF16 widen to
F32; F32/F64, signed/unsigned 8/16/32/64-bit integers and BOOL retain their
representation. Empty tensors are supported without a payload request.
Outputs never overwrite existing files. If receipt publication fails, the new
NPY is removed. Publication is not an atomic transaction across two files.

The `secant.fetch-tensor.v1` receipt records the resolved commit, shard, original
and selected shape, half-open byte range, raw/header/index/output hashes and
attempt identity. It defaults to `receipt.json` beside the NPY. `HF_TOKEN` is
optional; an explicitly empty token is rejected. Tokens and signed redirect URLs
are not recorded. `--endpoint` supports explicit Hub mirrors and local tests.

## Python integration

Python exposes `secant.rotation_test(...)` and `secant.fetch_tensor(...)` using
the established `CliResult` and strict-error conventions. They locate the native
binary through `SECANT_BIN`, in the repository release directory, or on `PATH`.
A present-but-empty override is an error. Fetch receipts on disk retain the
native `secant.fetch-tensor.v1` schema, and binary SHA-256 digests bracket
execution. Neither command changes the frozen C ABI.

Protocol references: [SafeTensors format](https://github.com/huggingface/safetensors#format)
and [Hugging Face Hub API](https://huggingface.co/docs/hub/en/api).
