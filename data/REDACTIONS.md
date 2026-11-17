# Host and path redactions in retained data

The receipts and scripts below were committed with absolute paths from the
maintainers' lab machines and with internal host names. Those strings identify
private infrastructure and are not needed to check any result, so they were
replaced. **No measured value, count, score, digest of a model or plan, or
timestamp was changed.** Every edit is a substitution inside a path or host
string, plus one renamed boolean key in the reference-diff receipt whose name
contained a host name.

## Substitutions

| Original form | Replacement |
| --- | --- |
| absolute lab run roots on the Linux workers | `<lab-runs>/` |
| the same roots flattened into file names (`__..._crucible-runs__`) | `__lab-runs__` |
| a lab run directory whose name contained a host name | `<lab-runs>/wsp-lab/` |
| absolute local model store on a Linux worker | `<lab-models>/` |
| absolute path of this repository on the macOS host | repository-relative path |
| absolute path of the private evidence bank | `<evidence-bank>/` |
| `host` values naming a Linux worker | `linux-x86_64-host` |
| `host` values naming the macOS host | `darwin-arm64-host` |
| key `source_hashes_match_reviewed_<host>_tree` | `source_hashes_match_reviewed_darwin_arm64_host_tree` |

In `build_and_validate_realworld_60.py`, the absolute packet root now comes from
the `SECANT_PACKETS_ROOT` environment variable (default `patch-rewards-packets`,
relative to the working directory), and the unused output-directory constant
points at the script's own directory.

## Digests

The pre-redaction bytes remain in public history at commit `f0ece91`. To check
a row, run `git show f0ece91:<path> | shasum -a 256` for the left digest and
`shasum -a 256 <path>` for the right one. Directory manifests
(`MANIFEST.sha256`, `SHA256SUMS`) and documents that cite these files were
updated to the redacted digests.

One exception is deliberate. `trojan-multilayer-behavioral-receipt.json` keeps
`preregistration_sha256` at the pre-redaction value, because that field records
the specification that existed when the cells ran. The same historical value is
quoted where the papers describe the fixed-grid specification.

| Path | Pre-redaction SHA-256 (`f0ece91`) | Redacted SHA-256 |
| --- | --- | --- |
| `data/asan_realworld_60/asan_realworld_60_results.json` | `59402d8fb6627ba08c3a399d97e57a5ac75ecf229ccd727019979536cd138ba8` | `0a5358cbd63533b33d0048795e2d8a7833f257189ee08eb66efcf3eac22bb169` |
| `data/asan_realworld_60/build_and_validate_realworld_60.py` | `7184d841dbd2f2a91043f29b9c370548a6379008ddbe2864e64c805fd24b1eab` | `2ffc8fcb078fe310448aa9680d99699a4f7dacd0e1c45158e87d575c4200e340` |
| `data/asan_realworld_60/runs/qwen25_1p5b/model-meta.json` | `f35597fda1acb1e55f96c1b28f20b73409f55421668e1ff67eda951d57900010` | `cf73898ee1f9665b23485060678d6d896ab654ba61ffbf6e3bb5737360e63503` |
| `data/asan_realworld_60/runs/qwen35_0p8b/model-meta.json` | `3f5b651356b516ba6c976e954cb5efd9285058b9a38722dcada62abeb6a1c85e` | `cdba02e53eb5cd1c7a20ee83592328fcab0b1d39c1c26a72e5b0df7bd9858358` |
| `data/asan_realworld_60/runs/qwen35_4b/model-meta.json` | `1ebe94b5f9ae1ccf87d9ef4bedb0045a8f7b365b39238b6b93e018a374e2f1b1` | `aa4a78f318df93f92bd63c31fd207641055e6c3993fefafd62fbc4e7c1c63724` |
| `data/asan_realworld_60/runs/smollm2_1p7b/model-meta.json` | `ee3333331649b636307974c6ceef248f0f45a32a3acf039eba9146f368aa758f` | `b89078621d99cd8500574efb684f5b9f5b7a2c368df92d2bb932b96cc05e908b` |
| `data/lora_controls_20261109/qwen25_coder_7b/receipt.json` | `fe65a39dd858de04e47c90193560687afed46afdee4489277e9458de9a238563` | `2a90ba6790c9d8e60741e22be227ef1ecac0d866e0de5ce939bbb1ece9fa8037` |
| `data/lora_controls_20261109/qwen25_math_7b/receipt.json` | `025065edcbcc28835d45caba06a61b658c180f05251210c1a5fc29dc267f0f00` | `36131804df9270e72ff14986e5029309eb1b3aac2e3b0daad2cdb57fc912f5fc` |
| `data/multiseed_and_dose_20261109/qwen35_08b_multiseed_receipt.json` | `f7da83186596c764f4711e4d2293772751dbcdaf75eac754faa45d90713e6dc2` | `be4aa480356cc6fca76c5f268acb43cfdcac0f83148fd615c39e1f9531dfa20e` |
| `data/multiseed_and_dose_20261109/qwen35_4b_dose_and_rand3_receipt.json` | `eae628d0611e382b5dd21f0719cf28deddc53dda6dbff92564aa9e2e88eaa5ca` | `f977078a1c0a16e0c7d14cc5e92fb8e91272e87a1d7bacd7a0491b3cc03fe3d5` |
| `data/reference_diff_late_layers_20261109/reference_diff_late_layer_receipt.json` | `76def87f376b3ff30e8cb98047f227384a0d3746fc86287964a63f0069f19d10` | `c162042deb7a3605ddff7f57be9fc62d9531267155435e020f87d329db4e83c3` |
| `data/trojan_multilayer_20261109/preregistration_spec.json` | `f979471c41a4ec88933d7951ef4edd4aa052fd0f9efa6245be2c382d44f55b70` | `d65b9f1dfe3c9b158291deb6de4a463194c2b1acedd423131217df34aa87f508` |
| `data/trojan_multilayer_20261109/trojan-multilayer-behavioral-receipt.json` | `0aad7ae7eb0a1d5f5b6e4bd85f31b6307e3317635c8af8e6a277a93d5fffc978` | `c63a024c252d945a4e01f8fc05c01f07b83c205b6db1c4242fd4d463cd8f871f` |
