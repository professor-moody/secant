# ASAN-RealWorld-60 Evaluation Summary (`crucible.asan_realworld_60.results.v1`)

- **Benchmark**: `asan_realworld_60_paired_benchmark.json` (`sha256=7aaf5593da3f9f84a5df8d77102966ce223a753f7d4d647a1292df1b5f33a151`)
- **Suite size**: 30 real sanitizer-reproduced upstream C/C++ defects paired with their 30 exact upstream patches = **60 functions** (expanding `ASAN-120` to `ASAN-180`)
- **Independent rescore mismatches**: `0`

## Primary Evaluation Table (`base` vs `composed` / `calibrated_1p0x` vs `random` / `random_1p0x`)

| Model | Arm | Balanced Acc (BA) | TPR (Vuln Recall) | TNR (Patched Spec.) | Patched FPR (`1-TNR`) | MCC | Root-Cause Recall (`RC`) | Class Acc | Sink Acc | Strict Pair-BA | Parsed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen35_0p8b` (`Qwen/Qwen3.5-0.8B`) | `base` | **50.0%** | 0.0% | 100.0% | 0.0% | `+0.000` | **0.0%** | 0.0% | 10.0% | 50.0% | `60/60` |
| `qwen35_0p8b` (`Qwen/Qwen3.5-0.8B`) | `composed` (`calibrated_1p0x`) | **50.0%** | 0.0% | 100.0% | 0.0% | `+0.000` | **0.0%** | 0.0% | 3.3% | 50.0% | `60/60` |
| `qwen35_0p8b` (`Qwen/Qwen3.5-0.8B`) | `random` (`random_1p0x`) | **50.0%** | 0.0% | 100.0% | 0.0% | `+0.000` | **0.0%** | 0.0% | 13.3% | 50.0% | `60/60` |
| `qwen35_0p8b` (`Qwen/Qwen3.5-0.8B`) | `composed_rms_1p0x` | **50.0%** | 0.0% | 100.0% | 0.0% | `+0.000` | **0.0%** | 0.0% | 10.0% | 50.0% | `60/60` |
| `qwen35_0p8b` (`Qwen/Qwen3.5-0.8B`) | `random_rms_1p0x` | **50.0%** | 0.0% | 100.0% | 0.0% | `+0.000` | **0.0%** | 0.0% | 10.0% | 50.0% | `60/60` |
| `qwen25_1p5b` (`Qwen/Qwen2.5-1.5B-Instruct`) | `base` | **45.0%** | 20.0% | 70.0% | 30.0% | `-0.115` | **0.0%** | 0.0% | 16.7% | 35.0% | `60/60` |
| `qwen25_1p5b` (`Qwen/Qwen2.5-1.5B-Instruct`) | `composed` (`calibrated_1p0x`) | **46.7%** | 20.0% | 73.3% | 26.7% | `-0.079` | **0.0%** | 3.3% | 13.3% | 36.7% | `60/60` |
| `qwen25_1p5b` (`Qwen/Qwen2.5-1.5B-Instruct`) | `random` (`random_1p0x`) | **46.7%** | 6.7% | 86.7% | 13.3% | `-0.111` | **0.0%** | 0.0% | 16.7% | 43.3% | `60/60` |
| `qwen25_1p5b` (`Qwen/Qwen2.5-1.5B-Instruct`) | `composed_rms_1p0x` | **45.0%** | 10.0% | 80.0% | 20.0% | `-0.140` | **0.0%** | 0.0% | 13.3% | 40.0% | `60/60` |
| `qwen25_1p5b` (`Qwen/Qwen2.5-1.5B-Instruct`) | `random_rms_1p0x` | **45.0%** | 20.0% | 70.0% | 30.0% | `-0.115` | **0.0%** | 3.3% | 13.3% | 35.0% | `60/60` |
| `smollm2_1p7b` (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) | `base` | **48.3%** | 96.7% | 0.0% | 100.0% | `-0.130` | **0.0%** | 16.7% | 13.3% | 0.0% | `58/60` |
| `smollm2_1p7b` (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) | `composed` (`calibrated_1p0x`) | **50.0%** | 100.0% | 0.0% | 100.0% | `+0.000` | **0.0%** | 16.7% | 6.7% | 0.0% | `59/60` |
| `smollm2_1p7b` (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) | `random` (`random_1p0x`) | **50.0%** | 100.0% | 0.0% | 100.0% | `+0.000` | **0.0%** | 16.7% | 6.7% | 0.0% | `59/60` |
| `smollm2_1p7b` (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) | `composed_rms_1p0x` | **48.3%** | 96.7% | 0.0% | 100.0% | `-0.130` | **0.0%** | 16.7% | 6.7% | 0.0% | `58/60` |
| `smollm2_1p7b` (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) | `random_rms_1p0x` | **50.0%** | 100.0% | 0.0% | 100.0% | `+0.000` | **0.0%** | 16.7% | 13.3% | 0.0% | `60/60` |
| `qwen35_4b` (`Qwen/Qwen3.5-4B`) | `base` | **51.7%** | 63.3% | 40.0% | 60.0% | `+0.034` | **10.0%** | 16.7% | 43.3% | 25.0% | `58/60` |
| `qwen35_4b` (`Qwen/Qwen3.5-4B`) | `composed` (`calibrated_1p0x`) | **50.0%** | 60.0% | 40.0% | 60.0% | `+0.000` | **10.0%** | 16.7% | 36.7% | 25.0% | `58/60` |
| `qwen35_4b` (`Qwen/Qwen3.5-4B`) | `random` (`random_1p0x`) | **53.3%** | 63.3% | 43.3% | 56.7% | `+0.068` | **10.0%** | 16.7% | 43.3% | 26.7% | `58/60` |

## Paired Contrasts (Cluster Bootstrap 95% CI over 30 Twin Pairs, B=5000 + Exact McNemar p)

| Model | Contrast | ΔBA `[95% CI]` | ΔTPR `[95% CI]` | ΔTNR `[95% CI]` | ΔRC `[95% CI]` | McNemar Vuln `(b,c,p)` | McNemar Safe `(b,c,p)` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen35_0p8b` | `calibrated_1p0x__vs__base` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `qwen35_0p8b` | `random_1p0x__vs__base` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `qwen35_0p8b` | `calibrated_1p0x__vs__random_1p0x` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `qwen35_0p8b` | `composed_rms_1p0x__vs__base` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `qwen35_0p8b` | `random_rms_1p0x__vs__base` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `qwen35_0p8b` | `composed_rms_1p0x__vs__random_rms_1p0x` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `qwen25_1p5b` | `calibrated_1p0x__vs__base` | `+1.7` `[+0.0, +5.0]` | `+0.0` `[-10.0, +10.0]` | `+3.3` `[+0.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `(1,1,1.000)` | `(1,0,1.000)` |
| `qwen25_1p5b` | `random_1p0x__vs__base` | `+1.7` `[-5.0, +8.3]` | `-13.3` `[-26.7, -3.3]` | `+16.7` `[+3.3, +30.0]` | `+0.0` `[+0.0, +0.0]` | `(0,4,0.125)` | `(5,0,0.062)` |
| `qwen25_1p5b` | `calibrated_1p0x__vs__random_1p0x` | `+0.0` `[-8.3, +8.3]` | `+13.3` `[+3.3, +26.7]` | `-13.3` `[-26.7, -3.3]` | `+0.0` `[+0.0, +0.0]` | `(4,0,0.125)` | `(0,4,0.125)` |
| `qwen25_1p5b` | `composed_rms_1p0x__vs__base` | `+0.0` `[-5.0, +5.0]` | `-10.0` `[-20.0, +0.0]` | `+10.0` `[+0.0, +20.0]` | `+0.0` `[+0.0, +0.0]` | `(0,3,0.250)` | `(3,0,0.250)` |
| `qwen25_1p5b` | `random_rms_1p0x__vs__base` | `+0.0` `[-6.7, +6.7]` | `+0.0` `[-10.0, +10.0]` | `+0.0` `[-10.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `(1,1,1.000)` | `(1,1,1.000)` |
| `qwen25_1p5b` | `composed_rms_1p0x__vs__random_rms_1p0x` | `+0.0` `[-6.7, +6.7]` | `-10.0` `[-20.0, +0.0]` | `+10.0` `[+0.0, +23.3]` | `+0.0` `[+0.0, +0.0]` | `(0,3,0.250)` | `(3,0,0.250)` |
| `smollm2_1p7b` | `calibrated_1p0x__vs__base` | `+1.7` `[+0.0, +5.0]` | `+3.3` `[+0.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(1,0,1.000)` | `(0,0,1.000)` |
| `smollm2_1p7b` | `random_1p0x__vs__base` | `+1.7` `[+0.0, +5.0]` | `+3.3` `[+0.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(1,0,1.000)` | `(0,0,1.000)` |
| `smollm2_1p7b` | `calibrated_1p0x__vs__random_1p0x` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(0,0,1.000)` |
| `smollm2_1p7b` | `composed_rms_1p0x__vs__base` | `+0.0` `[-5.0, +5.0]` | `+0.0` `[-10.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(1,1,1.000)` | `(0,0,1.000)` |
| `smollm2_1p7b` | `random_rms_1p0x__vs__base` | `+1.7` `[+0.0, +5.0]` | `+3.3` `[+0.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(1,0,1.000)` | `(0,0,1.000)` |
| `smollm2_1p7b` | `composed_rms_1p0x__vs__random_rms_1p0x` | `-1.7` `[-5.0, +0.0]` | `-3.3` `[-10.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,1,1.000)` | `(0,0,1.000)` |
| `qwen35_4b` | `calibrated_1p0x__vs__base` | `-1.7` `[-8.3, +3.3]` | `-3.3` `[-10.0, +0.0]` | `+0.0` `[-10.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `(0,1,1.000)` | `(1,1,1.000)` |
| `qwen35_4b` | `random_1p0x__vs__base` | `+1.7` `[+0.0, +5.0]` | `+0.0` `[+0.0, +0.0]` | `+3.3` `[+0.0, +10.0]` | `+0.0` `[+0.0, +0.0]` | `(0,0,1.000)` | `(1,0,1.000)` |
| `qwen35_4b` | `calibrated_1p0x__vs__random_1p0x` | `-3.3` `[-8.3, +0.0]` | `-3.3` `[-10.0, +0.0]` | `-3.3` `[-10.0, +0.0]` | `+0.0` `[+0.0, +0.0]` | `(0,1,1.000)` | `(0,1,1.000)` |

## Sealed 30-Defect Real-World Upstream Manifest (`RW01`..`RW30`, 60 Functions)

- **Manifest**: `asan_realworld_60_manifest.json` (`sha256=d4ac48b0d9ed9414d4bd06a46322a76b083576fd97a071db969e0c263dc21d08`)
- **Sanitizer Verification Receipt**: `validation_receipt.json` (`30/30` pairs PASS across `clang++ -O1`, `-O0`, `-O2` with `-fsanitize=address,undefined`)

| Case (`V` / `S`) | Upstream Repo | Base Commit | File Path | Function | Bug Class (`ASAN-120` Family) | Crucible Packet ID | `patch.diff` SHA-256 (prefix) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RW01` (`RW01-V` / `RW01-S`) | `freetype/freetype` | `c4a3c188e82b` | `src/cff/cffload.c` | `cff_blend_build_vector` | `null_pointer_offset_before_bounds_check` (`F07`) | `freetype-cff-blend-region-index-check` | `b452858548ebb355…` |
| `RW02` (`RW02-V` / `RW02-S`) | `protocolbuffers/protobuf` | `ab41516b5544` | `upb/mem/arena.c` | `_upb_Arena_AllocBlockInternal` | `integer_overflow_in_arena_block_allocation` (`F04`) | `protobuf-upb-arena-overflow-main-rebase` | `4911998b461f41a6…` |
| `RW03` (`RW03-V` / `RW03-S`) | `protocolbuffers/protobuf` | `bbe8d1453a43` | `upb/reflection/file_def.c` | `_upb_FileDef_Create` | `null_pointer_passed_to_memset_on_zero_extensions` (`F07`) | `protobuf-upb-filedef-null-memset` | `ae10a2eb7c910aa3…` |
| `RW04` (`RW04-V` / `RW04-S`) | `protocolbuffers/protobuf` | `d99b5ba1b633` | `upb/message/promote.c` | `upb_MiniTable_ParseUnknownMessage` | `heap_oob_read_on_unvalidated_unknown_field_length` (`F01`) | `protobuf-upb-promote-wiretype-validation-v2` | `14fe6165d49c10cc…` |
| `RW05` (`RW05-V` / `RW05-S`) | `pnggroup/libpng` | `ac0c3532476f` | `pngpread.c` | `png_process_IDAT_data` | `heap_uaf_on_save_buffer_reallocation_during_row_pause` (`F03`) | `libpng-progressive-pause-lifecycle` | `e4dbf691888d83ce…` |
| `RW06` (`RW06-V` / `RW06-S`) | `openxla/xla` | `bae1f051129a` | `xla/service/shape_inference.cc` | `ShapeInference::InferTopKShape` | `heap_oob_read_on_scalar_topk_operand_last_dim` (`F01`) | `openxla-xla-shape-inference-dynamic-dimension-bounds` | `bc89b6370d0d40d3…` |
| `RW07` (`RW07-V` / `RW07-S`) | `harfbuzz/harfbuzz` | `0ba555a3d4f0` | `src/hb-ot-layout-common.hh` | `VarRegionList::serialize` | `unchecked_unsigned_multiplication_before_overflow_helper` (`F04`) | `harfbuzz-cff2-private-blend-overflow-and-serialize-bounds` | `15d15781206b332f…` |
| `RW08` (`RW08-V` / `RW08-S`) | `AOMediaCodec/libavif` | `90d3b6661e14` | `src/write.c` | `avifEncoderWriteHDRProperties` | `null_dereference_of_dedupstream_on_sequence_track_hdr` (`F07`) | `libavif-sequence-track-null-dedupstream` | `8a01b5290e015666…` |
| `RW09` (`RW09-V` / `RW09-S`) | `AOMediaCodec/libavif` | `9b07a9d7a874` | `src/read.c` | `avifDecoderCreateCodecs` | `null_dereference_after_partial_tile_codec_creation_failure` (`F07`) | `libavif-decoder-lifecycle-persistent-io-and-codec-rollback` | `87b2e8ce2e3181b2…` |
| `RW10` (`RW10-V` / `RW10-S`) | `lz4/lz4` | `d5f17575750e` | `lib/lz4frame.c` | `LZ4F_decodeBlock` | `unsigned_remaining_content_size_underflow_in_lz4frame` (`F04`) | `lz4-lz4frame-remaining-content-size-underflow` | `146347716c5e57f4…` |
| `RW11` (`RW11-V` / `RW11-S`) | `google/flatbuffers` | `5eb190f00005` | `src/idl_gen_text.cpp` | `JsonPrinter::GenStruct` | `heap_oob_read_of_unverified_deprecated_vtable_slot` (`F01`) | `flatbuffers-gentext-skip-deprecated-fields` | `c6686dbcf01da3af…` |
| `RW12` (`RW12-V` / `RW12-S`) | `libjxl/libjxl` | `b7ec9c0f63a6` | `lib/extras/enc/pnm.cc` | `BasePNMEncoder::Encode` | `heap_oob_read_on_mismatched_frame_extra_channels_count` (`F01`) | `libjxl-extras-apng-pnm-extra-channel-bounds` | `dc361f0dbc28afad…` |
| `RW13` (`RW13-V` / `RW13-S`) | `google/XNNPACK` | `fc1a2623138f` | `src/subgraph/reshape-helpers.c` | `resize_unary_elementwise_output_tensor` | `oob_write_copying_unvalidated_input_rank_into_tensor_shape` (`F01`) | `xnnpack-subgraph-slice-and-reshape-bounds-v2` | `cf2d206349c391e5…` |
| `RW14` (`RW14-V` / `RW14-S`) | `vllm-project/vllm` | `4527515cd705` | `csrc/cpu/sgl-kernels/moe.cpp` | `moe_align_block_size` | `heap_oob_write_indexing_moe_workspace_with_unvalidated_topk_ids` (`F01`) | `vllm-cpu-hash-moe-tid2eid-bounds` | `3a3651e417367101…` |
| `RW15` (`RW15-V` / `RW15-S`) | `mlc-ai/xgrammar` | `11f65251738b` | `cpp/grammar_functor.cc` | `GrammarFSMBuilderImpl::BuildRegex` | `heap_uaf_reading_rule_name_pointer_after_rule_vector_reallocation` (`F03`) | `xgrammar-build-regex-rule-reallocation-uaf-and-recursion-bounds` | `76122bec5a8f81a8…` |
| `RW16` (`RW16-V` / `RW16-S`) | `mlc-ai/xgrammar` | `ca1acf28bf35` | `cpp/lark_converter.cc` | `LarkParser::ParseAtom` | `unbounded_recursion_on_nested_lark_grammar_blocks` (`F12`) | `xgrammar-lark-nested-grammar-depth-guard-v2` | `76b2f09fbb35d026…` |
| `RW17` (`RW17-V` / `RW17-S`) | `onnx/onnx` | `48ace3d46387` | `onnx/defs/controlflow/utils.cc` | `LoopInferenceFunction` | `assert_only_loop_input_count_check_underflows_in_ndebug` (`F10`) | `onnx-loop-shape-inference-input-count` | `98d31397be797776…` |
| `RW18` (`RW18-V` / `RW18-S`) | `opencv/opencv` | `7dc486f74c62` | `modules/dnn/src/darknet/darknet_io.cpp` | `ReadDarknetConvGroups` | `divide_by_zero_on_zero_convolutional_groups_in_darknet_importer` (`F04`) | `opencv-darknet-conv-groups-zero` | `c52000389e0c1f3d…` |
| `RW19` (`RW19-V` / `RW19-S`) | `opencv/opencv` | `7dc486f74c62` | `modules/ml/src/tree.cpp` | `DTreesImpl::predictTrees` | `heap_oob_read_on_unvalidated_dtree_split_variable_index` (`F01`) | `opencv-ml-dtrees-load-bounds` | `20f13b0b629989b4…` |
| `RW20` (`RW20-V` / `RW20-S`) | `opencv/opencv` | `f706833347e1` | `modules/dnn/src/torch/torch_importer.cpp` | `TorchImporter::readTensor` | `type_confusion_and_heap_oob_when_reusing_torch_storage_of_smaller_type` (`F11`) | `opencv-torch-storage-reuse-type-check` | `e68262e635bb78ff…` |
| `RW21` (`RW21-V` / `RW21-S`) | `openvinotoolkit/openvino` | `a05daefea7cb` | `src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp` | `ModelDeserializer::process_model` | `heap_oob_read_in_memcpy_when_compiled_model_blob_smaller_than_header` (`F01`) | `openvino-cpu-compiled-model-sections-master-rebase` | `765491c6dd29c329…` |
| `RW22` (`RW22-V` / `RW22-S`) | `apache/arrow` | `1d3cf6f18205` | `cpp/src/arrow/ipc/message.cc` | `ReadMessageAsync` | `signed_int64_overflow_in_arrow_ipc_metadata_plus_body_length` (`F04`) | `arrow-ipc-readmessage-body-length` | `c1fb2cd98f9daf6e…` |
| `RW23` (`RW23-V` / `RW23-S`) | `apache/arrow` | `1d3cf6f18205` | `cpp/src/arrow/ipc/reader.cc` | `RecordBatchFileReaderImpl::GetRecordBatchBlock` | `dcheck_only_record_batch_index_check_allows_oob_in_release` (`F10`) | `arrow-ipc-record-batch-index-bounds` | `5719a47836aaca6a…` |
| `RW24` (`RW24-V` / `RW24-S`) | `openxla/xla` | `bae1f051129a` | `xla/hlo/evaluator/hlo_evaluator.cc` | `HloEvaluator::HandleSort` | `heap_oob_read_indexing_key_shape_dimensions_before_sort_dim_check` (`F01`) | `openxla-xla-hlo-evaluator-dimension-bounds` | `8e77bb1de80857cd…` |
| `RW25` (`RW25-V` / `RW25-S`) | `openxla/xla` | `bae1f051129a` | `xla/tsl/lib/io/snappy/snappy_inputbuffer.cc` | `SnappyInputBuffer::Inflate` | `dcheck_only_snappy_output_buffer_capacity_check_allows_heap_oob_write` (`F10`) | `openxla-xla-tsl-snappy-input-buffer-capacity` | `aea5089a50752871…` |
| `RW26` (`RW26-V` / `RW26-S`) | `openxla/xla` | `bae1f051129a` | `xla/service/hlo_verifier.cc` | `ShapeVerifier::VerifyEntryComputationLayout` | `heap_oob_read_indexing_parameter_layout_on_out_of_range_alias_parameter_number` (`F01`) | `openxla-xla-hlo-verifier-alias-and-sort-bounds` | `01a291bc7745c231…` |
| `RW27` (`RW27-V` / `RW27-S`) | `tensorflow/tensorflow` | `4a50894fcbae` | `tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc` | `IsCancellableConstPermTransposeNodePair` | `heap_oob_read_indexing_fanout_permutation_with_unvalidated_fanin_entry` (`F01`) | `tensorflow-grappler-generic-layout-pad-perm-oob` | `20268b61b8d8cc32…` |
| `RW28` (`RW28-V` / `RW28-S`) | `tensorflow/tensorflow` | `ace46d5052cf` | `tensorflow/core/grappler/optimizers/constant_folding.cc` | `ConstantFolding::PrepareConstantPushDown` | `null_dereference_binding_reference_to_missing_child_input_node` (`F07`) | `tensorflow-grappler-constant-folding-mulconv-and-rank-guards` | `4e21ba7abf572195…` |
| `RW29` (`RW29-V` / `RW29-S`) | `tensorflow/tensorflow` | `cd7fc0e41c85` | `tensorflow/core/util/tensor_bundle/tensor_bundle.cc` | `BundleReader::GetValue` | `negative_section_size_converted_to_size_t_in_threaded_tensor_bundle_restore` (`F05`) | `tensorflow-tensor-bundle-getvalue-threaded-restore-bounds` | `1e805f02c9a079e8…` |
| `RW30` (`RW30-V` / `RW30-S`) | `tensorflow/tensorflow` | `d384f6f8186e` | `tensorflow/lite/core/interpreter_builder.cc` | `InterpreterBuilder::operator()` | `null_dereference_when_tflite_model_omits_subgraphs_vector` (`F07`) | `tensorflow-tflite-interpreter-builder-null-subgraphs` | `6e36cd58011dbbaa…` |
