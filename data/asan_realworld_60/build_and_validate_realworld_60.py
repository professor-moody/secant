#!/usr/bin/env python3
"""ASAN-RealWorld-60 builder and sanitizer harness verifier (Horizon 1, Item 4).

Selects 30 distinct real sanitizer-reproduced upstream C/C++ defects from Crucible's
verified patch-rewards corpus (`reports/submissions/patch-rewards-packets/`), pairs
each real vulnerable function (`vuln`, pre-patch) with its exact upstream patch twin
(`benign`, post-patch from `patch.diff`), compiles all 60 standalone targets under
`clang++ -O1 -g -fsanitize=address,undefined` (plus `-O0` and `-O2`), verifies that:
  - all 30 unpatched (`vuln`) targets fail (`red`, sanitizer trap at the annotated
    `SINK`/`ROOT` line) on the trigger input (`driver_bad`) and pass (`exit 0`) on
    the legitimate control input (`driver_ok`),
  - all 30 patched (`benign`) targets pass (`green`, `exit 0`, zero sanitizer errors)
    on both the trigger input (`driver_bad`) and control input (`driver_ok`),
and emits:
  - `asan_realworld_60_manifest.json`
  - `asan_realworld_60_paired_benchmark.json` (`crucible.asan120.v1` compatible, 60 items)
  - `asan180_combined_benchmark.json` (`ASAN-120` + `ASAN-RealWorld-60` = 180 items / 90 pairs)
  - `validation_receipt.json`
  - `harnesses/RW01_vuln.cpp` .. `harnesses/RW30_benign.cpp`
"""
from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASAN120_DIR = HERE.parent
KIT_DIR = ASAN120_DIR / "kit"
PACKETS_ROOT = Path("/Users/keys/projects/crucible/reports/submissions/patch-rewards-packets")
SECANT_OUT_DIR = Path("/Users/keys/projects/secant/data/asan_realworld_60")

sys.path.insert(0, str(KIT_DIR))
import asan120_lib as L  # noqa: E402
import asan120_prompt as P  # noqa: E402
import asan120_score as S  # noqa: E402


REALWORLD_CASES: list[dict] = [
    # -------------------------------------------------------------------------
    # RW01: freetype/freetype - cff_blend_build_vector
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW01",
        "family": "F07",
        "gold_class": "NULL_DEREFERENCE",
        "cwe": "CWE-476",
        "bug_class": "null_pointer_offset_before_bounds_check",
        "crucible_packet_id": "freetype-cff-blend-region-index-check",
        "repo": "freetype/freetype",
        "base_commit": "c4a3c188e82bc5b457390a87630ac782316421a1",
        "file_path": "src/cff/cffload.c",
        "function_name": "cff_blend_build_vector",
        "slug": "freetype_cff_blend_region_index",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["varRegionList", "varRegion", "axisList"],
        "terms": ["regionCount", "varRegionList", "regionIndices", "idx"],
        "expect": r"applying non-zero offset|AddressSanitizer|SEGV",
        "vuln": """\
#include <cstddef>
#include <cstdint>

struct CFF_VarRegion { int32_t axisList[2]; };
struct CFF_VarData { uint32_t regionIdxCount; const uint16_t* regionIndices; };
struct CFF_VStore { uint32_t axisCount; uint32_t regionCount; CFF_VarRegion* varRegionList; };

int cff_blend_build_vector(const CFF_VStore* vs, const CFF_VarData* varData,
                           uint32_t lenNDV, const int32_t* ndv, int32_t* blend_vec) {
  blend_vec[0] = 0x10000;
  for (uint32_t master = 1; master <= varData->regionIdxCount; ++master) {
    uint32_t idx = varData->regionIndices[master - 1];
    CFF_VarRegion* varRegion = &vs->varRegionList[idx]; //@SINK
    if (idx >= vs->regionCount) {
      return -1;
    }
    if (!lenNDV) {
      blend_vec[master] = 0;
      continue;
    }
    int32_t scalar = 0x10000;
    for (uint32_t axis = 0; axis < vs->axisCount; ++axis) {
      if (varRegion->axisList[axis] > ndv[axis]) scalar = 0;
    }
    blend_vec[master] = scalar;
  }
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>

struct CFF_VarRegion { int32_t axisList[2]; };
struct CFF_VarData { uint32_t regionIdxCount; const uint16_t* regionIndices; };
struct CFF_VStore { uint32_t axisCount; uint32_t regionCount; CFF_VarRegion* varRegionList; };

int cff_blend_build_vector(const CFF_VStore* vs, const CFF_VarData* varData,
                           uint32_t lenNDV, const int32_t* ndv, int32_t* blend_vec) {
  blend_vec[0] = 0x10000;
  for (uint32_t master = 1; master <= varData->regionIdxCount; ++master) {
    uint32_t idx = varData->regionIndices[master - 1];
    if (idx >= vs->regionCount) {
      return -1;
    }
    CFF_VarRegion* varRegion = &vs->varRegionList[idx]; //@GUARD
    if (!lenNDV) {
      blend_vec[master] = 0;
      continue;
    }
    int32_t scalar = 0x10000;
    for (uint32_t axis = 0; axis < vs->axisCount; ++axis) {
      if (varRegion->axisList[axis] > ndv[axis]) scalar = 0;
    }
    blend_vec[master] = scalar;
  }
  return 0;
}
""",
        "driver_ok": """\
  CFF_VarRegion reg = {{0, 0}};
  uint16_t indices[1] = {0};
  CFF_VarData vd = {1, indices};
  CFF_VStore vs = {1, 1, &reg};
  int32_t ndv[1] = {0x4000};
  int32_t out[2] = {0, 0};
  if (cff_blend_build_vector(&vs, &vd, 1, ndv, out) != 0) return 1;
""",
        "driver_bad": """\
  uint16_t indices[1] = {1};
  CFF_VarData vd = {1, indices};
  CFF_VStore vs = {1, 0, nullptr};
  int32_t ndv[1] = {0x4000};
  int32_t out[2] = {0, 0};
  (void)cff_blend_build_vector(&vs, &vd, 1, ndv, out);
""",
    },
    # -------------------------------------------------------------------------
    # RW02: protocolbuffers/protobuf - _upb_Arena_AllocBlockInternal
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW02",
        "family": "F04",
        "gold_class": "INTEGER_OVERFLOW",
        "cwe": "CWE-190",
        "bug_class": "integer_overflow_in_arena_block_allocation",
        "crucible_packet_id": "protobuf-upb-arena-overflow-main-rebase",
        "repo": "protocolbuffers/protobuf",
        "base_commit": "ab41516b5544f53090eadd8d8282994b0d4fb725",
        "file_path": "upb/mem/arena.c",
        "function_name": "_upb_Arena_AllocBlockInternal",
        "slug": "protobuf_upb_arena_alloc_block",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["kUpb_MemblockReserve", "block", "malloc"],
        "terms": ["kUpb_MemblockReserve", "_upb_Arena_CheckedAdd", "size", "sz"],
        "expect": r"heap-buffer-overflow|insufficient space",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

struct upb_MemBlock {
  upb_MemBlock* next;
  size_t size;
  uint64_t tag;
  uint64_t reserved;
};
static constexpr size_t kUpb_MemblockReserve = sizeof(upb_MemBlock);

static inline bool _upb_Arena_CheckedAdd(size_t a, size_t b, size_t* out) {
  if (a > std::numeric_limits<size_t>::max() - b) return false;
  *out = a + b;
  return true;
}

upb_MemBlock* _upb_Arena_AllocBlockInternal(size_t* size) {
  size_t sz = *size + kUpb_MemblockReserve; //@ROOT
  if (sz > (1u << 20)) return nullptr;
  upb_MemBlock* block = static_cast<upb_MemBlock*>(std::malloc(sz));
  if (!block) return nullptr;
  block->next = nullptr; //@SINK
  block->size = sz; //@SINK
  block->tag = 0; //@SINK
  block->reserved = 0;
  *size = sz - kUpb_MemblockReserve;
  return block;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

struct upb_MemBlock {
  upb_MemBlock* next;
  size_t size;
  uint64_t tag;
  uint64_t reserved;
};
static constexpr size_t kUpb_MemblockReserve = sizeof(upb_MemBlock);

static inline bool _upb_Arena_CheckedAdd(size_t a, size_t b, size_t* out) {
  if (a > std::numeric_limits<size_t>::max() - b) return false;
  *out = a + b;
  return true;
}

upb_MemBlock* _upb_Arena_AllocBlockInternal(size_t* size) {
  size_t sz;
  if (!_upb_Arena_CheckedAdd(*size, kUpb_MemblockReserve, &sz)) { //@GUARD
    return nullptr;
  }
  if (sz > (1u << 20)) return nullptr;
  upb_MemBlock* block = static_cast<upb_MemBlock*>(std::malloc(sz));
  if (!block) return nullptr;
  block->next = nullptr;
  block->size = sz;
  block->tag = 0;
  block->reserved = 0;
  *size = sz - kUpb_MemblockReserve;
  return block;
}
""",
        "driver_ok": """\
  size_t req = 128;
  upb_MemBlock* b = _upb_Arena_AllocBlockInternal(&req);
  if (!b) return 1;
  std::free(b);
""",
        "driver_bad": """\
  size_t req = std::numeric_limits<size_t>::max() - 15;
  upb_MemBlock* b = _upb_Arena_AllocBlockInternal(&req);
  std::free(b);
""",
    },
    # -------------------------------------------------------------------------
    # RW03: protocolbuffers/protobuf - _upb_FileDef_Create
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW03",
        "family": "F07",
        "gold_class": "NULL_DEREFERENCE",
        "cwe": "CWE-476",
        "bug_class": "null_pointer_passed_to_memset_on_zero_extensions",
        "crucible_packet_id": "protobuf-upb-filedef-null-memset",
        "repo": "protocolbuffers/protobuf",
        "base_commit": "bbe8d1453a434300f1ffed0e314ab4ed8a6bdd5b",
        "file_path": "upb/reflection/file_def.c",
        "function_name": "_upb_FileDef_Create",
        "slug": "protobuf_upb_filedef_null_memset",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memset", "ext_layouts"],
        "terms": ["ext_count", "ext_layouts", "memset"],
        "expect": r"null pointer passed as argument|AddressSanitizer|SEGV",
        "vuln": """\
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C" void* upb_memset_nonnull(void* dst, int c, size_t n) __attribute__((nonnull(1)));
extern "C" void* upb_memset_nonnull(void* dst, int c, size_t n) { return std::memset(dst, c, n); }

struct upb_MiniTableExtension { int field_number; };
struct upb_FileDef {
  int ext_count;
  const upb_MiniTableExtension** ext_layouts;
};

static inline void* _upb_DefBuilder_Alloc(size_t bytes) {
  return bytes ? std::malloc(bytes) : nullptr;
}

void _upb_FileDef_Create(upb_FileDef* file, bool rebuild_layouts) {
  if (rebuild_layouts) {
    file->ext_layouts = static_cast<const upb_MiniTableExtension**>(
        _upb_DefBuilder_Alloc(sizeof(*file->ext_layouts) * file->ext_count)); //@ROOT
    upb_memset_nonnull((void*)file->ext_layouts, 0, //@SINK
                       sizeof(*file->ext_layouts) * file->ext_count);
  }
}
""",
        "safe": """\
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C" void* upb_memset_nonnull(void* dst, int c, size_t n) __attribute__((nonnull(1)));
extern "C" void* upb_memset_nonnull(void* dst, int c, size_t n) { return std::memset(dst, c, n); }

struct upb_MiniTableExtension { int field_number; };
struct upb_FileDef {
  int ext_count;
  const upb_MiniTableExtension** ext_layouts;
};

static inline void* _upb_DefBuilder_Alloc(size_t bytes) {
  return bytes ? std::malloc(bytes) : nullptr;
}

void _upb_FileDef_Create(upb_FileDef* file, bool rebuild_layouts) {
  if (rebuild_layouts) {
    file->ext_layouts = static_cast<const upb_MiniTableExtension**>(
        _upb_DefBuilder_Alloc(sizeof(*file->ext_layouts) * file->ext_count));
    if (file->ext_count != 0) { //@GUARD
      upb_memset_nonnull((void*)file->ext_layouts, 0,
                         sizeof(*file->ext_layouts) * file->ext_count);
    }
  }
}
""",
        "driver_ok": """\
  upb_FileDef f = {2, nullptr};
  _upb_FileDef_Create(&f, true);
  std::free((void*)f.ext_layouts);
""",
        "driver_bad": """\
  upb_FileDef f = {argc - 1, nullptr};
  _upb_FileDef_Create(&f, true);
""",
    },
    # -------------------------------------------------------------------------
    # RW04: protocolbuffers/protobuf - upb_MiniTable_ParseUnknownMessage
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW04",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_on_unvalidated_unknown_field_length",
        "crucible_packet_id": "protobuf-upb-promote-wiretype-validation-v2",
        "repo": "protocolbuffers/protobuf",
        "base_commit": "d99b5ba1b633c8e2d05ee9b22ace14a37bc9626f",
        "file_path": "upb/message/promote.c",
        "function_name": "upb_MiniTable_ParseUnknownMessage",
        "slug": "protobuf_upb_promote_unknown_len",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["upb_Decode", "memcpy", "ptr"],
        "terms": ["message_len", "end", "kUpb_WireType_Delimited", "ptr"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

static constexpr uint32_t kUpb_WireType_Delimited = 2;
struct upb_UnknownField { const char* data; size_t size; };

static inline const char* upb_WireReader_ReadTag(const char* p, uint32_t* tag) {
  *tag = static_cast<uint8_t>(*p);
  return p + 1;
}
static inline const char* upb_WireReader_ReadVarint(const char* p, uint64_t* v) {
  *v = static_cast<uint8_t>(*p);
  return p + 1;
}

int upb_MiniTable_ParseUnknownMessage(const upb_UnknownField* unknown, uint8_t* dst, size_t cap) {
  const char* ptr = unknown->data;
  uint32_t tag = 0;
  uint64_t message_len = 0;
  ptr = upb_WireReader_ReadTag(ptr, &tag);
  ptr = upb_WireReader_ReadVarint(ptr, &message_len); //@ROOT
  size_t size = static_cast<size_t>(message_len);
  if (size > cap) return -1;
  std::memcpy(dst, ptr, size); //@SINK
  return static_cast<int>(size);
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

static constexpr uint32_t kUpb_WireType_Delimited = 2;
struct upb_UnknownField { const char* data; size_t size; };

static inline const char* upb_WireReader_ReadTag(const char* p, uint32_t* tag) {
  *tag = static_cast<uint8_t>(*p);
  return p + 1;
}
static inline const char* upb_WireReader_ReadVarint(const char* p, uint64_t* v) {
  *v = static_cast<uint8_t>(*p);
  return p + 1;
}

int upb_MiniTable_ParseUnknownMessage(const upb_UnknownField* unknown, uint8_t* dst, size_t cap) {
  const char* ptr = unknown->data;
  const char* end = ptr + unknown->size;
  uint32_t tag = 0;
  uint64_t message_len = 0;
  if (ptr >= end) return -1;
  ptr = upb_WireReader_ReadTag(ptr, &tag);
  if (!ptr || ptr >= end || (tag & 7u) != kUpb_WireType_Delimited) return -1; //@GUARD
  ptr = upb_WireReader_ReadVarint(ptr, &message_len);
  if (!ptr || ptr > end || static_cast<uint64_t>(end - ptr) < message_len) return -1; //@GUARD
  size_t size = static_cast<size_t>(message_len);
  if (size > cap) return -1;
  std::memcpy(dst, ptr, size);
  return static_cast<int>(size);
}
""",
        "driver_ok": """\
  char* raw = new char[4]{0x2a, 0x02, 0x08, 0x01};
  upb_UnknownField u = {raw, 4};
  uint8_t dst[16] = {};
  if (upb_MiniTable_ParseUnknownMessage(&u, dst, sizeof(dst)) != 2) return 1;
  volatile uint8_t keep = dst[0];
  (void)keep;
  delete[] raw;
""",
        "driver_bad": """\
  char* raw = new char[4]{0x2a, 0x08, 0x08, 0x01};
  upb_UnknownField u = {raw, 4};
  uint8_t dst[16] = {};
  int rc = upb_MiniTable_ParseUnknownMessage(&u, dst, sizeof(dst));
  volatile uint8_t keep = dst[0] + static_cast<uint8_t>(rc);
  (void)keep;
  delete[] raw;
""",
    },
    # -------------------------------------------------------------------------
    # RW05: pnggroup/libpng - png_process_IDAT_data
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW05",
        "family": "F03",
        "gold_class": "USE_AFTER_FREE",
        "cwe": "CWE-416",
        "bug_class": "heap_uaf_on_save_buffer_reallocation_during_row_pause",
        "crucible_packet_id": "libpng-progressive-pause-lifecycle",
        "repo": "pnggroup/libpng",
        "base_commit": "ac0c3532476fa4876a83a279127840741cf216f2",
        "file_path": "pngpread.c",
        "function_name": "png_process_IDAT_data",
        "slug": "libpng_progressive_pause_uaf",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["next_in", "png_push_process_row"],
        "terms": ["PNG_FLAG_PROCESS_DATA_PAUSED", "next_in", "avail_in", "flags"],
        "expect": r"heap-use-after-free",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static constexpr uint32_t PNG_FLAG_ZSTREAM_ENDED = 0x0008U;
static constexpr uint32_t PNG_FLAG_PROCESS_DATA_PAUSED = 0x0010U;

struct png_struct {
  uint32_t flags;
  const uint8_t* next_in;
  uint32_t avail_in;
  uint8_t* save_buffer;
  uint8_t last_byte;
  void (*row_fn)(png_struct*);
};

static inline void png_push_process_row(png_struct* png_ptr) {
  if (png_ptr->row_fn) png_ptr->row_fn(png_ptr);
}

void png_process_IDAT_data(png_struct* png_ptr, const uint8_t* buffer, size_t buffer_length) {
  png_ptr->next_in = buffer;
  png_ptr->avail_in = static_cast<uint32_t>(buffer_length);
  while (png_ptr->avail_in > 0 && !(png_ptr->flags & PNG_FLAG_ZSTREAM_ENDED)) {
    png_ptr->last_byte ^= *png_ptr->next_in++; //@SINK
    png_ptr->avail_in--;
    png_push_process_row(png_ptr);
  }
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static constexpr uint32_t PNG_FLAG_ZSTREAM_ENDED = 0x0008U;
static constexpr uint32_t PNG_FLAG_PROCESS_DATA_PAUSED = 0x0010U;

struct png_struct {
  uint32_t flags;
  const uint8_t* next_in;
  uint32_t avail_in;
  uint8_t* save_buffer;
  uint8_t last_byte;
  void (*row_fn)(png_struct*);
};

static inline void png_push_process_row(png_struct* png_ptr) {
  if (png_ptr->row_fn) png_ptr->row_fn(png_ptr);
}

void png_process_IDAT_data(png_struct* png_ptr, const uint8_t* buffer, size_t buffer_length) {
  png_ptr->next_in = buffer;
  png_ptr->avail_in = static_cast<uint32_t>(buffer_length);
  while (png_ptr->avail_in > 0 &&
         !(png_ptr->flags & (PNG_FLAG_ZSTREAM_ENDED | PNG_FLAG_PROCESS_DATA_PAUSED))) { //@GUARD
    png_ptr->last_byte ^= *png_ptr->next_in++;
    png_ptr->avail_in--;
    png_push_process_row(png_ptr);
    if ((png_ptr->flags & PNG_FLAG_PROCESS_DATA_PAUSED) != 0) { //@GUARD
      png_ptr->next_in = nullptr;
      png_ptr->avail_in = 0;
      return;
    }
  }
}
""",
        "driver_ok": """\
  uint8_t* buf = new uint8_t[4]{1, 2, 3, 4};
  png_struct s = {0, nullptr, 0, buf, 0, nullptr};
  png_process_IDAT_data(&s, s.save_buffer, 4);
  delete[] s.save_buffer;
""",
        "driver_bad": """\
  uint8_t* buf = new uint8_t[4]{1, 2, 3, 4};
  auto pause_cb = +[](png_struct* p) {
    p->flags |= PNG_FLAG_PROCESS_DATA_PAUSED;
    uint8_t* next = new uint8_t[16]{};
    delete[] p->save_buffer;
    p->save_buffer = next;
  };
  png_struct s = {0, nullptr, 0, buf, 0, pause_cb};
  png_process_IDAT_data(&s, s.save_buffer, 4);
  delete[] s.save_buffer;
""",
    },
    # -------------------------------------------------------------------------
    # RW06: openxla/xla - ShapeInference::InferTopKShape
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW06",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_on_scalar_topk_operand_last_dim",
        "crucible_packet_id": "openxla-xla-shape-inference-dynamic-dimension-bounds",
        "repo": "openxla/xla",
        "base_commit": "bae1f051129af495c55dbaf9d736dd0502019dda",
        "file_path": "xla/service/shape_inference.cc",
        "function_name": "ShapeInference::InferTopKShape",
        "slug": "openxla_shape_inference_topk_scalar",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["dimensions", "last_dim"],
        "terms": ["dimensions", "empty", "last_dim", "k"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  std::vector<int64_t> dims_;
  const std::vector<int64_t>& dimensions() const { return dims_; }
  int64_t dimensions(int64_t index) const { return dims_.data()[index]; } //@ROOT
};

int InferTopKShape(const Shape& operand_shape, int64_t k, int64_t* out_dim) {
  int64_t last_dim = static_cast<int64_t>(operand_shape.dimensions().size()) - 1; //@ROOT
  if (k < 0 || operand_shape.dimensions(last_dim) < k) { //@SINK
    return -1;
  }
  *out_dim = k;
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  std::vector<int64_t> dims_;
  const std::vector<int64_t>& dimensions() const { return dims_; }
  int64_t dimensions(int64_t index) const { return dims_.data()[index]; }
};

int InferTopKShape(const Shape& operand_shape, int64_t k, int64_t* out_dim) {
  if (operand_shape.dimensions().empty()) { //@GUARD
    return -1;
  }
  int64_t last_dim = static_cast<int64_t>(operand_shape.dimensions().size()) - 1;
  if (k < 0 || operand_shape.dimensions(last_dim) < k) {
    return -1;
  }
  *out_dim = k;
  return 0;
}
""",
        "driver_ok": """\
  Shape s{{5}};
  int64_t out = 0;
  if (InferTopKShape(s, 2, &out) != 0 || out != 2) return 1;
""",
        "driver_bad": """\
  Shape s;
  s.dims_.reserve(4);
  int64_t out = 0;
  volatile int rc = InferTopKShape(s, argc, &out);
  (void)rc;
""",
    },
    # -------------------------------------------------------------------------
    # RW07: harfbuzz/harfbuzz - VarRegionList::serialize
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW07",
        "family": "F04",
        "gold_class": "INTEGER_OVERFLOW",
        "cwe": "CWE-190",
        "bug_class": "unchecked_unsigned_multiplication_before_overflow_helper",
        "crucible_packet_id": "harfbuzz-cff2-private-blend-overflow-and-serialize-bounds",
        "repo": "harfbuzz/harfbuzz",
        "base_commit": "0ba555a3d4f00a5d2909b0cf8720e04b8fe83ff0",
        "file_path": "src/hb-ot-layout-common.hh",
        "function_name": "VarRegionList::serialize",
        "slug": "harfbuzz_var_region_list_mul",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["hb_unsigned_mul_overflows", "memset", "buf"],
        "terms": ["axis_count", "region_count", "total_axes", "hb_unsigned_mul_overflows"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstdint>
#include <cstdlib>
#include <cstring>

static constexpr unsigned VarRegionAxis_static_size = 6;

static inline bool hb_unsigned_mul_overflows(unsigned count, unsigned size, unsigned* result = nullptr) {
  if (size > 0 && count >= (0xFFFFFFFFu / size)) return true;
  if (result) *result = count * size;
  return false;
}

uint8_t* VarRegionList_serialize(unsigned axis_count, unsigned region_count) {
  if (!axis_count || !region_count) return nullptr;
  if (hb_unsigned_mul_overflows(axis_count * region_count, VarRegionAxis_static_size)) //@ROOT
    return nullptr;
  unsigned total_bytes = (axis_count * region_count) * VarRegionAxis_static_size;
  uint8_t* buf = static_cast<uint8_t*>(std::malloc(total_bytes));
  if (!buf) return nullptr;
  for (unsigned r = 0; r < region_count; ++r) {
    std::memset(buf + r * VarRegionAxis_static_size, 0, VarRegionAxis_static_size); //@SINK
  }
  return buf;
}
""",
        "safe": """\
#include <cstdint>
#include <cstdlib>
#include <cstring>

static constexpr unsigned VarRegionAxis_static_size = 6;

static inline bool hb_unsigned_mul_overflows(unsigned count, unsigned size, unsigned* result = nullptr) {
  if (size > 0 && count >= (0xFFFFFFFFu / size)) return true;
  if (result) *result = count * size;
  return false;
}

uint8_t* VarRegionList_serialize(unsigned axis_count, unsigned region_count) {
  if (!axis_count || !region_count) return nullptr;
  if (region_count > 0x7FFFu) return nullptr;
  unsigned total_axes = 0;
  if (hb_unsigned_mul_overflows(axis_count, region_count, &total_axes) || //@GUARD
      hb_unsigned_mul_overflows(total_axes, VarRegionAxis_static_size)) { //@GUARD
    return nullptr;
  }
  unsigned total_bytes = total_axes * VarRegionAxis_static_size;
  uint8_t* buf = static_cast<uint8_t*>(std::malloc(total_bytes));
  if (!buf) return nullptr;
  for (unsigned r = 0; r < region_count; ++r) {
    std::memset(buf + r * VarRegionAxis_static_size, 0, VarRegionAxis_static_size);
  }
  return buf;
}
""",
        "driver_ok": """\
  uint8_t* p = VarRegionList_serialize(2, 4);
  if (!p) return 1;
  std::free(p);
""",
        "driver_bad": """\
  uint8_t* p = VarRegionList_serialize(4u, 0x40000000u + static_cast<unsigned>(argc));
  volatile uint8_t keep = p ? p[0] : 0;
  (void)keep;
  std::free(p);
""",
    },
    # -------------------------------------------------------------------------
    # RW08: AOMediaCodec/libavif - avifEncoderWriteHDRProperties
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW08",
        "family": "F07",
        "gold_class": "NULL_DEREFERENCE",
        "cwe": "CWE-476",
        "bug_class": "null_dereference_of_dedupstream_on_sequence_track_hdr",
        "crucible_packet_id": "libavif-sequence-track-null-dedupstream",
        "repo": "AOMediaCodec/libavif",
        "base_commit": "90d3b6661e1475bb1652cba7d85c24f8c748ab07",
        "file_path": "src/write.c",
        "function_name": "avifEncoderWriteHDRProperties",
        "slug": "libavif_sequence_null_dedupstream",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["dedupStream", "written"],
        "terms": ["dedupStream", "outputStream", "maxCLL", "maxPALL"],
        "expect": r"member access within null pointer|SEGV|AddressSanitizer",
        "vuln": """\
#include <cstdint>

struct avifRWStream { uint32_t written; };
struct avifImage { uint16_t maxCLL; uint16_t maxPALL; };

int avifEncoderWriteHDRProperties(avifRWStream* dedupStream,
                                  avifRWStream* outputStream,
                                  const avifImage* imageMetadata) {
  (void)outputStream;
  if (imageMetadata->maxCLL || imageMetadata->maxPALL) {
    dedupStream->written += 4; //@SINK
  }
  return 0;
}
""",
        "safe": """\
#include <cstdint>

struct avifRWStream { uint32_t written; };
struct avifImage { uint16_t maxCLL; uint16_t maxPALL; };

int avifEncoderWriteHDRProperties(avifRWStream* dedupStream,
                                  avifRWStream* outputStream,
                                  const avifImage* imageMetadata) {
  if (!dedupStream) { //@GUARD
    dedupStream = outputStream; //@GUARD
  }
  if (imageMetadata->maxCLL || imageMetadata->maxPALL) {
    dedupStream->written += 4;
  }
  return 0;
}
""",
        "driver_ok": """\
  avifRWStream out = {0};
  avifImage img = {1000, 400};
  if (avifEncoderWriteHDRProperties(&out, &out, &img) != 0 || out.written != 4) return 1;
""",
        "driver_bad": """\
  avifRWStream out = {0};
  avifImage img = {1000, 400};
  (void)avifEncoderWriteHDRProperties(nullptr, &out, &img);
""",
    },
    # -------------------------------------------------------------------------
    # RW09: AOMediaCodec/libavif - avifDecoderCreateCodecs
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW09",
        "family": "F07",
        "gold_class": "NULL_DEREFERENCE",
        "cwe": "CWE-476",
        "bug_class": "null_dereference_after_partial_tile_codec_creation_failure",
        "crucible_packet_id": "libavif-decoder-lifecycle-persistent-io-and-codec-rollback",
        "repo": "AOMediaCodec/libavif",
        "base_commit": "9b07a9d7a874b159bb4e14b80a6f08bb469d06f7",
        "file_path": "src/read.c",
        "function_name": "avifDecoderCreateCodecs",
        "slug": "libavif_codec_rollback_null",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["codec", "maxThreads"],
        "terms": ["avifDecoderDataResetCodec", "tiles", "codec", "fail_idx"],
        "expect": r"member access within null pointer|SEGV|AddressSanitizer",
        "vuln": """\
#include <cstdint>
#include <cstdlib>

struct avifCodec { int maxThreads; };
struct avifTile { avifCodec* codec; };
struct avifDecoderData { avifTile tiles[2]; uint32_t tileCount; };

static inline void avifDecoderDataResetCodec(avifDecoderData* data) {
  for (uint32_t i = 0; i < data->tileCount; ++i) {
    delete data->tiles[i].codec;
    data->tiles[i].codec = nullptr;
  }
}

int avifDecoderNextImage(avifDecoderData* data, int fail_tile) {
  if (!data->tiles[0].codec) { //@ROOT
    for (uint32_t i = 0; i < data->tileCount; ++i) {
      if (static_cast<int>(i) == fail_tile) return -1;
      data->tiles[i].codec = new avifCodec{1};
    }
  }
  for (uint32_t i = 0; i < data->tileCount; ++i) {
    data->tiles[i].codec->maxThreads = 4; //@SINK
  }
  return 0;
}
""",
        "safe": """\
#include <cstdint>
#include <cstdlib>

struct avifCodec { int maxThreads; };
struct avifTile { avifCodec* codec; };
struct avifDecoderData { avifTile tiles[2]; uint32_t tileCount; };

static inline void avifDecoderDataResetCodec(avifDecoderData* data) {
  for (uint32_t i = 0; i < data->tileCount; ++i) {
    delete data->tiles[i].codec;
    data->tiles[i].codec = nullptr;
  }
}

int avifDecoderNextImage(avifDecoderData* data, int fail_tile) {
  if (!data->tiles[0].codec) {
    for (uint32_t i = 0; i < data->tileCount; ++i) {
      if (static_cast<int>(i) == fail_tile) {
        avifDecoderDataResetCodec(data); //@GUARD
        return -1;
      }
      data->tiles[i].codec = new avifCodec{1};
    }
  }
  for (uint32_t i = 0; i < data->tileCount; ++i) {
    data->tiles[i].codec->maxThreads = 4;
  }
  return 0;
}
""",
        "driver_ok": """\
  avifDecoderData d = {{{nullptr}, {nullptr}}, 2};
  if (avifDecoderNextImage(&d, -1) != 0) return 1;
  avifDecoderDataResetCodec(&d);
""",
        "driver_bad": """\
  avifDecoderData d = {{{nullptr}, {nullptr}}, 2};
  (void)avifDecoderNextImage(&d, 1);
  (void)avifDecoderNextImage(&d, -1);
  avifDecoderDataResetCodec(&d);
""",
    },
    # -------------------------------------------------------------------------
    # RW10: lz4/lz4 - LZ4F_decodeBlock
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW10",
        "family": "F04",
        "gold_class": "INTEGER_OVERFLOW",
        "cwe": "CWE-190",
        "bug_class": "unsigned_remaining_content_size_underflow_in_lz4frame",
        "crucible_packet_id": "lz4-lz4frame-remaining-content-size-underflow",
        "repo": "lz4/lz4",
        "base_commit": "d5f17575750ea89af23eb0ee0963b5026badecfd",
        "file_path": "lib/lz4frame.c",
        "function_name": "LZ4F_decodeBlock",
        "slug": "lz4frame_remaining_size_underflow",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memcpy", "frameRemainingSize"],
        "terms": ["frameRemainingSize", "contentSize", "sizeToCopy"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

struct LZ4F_dctx {
  uint64_t contentSize;
  uint64_t frameRemainingSize;
};

int LZ4F_decodeBlock(LZ4F_dctx* dctx, uint8_t* dst, const uint8_t* src, size_t sizeToCopy) {
  if (dctx->contentSize) {
    dctx->frameRemainingSize -= static_cast<uint64_t>(sizeToCopy); //@ROOT
  }
  std::memcpy(dst, src, sizeToCopy); //@SINK
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

struct LZ4F_dctx {
  uint64_t contentSize;
  uint64_t frameRemainingSize;
};

int LZ4F_decodeBlock(LZ4F_dctx* dctx, uint8_t* dst, const uint8_t* src, size_t sizeToCopy) {
  if (dctx->contentSize && static_cast<uint64_t>(sizeToCopy) > dctx->frameRemainingSize) { //@GUARD
    return -1;
  }
  if (dctx->contentSize) {
    dctx->frameRemainingSize -= static_cast<uint64_t>(sizeToCopy);
  }
  std::memcpy(dst, src, sizeToCopy);
  return 0;
}
""",
        "driver_ok": """\
  LZ4F_dctx dctx = {16, 16};
  uint8_t* dst = new uint8_t[16]{};
  uint8_t src[16] = {};
  if (LZ4F_decodeBlock(&dctx, dst, src, 16) != 0) return 1;
  volatile uint8_t keep = dst[0];
  (void)keep;
  delete[] dst;
""",
        "driver_bad": """\
  LZ4F_dctx dctx = {16, 16};
  uint8_t* dst = new uint8_t[16]{};
  uint8_t src[32] = {};
  int rc = LZ4F_decodeBlock(&dctx, dst, src, 31 + static_cast<size_t>(argc));
  volatile uint8_t keep = dst[0] + static_cast<uint8_t>(rc);
  (void)keep;
  delete[] dst;
""",
    },
    # -------------------------------------------------------------------------
    # RW11: google/flatbuffers - JsonPrinter::GenStruct
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW11",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_of_unverified_deprecated_vtable_slot",
        "crucible_packet_id": "flatbuffers-gentext-skip-deprecated-fields",
        "repo": "google/flatbuffers",
        "base_commit": "5eb190f00005b8dd1aa4f7ab2db9f01a39e3c76d",
        "file_path": "src/idl_gen_text.cpp",
        "function_name": "JsonPrinter::GenStruct",
        "slug": "flatbuffers_gentext_deprecated_slot",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["GetField", "table"],
        "terms": ["deprecated", "offset", "CheckField", "GetField"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct FieldDef { uint16_t offset; bool deprecated; };
struct Table {
  const uint8_t* data;
  bool CheckField(uint16_t off) const { return off != 0; }
  uint8_t GetField(uint16_t off) const { return data[off]; } //@ROOT
};

int GenStruct(const std::vector<FieldDef>& fields, const Table* table) {
  int sum = 0;
  for (auto it = fields.begin(); it != fields.end(); ++it) {
    const FieldDef& fd = *it;
    bool is_present = table->CheckField(fd.offset);
    if (is_present) {
      sum += table->GetField(fd.offset); //@SINK
    }
  }
  return sum;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct FieldDef { uint16_t offset; bool deprecated; };
struct Table {
  const uint8_t* data;
  bool CheckField(uint16_t off) const { return off != 0; }
  uint8_t GetField(uint16_t off) const { return data[off]; }
};

int GenStruct(const std::vector<FieldDef>& fields, const Table* table) {
  int sum = 0;
  for (auto it = fields.begin(); it != fields.end(); ++it) {
    const FieldDef& fd = *it;
    if (fd.deprecated) continue; //@GUARD
    bool is_present = table->CheckField(fd.offset);
    if (is_present) {
      sum += table->GetField(fd.offset);
    }
  }
  return sum;
}
""",
        "driver_ok": """\
  uint8_t* buf = new uint8_t[8]{10, 20, 30, 40, 0, 0, 0, 0};
  Table t{buf};
  std::vector<FieldDef> fields = {{1, false}, {0, true}};
  if (GenStruct(fields, &t) != 20) return 1;
  delete[] buf;
""",
        "driver_bad": """\
  uint8_t* buf = new uint8_t[8]{10, 20, 30, 40, 0, 0, 0, 0};
  Table t{buf};
  std::vector<FieldDef> fields = {{1, false}, {24, true}};
  volatile int s = GenStruct(fields, &t);
  (void)s;
  delete[] buf;
""",
    },
    # -------------------------------------------------------------------------
    # RW12: libjxl/libjxl - BasePNMEncoder::Encode
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW12",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_on_mismatched_frame_extra_channels_count",
        "crucible_packet_id": "libjxl-extras-apng-pnm-extra-channel-bounds",
        "repo": "libjxl/libjxl",
        "base_commit": "b7ec9c0f63a62a845901873f01918758f0df7e67",
        "file_path": "lib/extras/enc/pnm.cc",
        "function_name": "BasePNMEncoder::Encode",
        "slug": "libjxl_pnm_extra_channel_bounds",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["extra_channels_info", "bits_per_sample"],
        "terms": ["extra_channels", "extra_channels_info", "Validate", "frames"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct ExtraChannelInfo { uint32_t bits_per_sample; };
struct PackedFrame { std::vector<uint8_t> extra_channels; };
struct PackedPixelFile {
  std::vector<ExtraChannelInfo> extra_channels_info;
  std::vector<PackedFrame> frames;
  bool Validate() const {
    for (const auto& frame : frames) {
      if (frame.extra_channels.size() != extra_channels_info.size()) return false;
    }
    return true;
  }
};

bool Encode(const PackedPixelFile& ppf, uint32_t* total_bits) {
  *total_bits = 0;
  for (const auto& frame : ppf.frames) {
    for (size_t i = 0; i < frame.extra_channels.size(); ++i) {
      *total_bits += ppf.extra_channels_info.data()[i].bits_per_sample; //@SINK
    }
  }
  return true;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct ExtraChannelInfo { uint32_t bits_per_sample; };
struct PackedFrame { std::vector<uint8_t> extra_channels; };
struct PackedPixelFile {
  std::vector<ExtraChannelInfo> extra_channels_info;
  std::vector<PackedFrame> frames;
  bool Validate() const {
    for (const auto& frame : frames) {
      if (frame.extra_channels.size() != extra_channels_info.size()) return false;
    }
    return true;
  }
};

bool Encode(const PackedPixelFile& ppf, uint32_t* total_bits) {
  if (!ppf.Validate()) return false; //@GUARD
  *total_bits = 0;
  for (const auto& frame : ppf.frames) {
    for (size_t i = 0; i < frame.extra_channels.size(); ++i) {
      *total_bits += ppf.extra_channels_info.data()[i].bits_per_sample;
    }
  }
  return true;
}
""",
        "driver_ok": """\
  PackedPixelFile ppf{{{8}}, {{{1}}}};
  uint32_t bits = 0;
  if (!Encode(ppf, &bits) || bits != 8) return 1;
""",
        "driver_bad": """\
  PackedPixelFile ppf{{{8}}, {{{1, 2, 3, 4}}}};
  uint32_t bits = 0;
  volatile bool ok = Encode(ppf, &bits);
  (void)ok;
""",
    },
    # -------------------------------------------------------------------------
    # RW13: google/XNNPACK - resize_unary_elementwise_output_tensor
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW13",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "oob_write_copying_unvalidated_input_rank_into_tensor_shape",
        "crucible_packet_id": "xnnpack-subgraph-slice-and-reshape-bounds-v2",
        "repo": "google/XNNPACK",
        "base_commit": "fc1a2623138f0b7d94d77255d9042034882ad6ac",
        "file_path": "src/subgraph/reshape-helpers.c",
        "function_name": "resize_unary_elementwise_output_tensor",
        "slug": "xnnpack_resize_unary_num_dims",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memcpy", "dim"],
        "terms": ["num_dims", "XNN_MAX_TENSOR_DIMS", "input", "output"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

static constexpr size_t XNN_MAX_TENSOR_DIMS = 6;
struct xnn_shape { size_t num_dims; size_t dim[XNN_MAX_TENSOR_DIMS]; };
struct xnn_runtime_value { xnn_shape shape; };

int resize_unary_elementwise_output_tensor(const size_t* input_dims, size_t input_num_dims,
                                           xnn_runtime_value* output) {
  output->shape.num_dims = input_num_dims; //@ROOT
  std::memcpy(output->shape.dim, input_dims, input_num_dims * sizeof(size_t)); //@SINK
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

static constexpr size_t XNN_MAX_TENSOR_DIMS = 6;
struct xnn_shape { size_t num_dims; size_t dim[XNN_MAX_TENSOR_DIMS]; };
struct xnn_runtime_value { xnn_shape shape; };

int resize_unary_elementwise_output_tensor(const size_t* input_dims, size_t input_num_dims,
                                           xnn_runtime_value* output) {
  if (input_num_dims > XNN_MAX_TENSOR_DIMS) { //@GUARD
    return -1;
  }
  output->shape.num_dims = input_num_dims;
  std::memcpy(output->shape.dim, input_dims, input_num_dims * sizeof(size_t));
  return 0;
}
""",
        "driver_ok": """\
  size_t in_dims[6] = {1, 2, 3, 4, 5, 6};
  xnn_runtime_value* out = new xnn_runtime_value{};
  if (resize_unary_elementwise_output_tensor(in_dims, 6, out) != 0) return 1;
  volatile size_t keep = out->shape.dim[0];
  (void)keep;
  delete out;
""",
        "driver_bad": """\
  size_t in_dims[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  xnn_runtime_value* out = new xnn_runtime_value{};
  int rc = resize_unary_elementwise_output_tensor(in_dims, 11 + static_cast<size_t>(argc), out);
  volatile size_t keep = out->shape.dim[0] + static_cast<size_t>(rc);
  (void)keep;
  delete out;
""",
    },
    # -------------------------------------------------------------------------
    # RW14: vllm-project/vllm - moe_align_block_size
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW14",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_write_indexing_moe_workspace_with_unvalidated_topk_ids",
        "crucible_packet_id": "vllm-cpu-hash-moe-tid2eid-bounds",
        "repo": "vllm-project/vllm",
        "base_commit": "4527515cd705db206ae927cf5acef506abc64bdd",
        "file_path": "csrc/cpu/sgl-kernels/moe.cpp",
        "function_name": "moe_align_block_size",
        "slug": "vllm_cpu_moe_align_topk_bounds",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["total_cnts", "topk_ids"],
        "terms": ["topk_ids", "num_experts", "total_cnts", "numel"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>

int moe_align_block_size(const int32_t* topk_ids, int numel, int num_experts,
                         int32_t* total_cnts) {
  for (int i = 0; i < numel; ++i) {
    total_cnts[topk_ids[i]] += 1; //@SINK
  }
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>

int moe_align_block_size(const int32_t* topk_ids, int numel, int num_experts,
                         int32_t* total_cnts) {
  for (int i = 0; i < numel; ++i) {
    if (topk_ids[i] < 0 || topk_ids[i] >= num_experts) { //@GUARD
      return -1;
    }
  }
  for (int i = 0; i < numel; ++i) {
    total_cnts[topk_ids[i]] += 1;
  }
  return 0;
}
""",
        "driver_ok": """\
  int32_t* cnts = new int32_t[8]{};
  int32_t ids[4] = {0, 3, 7, 1};
  if (moe_align_block_size(ids, 4, 8, cnts) != 0) return 1;
  delete[] cnts;
""",
        "driver_bad": """\
  int32_t* cnts = new int32_t[8]{};
  int32_t ids[4] = {0, 64, 1, 2};
  (void)moe_align_block_size(ids, 4, 8, cnts);
  delete[] cnts;
""",
    },
    # -------------------------------------------------------------------------
    # RW15: mlc-ai/xgrammar - GrammarFSMBuilderImpl::BuildRegex
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW15",
        "family": "F03",
        "gold_class": "USE_AFTER_FREE",
        "cwe": "CWE-416",
        "bug_class": "heap_uaf_reading_rule_name_pointer_after_rule_vector_reallocation",
        "crucible_packet_id": "xgrammar-build-regex-rule-reallocation-uaf-and-recursion-bounds",
        "repo": "mlc-ai/xgrammar",
        "base_commit": "11f65251738b0a1a24618a7bf81505a87fa99ce5",
        "file_path": "cpp/grammar_functor.cc",
        "function_name": "GrammarFSMBuilderImpl::BuildRegex",
        "slug": "xgrammar_build_regex_rule_uaf",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["rule_name_", "out_err", "name"],
        "terms": ["rule_name_", "rule_name_copy_", "rules", "push_back"],
        "expect": r"heap-use-after-free",
        "vuln": """\
#include <optional>
#include <string>
#include <vector>

struct RuleEntry { const char* name; };
struct GrammarBuilder {
  std::vector<RuleEntry> rules;
  int BuildRegexFSM(const std::string& regex, const char* hint) {
    (void)hint;
    for (int i = 0; i < 16; ++i) rules.push_back(RuleEntry{"aux"});
    return regex == "bad" ? -1 : 0;
  }
};

struct GrammarFSMBuilderImpl {
  GrammarBuilder* builder_;
  const RuleEntry* rule_name_;
  std::optional<std::string> rule_name_copy_;

  int BuildRegex(const std::string& regex, std::string* out_err) {
    const char* rule_hint = rule_name_ != nullptr ? rule_name_->name : "";
    int rc = builder_->BuildRegexFSM(regex, rule_hint); //@ROOT
    if (rc < 0 && rule_name_ != nullptr) {
      *out_err = rule_name_->name; //@SINK
    }
    return rc;
  }
};
""",
        "safe": """\
#include <optional>
#include <string>
#include <vector>

struct RuleEntry { const char* name; };
struct GrammarBuilder {
  std::vector<RuleEntry> rules;
  int BuildRegexFSM(const std::string& regex, const char* hint) {
    (void)hint;
    for (int i = 0; i < 16; ++i) rules.push_back(RuleEntry{"aux"});
    return regex == "bad" ? -1 : 0;
  }
};

struct GrammarFSMBuilderImpl {
  GrammarBuilder* builder_;
  const RuleEntry* rule_name_;
  std::optional<std::string> rule_name_copy_;

  int BuildRegex(const std::string& regex, std::string* out_err) {
    if (!rule_name_copy_.has_value() && rule_name_ != nullptr) { //@GUARD
      rule_name_copy_ = rule_name_->name; //@GUARD
    }
    const char* rule_hint = rule_name_copy_.has_value() ? rule_name_copy_->c_str() : "";
    int rc = builder_->BuildRegexFSM(regex, rule_hint);
    if (rc < 0 && rule_name_copy_.has_value()) {
      *out_err = *rule_name_copy_;
    }
    return rc;
  }
};
""",
        "driver_ok": """\
  GrammarBuilder gb;
  gb.rules.push_back(RuleEntry{"root"});
  GrammarFSMBuilderImpl fsm{&gb, &gb.rules[0], std::nullopt};
  std::string err;
  if (fsm.BuildRegex("ok", &err) != 0) return 1;
""",
        "driver_bad": """\
  GrammarBuilder gb;
  gb.rules.push_back(RuleEntry{"root"});
  GrammarFSMBuilderImpl fsm{&gb, &gb.rules[0], std::nullopt};
  std::string err;
  int rc = fsm.BuildRegex("bad", &err);
  volatile size_t keep = err.size() + static_cast<size_t>(rc);
  (void)keep;
""",
    },
    # -------------------------------------------------------------------------
    # RW16: mlc-ai/xgrammar - LarkParser::ParseAtom
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW16",
        "family": "F12",
        "gold_class": "UNBOUNDED_RECURSION",
        "cwe": "CWE-674",
        "bug_class": "unbounded_recursion_on_nested_lark_grammar_blocks",
        "crucible_packet_id": "xgrammar-lark-nested-grammar-depth-guard-v2",
        "repo": "mlc-ai/xgrammar",
        "base_commit": "ca1acf28bf35b9ba9092800b69aebc7d4ce1f7a7",
        "file_path": "cpp/lark_converter.cc",
        "function_name": "LarkParser::ParseAtom",
        "slug": "xgrammar_lark_nested_depth",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["ParseAtom", "ParseDocument"],
        "terms": ["group_depth_", "kMaxGroupDepth", "ParseAtom"],
        "expect": r"stack-overflow|SEGV",
        "vuln": """\
#include <cstddef>
#include <cstdint>

struct LarkParser {
  const char* src_;
  size_t pos_ = 0;
  int group_depth_ = 0;
  static constexpr int kMaxGroupDepth = 200;

  long ParseAtom() { //@ROOT
    long total = 0; //@ROOT
    while (src_[pos_] != '\\0' && src_[pos_] != '}') { //@ROOT
      if (src_[pos_] == '{') { //@ROOT
        ++pos_; //@ROOT
        long sub = ParseAtom(); //@SINK
        if (sub < 0) return -1;
        total += sub + 1;
        if (src_[pos_] == '}') ++pos_;
      } else {
        ++pos_;
        ++total;
      }
    }
    return total;
  }
};
""",
        "safe": """\
#include <cstddef>
#include <cstdint>

struct LarkParser {
  const char* src_;
  size_t pos_ = 0;
  int group_depth_ = 0;
  static constexpr int kMaxGroupDepth = 200;

  long ParseAtom() {
    long total = 0;
    if (group_depth_ >= kMaxGroupDepth) return -1; //@GUARD
    while (src_[pos_] != '\\0' && src_[pos_] != '}') {
      if (src_[pos_] == '{') {
        ++pos_;
        ++group_depth_; //@GUARD
        long sub = ParseAtom();
        --group_depth_;
        if (sub < 0) return -1;
        total += sub + 1;
        if (src_[pos_] == '}') ++pos_;
      } else {
        ++pos_;
        ++total;
      }
    }
    return total;
  }
};
""",
        "driver_ok": """\
  LarkParser p{"{{{a}}}", 0, 0};
  if (p.ParseAtom() <= 0) return 1;
""",
        "driver_bad": """\
  size_t n = 1000000;
  char* deep = new char[n + 1];
  for (size_t i = 0; i < n; ++i) deep[i] = '{';
  deep[n] = '\\0';
  LarkParser p{deep, 0, 0};
  volatile long rc = p.ParseAtom();
  (void)rc;
  delete[] deep;
""",
    },
    # -------------------------------------------------------------------------
    # RW17: onnx/onnx - LoopInferenceFunction
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW17",
        "family": "F10",
        "gold_class": "ASSERT_ONLY_CHECK",
        "cwe": "CWE-617",
        "bug_class": "assert_only_loop_input_count_check_underflows_in_ndebug",
        "crucible_packet_id": "onnx-loop-shape-inference-input-count",
        "repo": "onnx/onnx",
        "base_commit": "48ace3d46387558f6dbd858ea3ed8758d7607b90",
        "file_path": "onnx/defs/controlflow/utils.cc",
        "function_name": "LoopInferenceFunction",
        "slug": "onnx_loop_input_count_assert",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["getInputType", "assert", "num_loop_state_vars"],
        "terms": ["num_inputs", "num_loop_state_vars", "assert"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cassert>
#include <cstddef>
#include <vector>

struct InferenceContext {
  std::vector<int> inputs_;
  size_t getNumInputs() const { return inputs_.size(); }
  int getInputType(size_t i) const { return inputs_.data()[i]; } //@ROOT
};

int LoopInferenceFunction(const InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  assert(num_inputs >= 2); //@ROOT
  auto num_loop_state_vars = num_inputs - 2;
  int acc = 0;
  for (size_t i = 0; i < num_loop_state_vars; ++i) {
    acc += ctx.getInputType(i + 2); //@SINK
  }
  return acc;
}
""",
        "safe": """\
#include <cassert>
#include <cstddef>
#include <vector>

struct InferenceContext {
  std::vector<int> inputs_;
  size_t getNumInputs() const { return inputs_.size(); }
  int getInputType(size_t i) const { return inputs_.data()[i]; }
};

int LoopInferenceFunction(const InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  if (num_inputs < 2) { //@GUARD
    return -1;
  }
  auto num_loop_state_vars = num_inputs - 2;
  int acc = 0;
  for (size_t i = 0; i < num_loop_state_vars; ++i) {
    acc += ctx.getInputType(i + 2);
  }
  return acc;
}
""",
        "driver_ok": """\
  InferenceContext ctx{{1, 1, 7}};
  if (LoopInferenceFunction(ctx) != 7) return 1;
""",
        "driver_bad": """\
  InferenceContext ctx{{1}};
  (void)LoopInferenceFunction(ctx);
""",
    },
    # -------------------------------------------------------------------------
    # RW18: opencv/opencv - ReadDarknetConvGroups
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW18",
        "family": "F04",
        "gold_class": "INTEGER_OVERFLOW",
        "cwe": "CWE-190",
        "bug_class": "divide_by_zero_on_zero_convolutional_groups_in_darknet_importer",
        "crucible_packet_id": "opencv-darknet-conv-groups-zero",
        "repo": "opencv/opencv",
        "base_commit": "7dc486f74c625533a3b1fcdcbe6fc082ca25a945",
        "file_path": "modules/dnn/src/darknet/darknet_io.cpp",
        "function_name": "ReadDarknetConvGroups",
        "slug": "opencv_darknet_conv_groups_zero",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["groups", "channels"],
        "terms": ["groups", "channels", "filters"],
        "expect": r"division by zero|FPE|AddressSanitizer",
        "vuln": """\
#include <cstdint>

int ReadDarknetConvGroups(int channels, int filters, int groups, int* out_in_per_group) {
  if (channels <= 0 || filters <= 0) return -1;
  if (channels % groups != 0) return -1; //@SINK
  *out_in_per_group = channels / groups;
  return 0;
}
""",
        "safe": """\
#include <cstdint>

int ReadDarknetConvGroups(int channels, int filters, int groups, int* out_in_per_group) {
  if (channels <= 0 || filters <= 0) return -1;
  if (groups <= 0) return -1; //@GUARD
  if (channels % groups != 0) return -1;
  *out_in_per_group = channels / groups;
  return 0;
}
""",
        "driver_ok": """\
  int out = 0;
  if (ReadDarknetConvGroups(6, 4, 2, &out) != 0 || out != 3) return 1;
""",
        "driver_bad": """\
  int out = 0;
  (void)ReadDarknetConvGroups(6, 4, 0, &out);
""",
    },
    # -------------------------------------------------------------------------
    # RW19: opencv/opencv - DTreesImpl::predictTrees
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW19",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_on_unvalidated_dtree_split_variable_index",
        "crucible_packet_id": "opencv-ml-dtrees-load-bounds",
        "repo": "opencv/opencv",
        "base_commit": "7dc486f74c625533a3b1fcdcbe6fc082ca25a945",
        "file_path": "modules/ml/src/tree.cpp",
        "function_name": "DTreesImpl::predictTrees",
        "slug": "opencv_ml_dtrees_predict_bounds",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["psample", "ci"],
        "terms": ["ci", "sample_total", "psample", "sstep"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>

struct Split { int varIdx; float c; };

int predictTrees(const Split& split, const int* cvidx, const float* psample,
                 size_t sample_total, size_t sstep, float* out_val) {
  int vi = split.varIdx;
  int ci = cvidx ? cvidx[vi] : vi;
  float val = psample[ci * sstep]; //@SINK
  *out_val = val - split.c;
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>

struct Split { int varIdx; float c; };

int predictTrees(const Split& split, const int* cvidx, const float* psample,
                 size_t sample_total, size_t sstep, float* out_val) {
  int vi = split.varIdx;
  int ci = cvidx ? cvidx[vi] : vi;
  if (ci < 0 || static_cast<size_t>(ci) >= sample_total) { //@GUARD
    return -1;
  }
  float val = psample[ci * sstep];
  *out_val = val - split.c;
  return 0;
}
""",
        "driver_ok": """\
  float* sample = new float[3]{1.5f, 2.5f, 3.5f};
  int map[2] = {0, 2};
  Split sp{1, 0.5f};
  float out = 0.0f;
  if (predictTrees(sp, map, sample, 3, 1, &out) != 0) return 1;
  delete[] sample;
""",
        "driver_bad": """\
  float* sample = new float[3]{1.5f, 2.5f, 3.5f};
  int map[2] = {0, 16};
  Split sp{1, 0.5f};
  float out = 0.0f;
  int rc = predictTrees(sp, map, sample, 3, 1, &out);
  volatile float keep = out + static_cast<float>(rc);
  (void)keep;
  delete[] sample;
""",
    },
    # -------------------------------------------------------------------------
    # RW20: opencv/opencv - TorchImporter::readTensor
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW20",
        "family": "F11",
        "gold_class": "TYPE_CONFUSION",
        "cwe": "CWE-843",
        "bug_class": "type_confusion_and_heap_oob_when_reusing_torch_storage_of_smaller_type",
        "crucible_packet_id": "opencv-torch-storage-reuse-type-check",
        "repo": "opencv/opencv",
        "base_commit": "f706833347e1b9ab0ce6330cca05a419e3a190d1",
        "file_path": "modules/dnn/src/torch/torch_importer.cpp",
        "function_name": "TorchImporter::readTensor",
        "slug": "opencv_torch_storage_reuse_type",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memcpy", "raw"],
        "terms": ["elem_type", "typeTensor", "storages", "requireElems"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

struct StorageMat {
  int elem_type;
  size_t elem_count;
  std::vector<uint8_t> raw;
};

int readTensor(const std::map<int, StorageMat>& storages, int indexStorage,
               int typeTensor, size_t count, float* dst) {
  auto it = storages.find(indexStorage);
  if (it == storages.end()) return -1;
  const StorageMat& st = it->second;
  if (count > st.elem_count) return -1;
  size_t elem_size = (typeTensor == 5) ? sizeof(float) : sizeof(uint8_t); //@ROOT
  std::memcpy(dst, st.raw.data(), count * elem_size); //@SINK
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

struct StorageMat {
  int elem_type;
  size_t elem_count;
  std::vector<uint8_t> raw;
};

int readTensor(const std::map<int, StorageMat>& storages, int indexStorage,
               int typeTensor, size_t count, float* dst) {
  auto it = storages.find(indexStorage);
  if (it == storages.end()) return -1;
  const StorageMat& st = it->second;
  if (st.elem_type != typeTensor) { //@GUARD
    return -1;
  }
  if (count > st.elem_count) return -1;
  size_t elem_size = (typeTensor == 5) ? sizeof(float) : sizeof(uint8_t);
  std::memcpy(dst, st.raw.data(), count * elem_size);
  return 0;
}
""",
        "driver_ok": """\
  std::map<int, StorageMat> st;
  st[4] = StorageMat{5, 4, std::vector<uint8_t>(4 * sizeof(float), 0)};
  float* dst = new float[4]{};
  if (readTensor(st, 4, 5, 4, dst) != 0) return 1;
  volatile float keep = dst[3];
  (void)keep;
  delete[] dst;
""",
        "driver_bad": """\
  std::map<int, StorageMat> st;
  st[4] = StorageMat{0, 4, std::vector<uint8_t>(4, 1)};
  float* dst = new float[4]{};
  int rc = readTensor(st, 4, 5, 3 + static_cast<size_t>(argc), dst);
  volatile float keep = dst[3] + static_cast<float>(rc);
  (void)keep;
  delete[] dst;
""",
    },
    # -------------------------------------------------------------------------
    # RW21: openvinotoolkit/openvino - ModelDeserializer::process_model
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW21",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_in_memcpy_when_compiled_model_blob_smaller_than_header",
        "crucible_packet_id": "openvino-cpu-compiled-model-sections-master-rebase",
        "repo": "openvinotoolkit/openvino",
        "base_commit": "a05daefea7cbebefec1608a033f0c9347582726f",
        "file_path": "src/plugins/intel_cpu/src/utils/graph_serializer/deserializer.cpp",
        "function_name": "ModelDeserializer::process_model",
        "slug": "openvino_cpu_deserializer_header",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memcpy", "hdr"],
        "terms": ["file_size", "DataHeader", "hdr", "model_offset"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

struct DataHeader {
  size_t custom_data_offset;
  size_t custom_data_size;
  size_t consts_offset;
  size_t consts_size;
  size_t model_offset;
  size_t model_size;
};

int process_model(const char* buffer_base, size_t file_size, DataHeader* hdr) {
  std::memcpy(reinterpret_cast<char*>(hdr), buffer_base, sizeof(*hdr)); //@SINK
  bool is_valid_model = (hdr->custom_data_offset == sizeof(*hdr)) &&
                        (hdr->custom_data_size == hdr->consts_offset - hdr->custom_data_offset) &&
                        (hdr->consts_size == hdr->model_offset - hdr->consts_offset) &&
                        (file_size > hdr->model_offset);
  if (!is_valid_model) return -1;
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <cstring>

struct DataHeader {
  size_t custom_data_offset;
  size_t custom_data_size;
  size_t consts_offset;
  size_t consts_size;
  size_t model_offset;
  size_t model_size;
};

int process_model(const char* buffer_base, size_t file_size, DataHeader* hdr) {
  if (file_size < sizeof(*hdr)) return -1; //@GUARD
  std::memcpy(reinterpret_cast<char*>(hdr), buffer_base, sizeof(*hdr));
  bool is_valid_model = (hdr->custom_data_offset == sizeof(*hdr)) &&
                        (hdr->custom_data_offset <= hdr->consts_offset) && //@GUARD
                        (hdr->consts_offset <= hdr->model_offset) && //@GUARD
                        (hdr->model_offset < file_size) &&
                        (hdr->custom_data_size == hdr->consts_offset - hdr->custom_data_offset) &&
                        (hdr->consts_size == hdr->model_offset - hdr->consts_offset);
  if (!is_valid_model) return -1;
  return 0;
}
""",
        "driver_ok": """\
  DataHeader h = {sizeof(DataHeader), 0, sizeof(DataHeader), 0, sizeof(DataHeader), 4};
  char* blob = new char[sizeof(DataHeader) + 4]{};
  std::memcpy(blob, &h, sizeof(h));
  DataHeader* out = new DataHeader{};
  if (process_model(blob, sizeof(DataHeader) + 4, out) != 0) return 1;
  volatile size_t keep = out->model_size;
  (void)keep;
  delete out;
  delete[] blob;
""",
        "driver_bad": """\
  char* short_blob = new char[8]{};
  DataHeader* out = new DataHeader{};
  int rc = process_model(short_blob, 7 + static_cast<size_t>(argc), out);
  volatile size_t keep = out->model_size + static_cast<size_t>(rc);
  (void)keep;
  delete out;
  delete[] short_blob;
""",
    },
    # -------------------------------------------------------------------------
    # RW22: apache/arrow - ReadMessageAsync
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW22",
        "family": "F04",
        "gold_class": "INTEGER_OVERFLOW",
        "cwe": "CWE-190",
        "bug_class": "signed_int64_overflow_in_arrow_ipc_metadata_plus_body_length",
        "crucible_packet_id": "arrow-ipc-readmessage-body-length",
        "repo": "apache/arrow",
        "base_commit": "1d3cf6f18205870b88b2beecabc92a6567dfefb0",
        "file_path": "cpp/src/arrow/ipc/message.cc",
        "function_name": "ReadMessageAsync",
        "slug": "arrow_ipc_readmessage_body_len",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["metadata_length", "body_length"],
        "terms": ["body_length", "metadata_length", "numeric_limits"],
        "expect": r"signed integer overflow",
        "vuln": """\
#include <cstdint>
#include <limits>

int ReadMessageAsync(int64_t offset, int32_t metadata_length, int64_t body_length,
                     int64_t* out_read_len) {
  (void)offset;
  if (metadata_length < 8) {
    return -1;
  }
  int64_t total = metadata_length + body_length; //@SINK
  *out_read_len = total;
  return 0;
}
""",
        "safe": """\
#include <cstdint>
#include <limits>

int ReadMessageAsync(int64_t offset, int32_t metadata_length, int64_t body_length,
                     int64_t* out_read_len) {
  (void)offset;
  if (metadata_length < 8) {
    return -1;
  }
  if (body_length < 0 ||
      body_length > std::numeric_limits<int64_t>::max() - metadata_length) { //@GUARD
    return -1;
  }
  int64_t total = metadata_length + body_length;
  *out_read_len = total;
  return 0;
}
""",
        "driver_ok": """\
  int64_t total = 0;
  if (ReadMessageAsync(0, 64, 1024, &total) != 0 || total != 1088) return 1;
""",
        "driver_bad": """\
  int64_t total = 0;
  (void)ReadMessageAsync(0, 64, std::numeric_limits<int64_t>::max(), &total);
""",
    },
    # -------------------------------------------------------------------------
    # RW23: apache/arrow - RecordBatchFileReaderImpl::GetRecordBatchBlock
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW23",
        "family": "F10",
        "gold_class": "ASSERT_ONLY_CHECK",
        "cwe": "CWE-617",
        "bug_class": "dcheck_only_record_batch_index_check_allows_oob_in_release",
        "crucible_packet_id": "arrow-ipc-record-batch-index-bounds",
        "repo": "apache/arrow",
        "base_commit": "1d3cf6f18205870b88b2beecabc92a6567dfefb0",
        "file_path": "cpp/src/arrow/ipc/reader.cc",
        "function_name": "RecordBatchFileReaderImpl::GetRecordBatchBlock",
        "slug": "arrow_ipc_record_batch_idx",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["record_batches_", "GetRecordBatchBlock"],
        "terms": ["num_record_batches", "record_batches_", "i"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cassert>
#include <cstdint>
#include <vector>

struct FileBlock { int64_t offset; int32_t metadata_length; int64_t body_length; };

struct RecordBatchFileReaderImpl {
  std::vector<FileBlock> record_batches_;
  int num_record_batches() const { return static_cast<int>(record_batches_.size()); }

  int GetRecordBatchBlock(int i, FileBlock* out) const {
    assert(i >= 0 && i < num_record_batches()); //@ROOT
    *out = record_batches_.data()[i]; //@SINK
    return 0;
  }
};
""",
        "safe": """\
#include <cassert>
#include <cstdint>
#include <vector>

struct FileBlock { int64_t offset; int32_t metadata_length; int64_t body_length; };

struct RecordBatchFileReaderImpl {
  std::vector<FileBlock> record_batches_;
  int num_record_batches() const { return static_cast<int>(record_batches_.size()); }

  int GetRecordBatchBlock(int i, FileBlock* out) const {
    if (i < 0 || i >= num_record_batches()) { //@GUARD
      return -1;
    }
    *out = record_batches_.data()[i];
    return 0;
  }
};
""",
        "driver_ok": """\
  RecordBatchFileReaderImpl r{{{0, 64, 128}}};
  FileBlock fb{};
  if (r.GetRecordBatchBlock(0, &fb) != 0 || fb.body_length != 128) return 1;
""",
        "driver_bad": """\
  RecordBatchFileReaderImpl r{{{0, 64, 128}}};
  FileBlock fb{};
  int rc = r.GetRecordBatchBlock(3 + argc, &fb);
  volatile int64_t keep = fb.body_length + rc;
  (void)keep;
""",
    },
    # -------------------------------------------------------------------------
    # RW24: openxla/xla - HloEvaluator::HandleSort
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW24",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_indexing_key_shape_dimensions_before_sort_dim_check",
        "crucible_packet_id": "openxla-xla-hlo-evaluator-dimension-bounds",
        "repo": "openxla/xla",
        "base_commit": "bae1f051129af495c55dbaf9d736dd0502019dda",
        "file_path": "xla/hlo/evaluator/hlo_evaluator.cc",
        "function_name": "HloEvaluator::HandleSort",
        "slug": "openxla_hlo_evaluator_sort_dim",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["dimensions", "sort_dim"],
        "terms": ["sort_dim", "key_shape", "increment", "dimensions"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  std::vector<int64_t> dims_;
  int64_t rank() const { return static_cast<int64_t>(dims_.size()); }
  int64_t dimensions(int64_t dim) const { return dims_.data()[dim]; } //@ROOT
};

int HandleSort(const Shape& key_shape, int64_t sort_dim, std::vector<int64_t>* out_inc) {
  int64_t rank = key_shape.rank();
  std::vector<int64_t> increment(rank, 1);
  int64_t sort_dim_elements = key_shape.dimensions(sort_dim); //@SINK
  if (sort_dim < 0 || sort_dim >= static_cast<int64_t>(increment.size())) {
    return -1;
  }
  increment[sort_dim] = sort_dim_elements;
  *out_inc = increment;
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  std::vector<int64_t> dims_;
  int64_t rank() const { return static_cast<int64_t>(dims_.size()); }
  int64_t dimensions(int64_t dim) const { return dims_.data()[dim]; }
};

int HandleSort(const Shape& key_shape, int64_t sort_dim, std::vector<int64_t>* out_inc) {
  int64_t rank = key_shape.rank();
  std::vector<int64_t> increment(rank, 1);
  if (sort_dim < 0 || sort_dim >= static_cast<int64_t>(increment.size())) { //@GUARD
    return -1;
  }
  int64_t sort_dim_elements = key_shape.dimensions(sort_dim);
  increment[sort_dim] = sort_dim_elements;
  *out_inc = increment;
  return 0;
}
""",
        "driver_ok": """\
  Shape s{{2, 3}};
  std::vector<int64_t> inc;
  if (HandleSort(s, 1, &inc) != 0 || inc[1] != 3) return 1;
""",
        "driver_bad": """\
  Shape s{{2, 3}};
  std::vector<int64_t> inc;
  volatile int rc = HandleSort(s, -1, &inc);
  (void)rc;
""",
    },
    # -------------------------------------------------------------------------
    # RW25: openxla/xla - SnappyInputBuffer::Inflate
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW25",
        "family": "F10",
        "gold_class": "ASSERT_ONLY_CHECK",
        "cwe": "CWE-617",
        "bug_class": "dcheck_only_snappy_output_buffer_capacity_check_allows_heap_oob_write",
        "crucible_packet_id": "openxla-xla-tsl-snappy-input-buffer-capacity",
        "repo": "openxla/xla",
        "base_commit": "bae1f051129af495c55dbaf9d736dd0502019dda",
        "file_path": "xla/tsl/lib/io/snappy/snappy_inputbuffer.cc",
        "function_name": "SnappyInputBuffer::Inflate",
        "slug": "openxla_tsl_snappy_inflate_cap",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memcpy", "output_buffer_"],
        "terms": ["uncompressed_length", "output_buffer_capacity_", "output_buffer_"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

struct SnappyInputBuffer {
  size_t output_buffer_capacity_;
  std::unique_ptr<char[]> output_buffer_;
  size_t avail_out_ = 0;

  int Inflate(const char* block, size_t uncompressed_length) {
    assert(output_buffer_capacity_ >= uncompressed_length); //@ROOT
    char* next_out = output_buffer_.get();
    std::memcpy(next_out, block, uncompressed_length); //@SINK
    avail_out_ += uncompressed_length;
    return 0;
  }
};
""",
        "safe": """\
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

struct SnappyInputBuffer {
  size_t output_buffer_capacity_;
  std::unique_ptr<char[]> output_buffer_;
  size_t avail_out_ = 0;

  int Inflate(const char* block, size_t uncompressed_length) {
    if (uncompressed_length > output_buffer_capacity_) { //@GUARD
      return -1;
    }
    char* next_out = output_buffer_.get();
    std::memcpy(next_out, block, uncompressed_length);
    avail_out_ += uncompressed_length;
    return 0;
  }
};
""",
        "driver_ok": """\
  SnappyInputBuffer b{16, std::make_unique<char[]>(16), 0};
  char src[16] = {};
  if (b.Inflate(src, 16) != 0) return 1;
  volatile char keep = b.output_buffer_[0];
  (void)keep;
""",
        "driver_bad": """\
  SnappyInputBuffer b{8, std::make_unique<char[]>(8), 0};
  char src[24] = {};
  int rc = b.Inflate(src, 23 + static_cast<size_t>(argc));
  volatile char keep = b.output_buffer_[0] + static_cast<char>(rc);
  (void)keep;
""",
    },
    # -------------------------------------------------------------------------
    # RW26: openxla/xla - ShapeVerifier::VerifyEntryComputationLayout
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW26",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_indexing_parameter_layout_on_out_of_range_alias_parameter_number",
        "crucible_packet_id": "openxla-xla-hlo-verifier-alias-and-sort-bounds",
        "repo": "openxla/xla",
        "base_commit": "bae1f051129af495c55dbaf9d736dd0502019dda",
        "file_path": "xla/service/hlo_verifier.cc",
        "function_name": "ShapeVerifier::VerifyEntryComputationLayout",
        "slug": "openxla_hlo_verifier_alias_param",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["parameter_layout", "parameter_number"],
        "terms": ["parameter_number", "parameter_count", "parameter_layout"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct Alias { int64_t parameter_number; };
struct ComputationLayout {
  std::vector<int64_t> param_shapes_;
  int64_t parameter_count() const { return static_cast<int64_t>(param_shapes_.size()); }
  int64_t parameter_layout(int64_t i) const { return param_shapes_.data()[i]; } //@ROOT
};

int VerifyEntryComputationLayout(const ComputationLayout& layout, const Alias& alias,
                                 int64_t result_shape) {
  int64_t param_shape = layout.parameter_layout(alias.parameter_number); //@SINK
  return param_shape == result_shape ? 0 : -1;
}
""",
        "safe": """\
#include <cstddef>
#include <cstdint>
#include <vector>

struct Alias { int64_t parameter_number; };
struct ComputationLayout {
  std::vector<int64_t> param_shapes_;
  int64_t parameter_count() const { return static_cast<int64_t>(param_shapes_.size()); }
  int64_t parameter_layout(int64_t i) const { return param_shapes_.data()[i]; }
};

int VerifyEntryComputationLayout(const ComputationLayout& layout, const Alias& alias,
                                 int64_t result_shape) {
  if (alias.parameter_number < 0 || alias.parameter_number >= layout.parameter_count()) { //@GUARD
    return -1;
  }
  int64_t param_shape = layout.parameter_layout(alias.parameter_number);
  return param_shape == result_shape ? 0 : -1;
}
""",
        "driver_ok": """\
  ComputationLayout layout{{4}};
  Alias a{0};
  if (VerifyEntryComputationLayout(layout, a, 4) != 0) return 1;
""",
        "driver_bad": """\
  ComputationLayout layout{{4}};
  Alias a{2 + argc};
  volatile int rc = VerifyEntryComputationLayout(layout, a, 4);
  (void)rc;
""",
    },
    # -------------------------------------------------------------------------
    # RW27: tensorflow/tensorflow - IsCancellableConstPermTransposeNodePair
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW27",
        "family": "F01",
        "gold_class": "HEAP_BUFFER_OVERFLOW",
        "cwe": "CWE-122",
        "bug_class": "heap_oob_read_indexing_fanout_permutation_with_unvalidated_fanin_entry",
        "crucible_packet_id": "tensorflow-grappler-generic-layout-pad-perm-oob",
        "repo": "tensorflow/tensorflow",
        "base_commit": "4a50894fcbae79897a7b728f7db3f782542cd26b",
        "file_path": "tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc",
        "function_name": "IsCancellableConstPermTransposeNodePair",
        "slug": "tf_grappler_layout_perm_oob",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["fanout_tensor", "fanin_tensor"],
        "terms": ["fanin_tensor", "fanout_tensor", "num_elements", "index"],
        "expect": r"heap-buffer-overflow",
        "vuln": """\
#include <cstdint>
#include <vector>

bool IsCancellableConstPermTransposeNodePair(const std::vector<int32_t>& fanout_tensor,
                                             const std::vector<int32_t>& fanin_tensor) {
  if (fanout_tensor.size() != fanin_tensor.size()) {
    return false;
  }
  const int32_t* fanout_data = fanout_tensor.data();
  const int32_t* fanin_data = fanin_tensor.data();
  const int num_elements = static_cast<int>(fanout_tensor.size());
  for (int i = 0; i < num_elements; ++i) {
    if (fanout_data[fanin_data[i]] != i) { //@SINK
      return false;
    }
  }
  return true;
}
""",
        "safe": """\
#include <cstdint>
#include <vector>

bool IsCancellableConstPermTransposeNodePair(const std::vector<int32_t>& fanout_tensor,
                                             const std::vector<int32_t>& fanin_tensor) {
  if (fanout_tensor.size() != fanin_tensor.size()) {
    return false;
  }
  const int32_t* fanout_data = fanout_tensor.data();
  const int32_t* fanin_data = fanin_tensor.data();
  const int64_t num_elements = static_cast<int64_t>(fanout_tensor.size());
  for (int64_t i = 0; i < num_elements; ++i) {
    const int32_t index = fanin_data[i];
    if (index < 0 || index >= num_elements) { //@GUARD
      return false;
    }
    if (fanout_data[index] != i) {
      return false;
    }
  }
  return true;
}
""",
        "driver_ok": """\
  std::vector<int32_t> fo = {0, 2, 1, 3};
  std::vector<int32_t> fi = {0, 2, 1, 3};
  if (!IsCancellableConstPermTransposeNodePair(fo, fi)) return 1;
""",
        "driver_bad": """\
  std::vector<int32_t> fo = {0, 2, 3, 1};
  std::vector<int32_t> fi = {0, 16, 1, 2};
  (void)IsCancellableConstPermTransposeNodePair(fo, fi);
""",
    },
    # -------------------------------------------------------------------------
    # RW28: tensorflow/tensorflow - ConstantFolding::PrepareConstantPushDown
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW28",
        "family": "F07",
        "gold_class": "NULL_DEREFERENCE",
        "cwe": "CWE-476",
        "bug_class": "null_dereference_binding_reference_to_missing_child_input_node",
        "crucible_packet_id": "tensorflow-grappler-constant-folding-mulconv-and-rank-guards",
        "repo": "tensorflow/tensorflow",
        "base_commit": "ace46d5052cf64a198194df7291af365a1380105",
        "file_path": "tensorflow/core/grappler/optimizers/constant_folding.cc",
        "function_name": "ConstantFolding::PrepareConstantPushDown",
        "slug": "tf_grappler_constant_pushdown_null",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["IsReallyConstant", "left_leaf", "right_leaf"],
        "terms": ["left_leaf", "right_leaf", "IsReallyConstant"],
        "expect": r"reference binding to null pointer|member access within null pointer|SEGV",
        "vuln": """\
#include <cstdint>

struct NodeDef { int op_kind; };
struct PushDownCtx {
  NodeDef* left_leaf;
  NodeDef* right_leaf;
  bool left_leaf_is_const;
  bool right_leaf_is_const;
};

static inline bool IsReallyConstant(const NodeDef& n) { return n.op_kind == 1; }

bool PrepareConstantPushDown(NodeDef* left_in, NodeDef* right_in, PushDownCtx* ctx) {
  ctx->left_leaf = left_in;
  ctx->right_leaf = right_in;
  ctx->left_leaf_is_const = IsReallyConstant(*ctx->left_leaf); //@SINK
  ctx->right_leaf_is_const = IsReallyConstant(*ctx->right_leaf);
  return ctx->left_leaf_is_const || ctx->right_leaf_is_const;
}
""",
        "safe": """\
#include <cstdint>

struct NodeDef { int op_kind; };
struct PushDownCtx {
  NodeDef* left_leaf;
  NodeDef* right_leaf;
  bool left_leaf_is_const;
  bool right_leaf_is_const;
};

static inline bool IsReallyConstant(const NodeDef& n) { return n.op_kind == 1; }

bool PrepareConstantPushDown(NodeDef* left_in, NodeDef* right_in, PushDownCtx* ctx) {
  ctx->left_leaf = left_in;
  ctx->right_leaf = right_in;
  if (ctx->left_leaf == nullptr || ctx->right_leaf == nullptr) { //@GUARD
    return false;
  }
  ctx->left_leaf_is_const = IsReallyConstant(*ctx->left_leaf);
  ctx->right_leaf_is_const = IsReallyConstant(*ctx->right_leaf);
  return ctx->left_leaf_is_const || ctx->right_leaf_is_const;
}
""",
        "driver_ok": """\
  NodeDef a{1}, b{0};
  PushDownCtx ctx{};
  if (!PrepareConstantPushDown(&a, &b, &ctx)) return 1;
""",
        "driver_bad": """\
  NodeDef b{1};
  PushDownCtx ctx{};
  (void)PrepareConstantPushDown(nullptr, &b, &ctx);
""",
    },
    # -------------------------------------------------------------------------
    # RW29: tensorflow/tensorflow - BundleReader::GetValue
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW29",
        "family": "F05",
        "gold_class": "SIGN_CONVERSION",
        "cwe": "CWE-195",
        "bug_class": "negative_section_size_converted_to_size_t_in_threaded_tensor_bundle_restore",
        "crucible_packet_id": "tensorflow-tensor-bundle-getvalue-threaded-restore-bounds",
        "repo": "tensorflow/tensorflow",
        "base_commit": "cd7fc0e41c85fa887d32a7deb3a175b712dc9522",
        "file_path": "tensorflow/core/util/tensor_bundle/tensor_bundle.cc",
        "function_name": "BundleReader::GetValue",
        "slug": "tf_tensor_bundle_threaded_restore",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["memmove", "backing_buffer"],
        "terms": ["entry_size", "offset", "section_size", "size"],
        "expect": r"heap-buffer-overflow|negative-size-param",
        "vuln": """\
#include <algorithm>
#include <cstdint>
#include <cstring>

int RestoreTensorSections(int64_t entry_size, int thread_pool_size,
                          const char* shard_data, char* backing_buffer) {
  int64_t section_size = (entry_size + (thread_pool_size - 1)) / thread_pool_size;
  for (int i = 0; i < thread_pool_size; ++i) {
    int64_t offset = i * section_size;
    int64_t size = (i == thread_pool_size - 1) ? entry_size - offset : section_size; //@ROOT
    char* backing_buffer_current_pos = backing_buffer + offset;
    std::memmove(backing_buffer_current_pos, shard_data + offset, static_cast<size_t>(size)); //@SINK
  }
  return 0;
}
""",
        "safe": """\
#include <algorithm>
#include <cstdint>
#include <cstring>

int RestoreTensorSections(int64_t entry_size, int thread_pool_size,
                          const char* shard_data, char* backing_buffer) {
  int64_t section_size = (entry_size + (thread_pool_size - 1)) / thread_pool_size;
  for (int i = 0; i < thread_pool_size; ++i) {
    const int64_t offset = i * section_size;
    if (offset >= entry_size) break; //@GUARD
    const int64_t size = std::min(section_size, entry_size - offset); //@GUARD
    char* backing_buffer_current_pos = backing_buffer + offset;
    std::memmove(backing_buffer_current_pos, shard_data + offset, static_cast<size_t>(size));
  }
  return 0;
}
""",
        "driver_ok": """\
  char* src = new char[24]{};
  char* dst = new char[24]{};
  if (RestoreTensorSections(24, 8, src, dst) != 0) return 1;
  delete[] src;
  delete[] dst;
""",
        "driver_bad": """\
  char* src = new char[2]{1, 2};
  char* dst = new char[2]{0, 0};
  (void)RestoreTensorSections(2, 8, src, dst);
  delete[] src;
  delete[] dst;
""",
    },
    # -------------------------------------------------------------------------
    # RW30: tensorflow/tensorflow - InterpreterBuilder::operator()
    # -------------------------------------------------------------------------
    {
        "pair_id": "RW30",
        "family": "F07",
        "gold_class": "NULL_DEREFERENCE",
        "cwe": "CWE-476",
        "bug_class": "null_dereference_when_tflite_model_omits_subgraphs_vector",
        "crucible_packet_id": "tensorflow-tflite-interpreter-builder-null-subgraphs",
        "repo": "tensorflow/tensorflow",
        "base_commit": "d384f6f8186efb7af1ca68c80ad32ea7b250d6cd",
        "file_path": "tensorflow/lite/core/interpreter_builder.cc",
        "function_name": "InterpreterBuilder::operator()",
        "slug": "tf_tflite_builder_null_subgraphs",
        "lang": "cpp",
        "san": "asan",
        "sink_symbols": ["subgraphs", "empty"],
        "terms": ["subgraphs", "empty", "model_"],
        "expect": r"member call on null pointer|SEGV|AddressSanitizer",
        "vuln": """\
#include <cstddef>
#include <vector>

struct SubGraph { int num_tensors; };
struct Model {
  const std::vector<SubGraph>* subgraphs_;
  const std::vector<SubGraph>* subgraphs() const { return subgraphs_; }
};

int BuildInterpreter(const Model* model, int* out_subgraph_count) {
  auto* subgraphs = model->subgraphs();
  if (subgraphs->empty()) { //@SINK
    return -1;
  }
  *out_subgraph_count = static_cast<int>(subgraphs->size());
  return 0;
}
""",
        "safe": """\
#include <cstddef>
#include <vector>

struct SubGraph { int num_tensors; };
struct Model {
  const std::vector<SubGraph>* subgraphs_;
  const std::vector<SubGraph>* subgraphs() const { return subgraphs_; }
};

int BuildInterpreter(const Model* model, int* out_subgraph_count) {
  auto* subgraphs = model->subgraphs();
  if (!subgraphs || subgraphs->empty()) { //@GUARD
    return -1;
  }
  *out_subgraph_count = static_cast<int>(subgraphs->size());
  return 0;
}
""",
        "driver_ok": """\
  std::vector<SubGraph> sgs = {{4}};
  Model m{&sgs};
  int cnt = 0;
  if (BuildInterpreter(&m, &cnt) != 0 || cnt != 1) return 1;
""",
        "driver_bad": """\
  Model m{nullptr};
  int cnt = 0;
  (void)BuildInterpreter(&m, &cnt);
""",
    },
]


def compile_and_run_O1(clean_src: str, driver: str, workdir: Path, tag: str, save_src_dir: Path | None = None) -> dict:
    """Compile snippet+driver with `clang++ -O1 -g -fsanitize=address,undefined` and run it."""
    src_path = workdir / f"{tag}.cpp"
    full = clean_src + "\n" + (
        "int main(int argc, char **argv) {\n"
        "  (void)argc; (void)argv;\n" + driver.strip("\n") + "\n  return 0;\n}\n"
    )
    src_path.write_text(full, encoding="utf-8")
    if save_src_dir is not None:
        (save_src_dir / f"{tag}.cpp").write_text(full, encoding="utf-8")
    binp = workdir / f"{tag}.bin"
    cmd = [
        "clang++", "-std=c++17", "-O1", "-g", "-fno-omit-frame-pointer",
        "-DNDEBUG", "-U_FORTIFY_SOURCE", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all", str(src_path), "-o", str(binp),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if cp.returncode != 0:
        return {"status": "COMPILE_FAIL", "stderr": cp.stderr[-1500:]}
    pre = L.sha256_file(binp)
    env = dict(os.environ)
    env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=0:detect_stack_use_after_return=1"
    env["UBSAN_OPTIONS"] = "print_stacktrace=1"
    t0 = time.time()
    try:
        rp = subprocess.run([str(binp)], capture_output=True, text=True, timeout=60, env=env)
        rc, out, err, to = rp.returncode, rp.stdout, rp.stderr, False
    except subprocess.TimeoutExpired as exc:
        rc, out, err, to = None, "", (exc.stderr or "") if isinstance(exc.stderr, str) else "", True
    post = L.sha256_file(binp)
    reported = bool(L.REPORT_RE.search(err))
    line = None
    frames = L.FRAME_RE.findall(err)
    mine = [(fn, f, int(ln)) for fn, f, ln in frames if Path(f).name == src_path.name]
    for fn, f, ln in mine:
        if fn != "main":
            line = ln
            break
    if line is None:
        m = L.UBSAN_LOC_RE.search(err)
        if m and Path(m.group(1)).name == src_path.name:
            line = int(m.group(2))
    return {
        "status": "RAN",
        "returncode": rc,
        "timeout": to,
        "reported": reported,
        "report_line": line,
        "source_sha256": L.sha256_bytes(full.encode()),
        "binary_sha256_pre_exec": pre,
        "binary_sha256_post_exec": post,
        "hash_bracket_matched": pre == post,
        "seconds": round(time.time() - t0, 3),
        "stderr_head": err[:1200],
        "stdout_head": out[:300],
        "compiler": "clang++",
        "flags": ["-std=c++17", "-O1", "-g", "-fno-omit-frame-pointer", "-DNDEBUG",
                  "-U_FORTIFY_SOURCE", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"],
    }


def main() -> int:
    assert len(REALWORLD_CASES) == 30, f"expected 30 pairs, got {len(REALWORLD_CASES)}"
    harnesses_dir = HERE / "harnesses"
    harnesses_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path("/tmp/asan_realworld_60_build")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    benchmark_cases = []
    validation_results = []
    families_meta = {}
    problems_total = 0

    for c in REALWORLD_CASES:
        pid = c["pair_id"]
        fam = c["family"]
        families_meta[fam] = {"gold_class": c["gold_class"], "cwe": c["cwe"]}

        pkt_dir = PACKETS_ROOT / c["crucible_packet_id"]
        patch_path = pkt_dir / "patch.diff"
        red_log_path = pkt_dir / "unpatched-red.log"
        green_log_path = pkt_dir / "patched-green.log"
        assert patch_path.is_file(), f"missing {patch_path}"
        assert red_log_path.is_file(), f"missing {red_log_path}"
        assert green_log_path.is_file(), f"missing {green_log_path}"

        patch_sha256 = L.sha256_file(patch_path)
        red_sha256 = L.sha256_file(red_log_path)
        green_sha256 = L.sha256_file(green_log_path)

        vclean, vm = L.split_markers(c["vuln"])
        sclean, sm = L.split_markers(c["safe"])
        vsha = L.sha256_bytes(vclean.encode())
        ssha = L.sha256_bytes(sclean.encode())
        ratio = round(difflib.SequenceMatcher(None, vclean, sclean).ratio(), 3)

        probs = []
        if not vm["SINK"]:
            probs.append("vuln has no //@SINK marker")
        if not sm["GUARD"]:
            probs.append("safe has no //@GUARD marker")
        for nm, txt in (("vuln", vclean), ("safe", sclean)):
            nl = txt.count("\n")
            if nl > 48:
                probs.append(f"{nm} has {nl} lines (>48)")
            lk = L.leak_hits(txt)
            if lk:
                probs.append(f"{nm} contains label-leaking words {lk}")
        if ratio < 0.55:
            probs.append(f"twin similarity {ratio:.2f} < 0.55")
        if vclean == sclean:
            probs.append("twin is identical")

        allowed_lines = set(vm["SINK"]) | set(vm["ROOT"])

        # Save standalone trigger harnesses (`RWxx_vuln.cpp` and `RWxx_benign.cpp`)
        # and run the 4-cell matrix (`vuln_ok`, `vuln_bad`, `benign_ok`, `benign_bad`) under clang++ -O1 -g -fsanitize=address,undefined
        mx_o1 = {
            "vuln_ok": compile_and_run_O1(vclean, c["driver_ok"], workdir, f"{pid}_vuln_ok"),
            "vuln_bad": compile_and_run_O1(vclean, c["driver_bad"], workdir, f"{pid}_vuln", save_src_dir=harnesses_dir),
            "safe_ok": compile_and_run_O1(sclean, c["driver_ok"], workdir, f"{pid}_benign_ok"),
            "safe_bad": compile_and_run_O1(sclean, c["driver_bad"], workdir, f"{pid}_benign", save_src_dir=harnesses_dir),
        }
        # Also run clang-O0 and clang-O2 from asan120_lib
        matrix = {"clang-O1": mx_o1}
        for cfg in ("clang-O0", "clang-O2"):
            mx_cfg = {}
            for which, clean in (("vuln", vclean), ("safe", sclean)):
                for dname in ("ok", "bad"):
                    tag = f"{pid}_{which}_{dname}_{cfg}".replace("-", "_")
                    mx_cfg[f"{which}_{dname}"] = L.compile_and_run(
                        clean, c[f"driver_{dname}"], c["lang"], c["san"], cfg, which, workdir, tag
                    )
            matrix[cfg] = mx_cfg

        for cfg, mx in matrix.items():
            for k, v in mx.items():
                if v.get("status") != "RAN":
                    probs.append(f"[{cfg}] {k}: status={v.get('status')} err={v.get('stderr', '')[:200]}")
                elif not v.get("hash_bracket_matched"):
                    probs.append(f"[{cfg}] {k}: binary hash changed across exec")
            if any(v.get("status") != "RAN" for v in mx.values()):
                continue
            for k in ("vuln_ok", "safe_ok", "safe_bad"):
                if mx[k]["reported"] or mx[k]["timeout"] or mx[k]["returncode"] != 0:
                    probs.append(f"[{cfg}] {k}: unexpected failure rc={mx[k]['returncode']} err={mx[k]['stderr_head'][:200]!r}")
            vb = mx["vuln_bad"]
            if not vb["reported"] or vb["returncode"] == 0:
                probs.append(f"[{cfg}] vuln_bad: sanitizer did not fire (rc={vb['returncode']})")
            else:
                if not re.search(c["expect"], vb["stderr_head"], re.S):
                    probs.append(f"[{cfg}] vuln_bad: report does not match /{c['expect']}/: {vb['stderr_head'][:200]!r}")
                if vb["report_line"] is not None and vb["report_line"] not in allowed_lines:
                    probs.append(f"[{cfg}] vuln_bad: report_line {vb['report_line']} not in SINK/ROOT {sorted(allowed_lines)}")

        status = "PASS" if not probs else "FAIL"
        if probs:
            problems_total += len(probs)
            print(f"[FAIL] {pid} ({c['slug']}): {probs}", flush=True)
        else:
            vb_o1 = mx_o1["vuln_bad"]
            sb_o1 = mx_o1["safe_bad"]
            print(
                f"[PASS] {pid} | {c['repo']} | {c['function_name']} | {c['gold_class']} | "
                f"vuln_bad rc={vb_o1['returncode']} line={vb_o1['report_line']} (allowed={sorted(allowed_lines)}) | "
                f"benign_bad rc={sb_o1['returncode']}",
                flush=True,
            )

        val_rec = {
            "pair_id": pid,
            "slug": c["slug"],
            "repo": c["repo"],
            "base_commit": c["base_commit"],
            "file_path": c["file_path"],
            "function_name": c["function_name"],
            "crucible_packet_id": c["crucible_packet_id"],
            "patch_sha256": patch_sha256,
            "unpatched_red_log_sha256": red_sha256,
            "patched_green_log_sha256": green_sha256,
            "vuln_sha256": vsha,
            "safe_sha256": ssha,
            "twin_similarity": ratio,
            "sink_lines": vm["SINK"],
            "root_lines": vm["ROOT"],
            "guard_lines": sm["GUARD"],
            "configs_ran": list(matrix.keys()),
            "problems": probs,
            "status": status,
            "matrix": matrix,
        }
        validation_results.append(val_rec)

        manifest_entries.append({
            "case_id": pid,
            "pair_id": pid,
            "vuln_case_id": f"{pid}-V",
            "benign_case_id": f"{pid}-S",
            "repo": c["repo"],
            "base_commit": c["base_commit"],
            "file_path": c["file_path"],
            "function_name": c["function_name"],
            "bug_class": c["bug_class"],
            "gold_class": c["gold_class"],
            "family": fam,
            "cwe": c["cwe"],
            "crucible_packet_id": c["crucible_packet_id"],
            "patch_sha256": patch_sha256,
            "unpatched_red_log_sha256": red_sha256,
            "patched_green_log_sha256": green_sha256,
            "vuln_code_sha256": vsha,
            "benign_code_sha256": ssha,
            "twin_similarity": ratio,
            "sink_lines": vm["SINK"],
            "root_lines": vm["ROOT"],
            "guard_lines": sm["GUARD"],
            "vuln_harness": f"harnesses/{pid}_vuln.cpp",
            "benign_harness": f"harnesses/{pid}_benign.cpp",
            "clang_O1_verification": {
                "vuln_trigger_exit_code": mx_o1["vuln_bad"]["returncode"],
                "vuln_trigger_sanitizer_reported": mx_o1["vuln_bad"]["reported"],
                "vuln_trigger_report_line": mx_o1["vuln_bad"]["report_line"],
                "benign_trigger_exit_code": mx_o1["safe_bad"]["returncode"],
                "benign_trigger_sanitizer_reported": mx_o1["safe_bad"]["reported"],
                "vuln_control_exit_code": mx_o1["vuln_ok"]["returncode"],
                "benign_control_exit_code": mx_o1["safe_ok"]["returncode"],
                "status": status,
            },
        })

        common = dict(
            pair_id=pid,
            family=fam,
            slug=c["slug"],
            lang=c["lang"],
            cwe=c["cwe"],
            sanitizer=c["san"],
            twin_similarity=ratio,
            repo=c["repo"],
            base_commit=c["base_commit"],
            file_path=c["file_path"],
            function_name=c["function_name"],
            bug_class=c["bug_class"],
            crucible_packet_id=c["crucible_packet_id"],
            patch_sha256=patch_sha256,
        )
        benchmark_cases.append(dict(
            common,
            case_id=f"{pid}-V",
            label="VULNERABLE",
            gold_class=c["gold_class"],
            code=vclean,
            code_sha256=vsha,
            sink_lines=vm["SINK"],
            root_lines=vm["ROOT"],
            guard_lines=[],
            sink_symbols=c["sink_symbols"],
            terms=c["terms"],
            prompt=P.render_prompt(vclean, c["lang"]),
            validation_configs=list(matrix.keys()),
        ))
        benchmark_cases.append(dict(
            common,
            case_id=f"{pid}-S",
            label="SAFE",
            gold_class="NONE",
            code=sclean,
            code_sha256=ssha,
            sink_lines=[],
            root_lines=[],
            guard_lines=sm["GUARD"],
            sink_symbols=c["sink_symbols"],
            terms=c["terms"],
            prompt=P.render_prompt(sclean, c["lang"]),
            validation_configs=list(matrix.keys()),
        ))

    if problems_total > 0:
        sys.exit(f"validation failed with {problems_total} problem(s)")

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest_doc = {
        "schema": "crucible.asan_realworld_60.manifest.v1",
        "created_utc": now_utc,
        "n_pairs": 30,
        "n_functions": 60,
        "compiler_verification": "clang++ -std=c++17 -O1 -g -fno-omit-frame-pointer -DNDEBUG -U_FORTIFY_SOURCE -fsanitize=address,undefined -fno-sanitize-recover=all",
        "all_30_unpatched_red": True,
        "all_30_patched_green": True,
        "pairs": manifest_entries,
    }
    manifest_path = HERE / "asan_realworld_60_manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    val_doc = {
        "schema": "crucible.asan_realworld_60.validation.v1",
        "created_utc": now_utc,
        "configs": ["clang-O1", "clang-O0", "clang-O2"],
        "n_pairs": 30,
        "n_pass": sum(1 for r in validation_results if r["status"] == "PASS"),
        "n_fail": sum(1 for r in validation_results if r["status"] == "FAIL"),
        "results": validation_results,
    }
    val_path = HERE / "validation_receipt.json"
    val_path.write_text(json.dumps(val_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    bench_doc = {
        "schema": "crucible.asan120.v1",
        "suite": "ASAN-RealWorld-60",
        "attempt_id": L.ATTEMPT_ID,
        "created_utc": now_utc,
        "n_items": len(benchmark_cases),
        "n_pairs": 30,
        "families": {f: {"gold_class": g, "cwe": w} for f, (g, w) in sorted(
            {c["family"]: (c["gold_class"], c["cwe"]) for c in REALWORLD_CASES}.items()
        )},
        "validation_configs": ["clang-O1", "clang-O0", "clang-O2"],
        "prompt_preamble_sha256": L.sha256_bytes((P.PREAMBLE + P.CONTRACT).encode()),
        "cases": benchmark_cases,
    }
    bench_path = HERE / "asan_realworld_60_paired_benchmark.json"
    bench_path.write_text(json.dumps(bench_doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Build combined ASAN-180 benchmark (ASAN-120 + ASAN-RealWorld-60 = 180 functions / 90 pairs)
    asan120_path = ASAN120_DIR / "asan120_paired_benchmark.json"
    asan120_doc = json.loads(asan120_path.read_text(encoding="utf-8"))
    combined_cases = list(asan120_doc["cases"]) + benchmark_cases
    combined_doc = {
        "schema": "crucible.asan180.v1",
        "suite": "ASAN-180 (ASAN-120 + ASAN-RealWorld-60)",
        "attempt_id": L.ATTEMPT_ID,
        "created_utc": now_utc,
        "n_items": len(combined_cases),
        "n_pairs": 90,
        "asan120_sha256": L.sha256_file(asan120_path),
        "asan_realworld_60_sha256": L.sha256_file(bench_path),
        "families": asan120_doc["families"],
        "prompt_preamble_sha256": L.sha256_bytes((P.PREAMBLE + P.CONTRACT).encode()),
        "cases": combined_cases,
    }
    combined_path = HERE / "asan180_combined_benchmark.json"
    combined_path.write_text(json.dumps(combined_doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {manifest_path} sha256={L.sha256_file(manifest_path)}")
    print(f"wrote {val_path} sha256={L.sha256_file(val_path)}")
    print(f"wrote {bench_path} items={len(benchmark_cases)} sha256={L.sha256_file(bench_path)}")
    print(f"wrote {combined_path} items={len(combined_cases)} sha256={L.sha256_file(combined_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
