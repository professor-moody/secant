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
  if (!_upb_Arena_CheckedAdd(*size, kUpb_MemblockReserve, &sz)) {
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

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  size_t req = std::numeric_limits<size_t>::max() - 15;
  upb_MemBlock* b = _upb_Arena_AllocBlockInternal(&req);
  std::free(b);
  return 0;
}
