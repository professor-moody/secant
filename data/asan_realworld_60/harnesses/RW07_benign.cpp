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
  if (hb_unsigned_mul_overflows(axis_count, region_count, &total_axes) ||
      hb_unsigned_mul_overflows(total_axes, VarRegionAxis_static_size)) {
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

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  uint8_t* p = VarRegionList_serialize(4u, 0x40000000u + static_cast<unsigned>(argc));
  volatile uint8_t keep = p ? p[0] : 0;
  (void)keep;
  std::free(p);
  return 0;
}
