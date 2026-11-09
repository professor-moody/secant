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
    upb_memset_nonnull((void*)file->ext_layouts, 0,
                       sizeof(*file->ext_layouts) * file->ext_count);
  }
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  upb_FileDef f = {argc - 1, nullptr};
  _upb_FileDef_Create(&f, true);
  return 0;
}
