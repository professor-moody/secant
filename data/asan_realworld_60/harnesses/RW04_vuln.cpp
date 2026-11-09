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
  ptr = upb_WireReader_ReadVarint(ptr, &message_len);
  size_t size = static_cast<size_t>(message_len);
  if (size > cap) return -1;
  std::memcpy(dst, ptr, size);
  return static_cast<int>(size);
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  char* raw = new char[4]{0x2a, 0x08, 0x08, 0x01};
  upb_UnknownField u = {raw, 4};
  uint8_t dst[16] = {};
  int rc = upb_MiniTable_ParseUnknownMessage(&u, dst, sizeof(dst));
  volatile uint8_t keep = dst[0] + static_cast<uint8_t>(rc);
  (void)keep;
  delete[] raw;
  return 0;
}
