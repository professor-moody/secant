#include <cstdint>
#include <limits>

int ReadMessageAsync(int64_t offset, int32_t metadata_length, int64_t body_length,
                     int64_t* out_read_len) {
  (void)offset;
  if (metadata_length < 8) {
    return -1;
  }
  if (body_length < 0 ||
      body_length > std::numeric_limits<int64_t>::max() - metadata_length) {
    return -1;
  }
  int64_t total = metadata_length + body_length;
  *out_read_len = total;
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  int64_t total = 0;
  (void)ReadMessageAsync(0, 64, std::numeric_limits<int64_t>::max(), &total);
  return 0;
}
