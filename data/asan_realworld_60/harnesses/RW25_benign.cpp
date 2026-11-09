#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

struct SnappyInputBuffer {
  size_t output_buffer_capacity_;
  std::unique_ptr<char[]> output_buffer_;
  size_t avail_out_ = 0;

  int Inflate(const char* block, size_t uncompressed_length) {
    if (uncompressed_length > output_buffer_capacity_) {
      return -1;
    }
    char* next_out = output_buffer_.get();
    std::memcpy(next_out, block, uncompressed_length);
    avail_out_ += uncompressed_length;
    return 0;
  }
};

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  SnappyInputBuffer b{8, std::make_unique<char[]>(8), 0};
  char src[24] = {};
  int rc = b.Inflate(src, 23 + static_cast<size_t>(argc));
  volatile char keep = b.output_buffer_[0] + static_cast<char>(rc);
  (void)keep;
  return 0;
}
