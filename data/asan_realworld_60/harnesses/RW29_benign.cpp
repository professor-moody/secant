#include <algorithm>
#include <cstdint>
#include <cstring>

int RestoreTensorSections(int64_t entry_size, int thread_pool_size,
                          const char* shard_data, char* backing_buffer) {
  int64_t section_size = (entry_size + (thread_pool_size - 1)) / thread_pool_size;
  for (int i = 0; i < thread_pool_size; ++i) {
    const int64_t offset = i * section_size;
    if (offset >= entry_size) break;
    const int64_t size = std::min(section_size, entry_size - offset);
    char* backing_buffer_current_pos = backing_buffer + offset;
    std::memmove(backing_buffer_current_pos, shard_data + offset, static_cast<size_t>(size));
  }
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  char* src = new char[2]{1, 2};
  char* dst = new char[2]{0, 0};
  (void)RestoreTensorSections(2, 8, src, dst);
  delete[] src;
  delete[] dst;
  return 0;
}
