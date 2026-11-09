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
  if (file_size < sizeof(*hdr)) return -1;
  std::memcpy(reinterpret_cast<char*>(hdr), buffer_base, sizeof(*hdr));
  bool is_valid_model = (hdr->custom_data_offset == sizeof(*hdr)) &&
                        (hdr->custom_data_offset <= hdr->consts_offset) &&
                        (hdr->consts_offset <= hdr->model_offset) &&
                        (hdr->model_offset < file_size) &&
                        (hdr->custom_data_size == hdr->consts_offset - hdr->custom_data_offset) &&
                        (hdr->consts_size == hdr->model_offset - hdr->consts_offset);
  if (!is_valid_model) return -1;
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  char* short_blob = new char[8]{};
  DataHeader* out = new DataHeader{};
  int rc = process_model(short_blob, 7 + static_cast<size_t>(argc), out);
  volatile size_t keep = out->model_size + static_cast<size_t>(rc);
  (void)keep;
  delete out;
  delete[] short_blob;
  return 0;
}
