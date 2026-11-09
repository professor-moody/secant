#include <cassert>
#include <cstdint>
#include <vector>

struct FileBlock { int64_t offset; int32_t metadata_length; int64_t body_length; };

struct RecordBatchFileReaderImpl {
  std::vector<FileBlock> record_batches_;
  int num_record_batches() const { return static_cast<int>(record_batches_.size()); }

  int GetRecordBatchBlock(int i, FileBlock* out) const {
    if (i < 0 || i >= num_record_batches()) {
      return -1;
    }
    *out = record_batches_.data()[i];
    return 0;
  }
};

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  RecordBatchFileReaderImpl r{{{0, 64, 128}}};
  FileBlock fb{};
  int rc = r.GetRecordBatchBlock(3 + argc, &fb);
  volatile int64_t keep = fb.body_length + rc;
  (void)keep;
  return 0;
}
