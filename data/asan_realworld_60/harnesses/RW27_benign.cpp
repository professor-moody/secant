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
    if (index < 0 || index >= num_elements) {
      return false;
    }
    if (fanout_data[index] != i) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  std::vector<int32_t> fo = {0, 2, 3, 1};
  std::vector<int32_t> fi = {0, 16, 1, 2};
  (void)IsCancellableConstPermTransposeNodePair(fo, fi);
  return 0;
}
