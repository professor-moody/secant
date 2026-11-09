#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

struct StorageMat {
  int elem_type;
  size_t elem_count;
  std::vector<uint8_t> raw;
};

int readTensor(const std::map<int, StorageMat>& storages, int indexStorage,
               int typeTensor, size_t count, float* dst) {
  auto it = storages.find(indexStorage);
  if (it == storages.end()) return -1;
  const StorageMat& st = it->second;
  if (st.elem_type != typeTensor) {
    return -1;
  }
  if (count > st.elem_count) return -1;
  size_t elem_size = (typeTensor == 5) ? sizeof(float) : sizeof(uint8_t);
  std::memcpy(dst, st.raw.data(), count * elem_size);
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  std::map<int, StorageMat> st;
  st[4] = StorageMat{0, 4, std::vector<uint8_t>(4, 1)};
  float* dst = new float[4]{};
  int rc = readTensor(st, 4, 5, 3 + static_cast<size_t>(argc), dst);
  volatile float keep = dst[3] + static_cast<float>(rc);
  (void)keep;
  delete[] dst;
  return 0;
}
