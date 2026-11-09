#include <cstddef>
#include <cstdint>

int moe_align_block_size(const int32_t* topk_ids, int numel, int num_experts,
                         int32_t* total_cnts) {
  for (int i = 0; i < numel; ++i) {
    if (topk_ids[i] < 0 || topk_ids[i] >= num_experts) {
      return -1;
    }
  }
  for (int i = 0; i < numel; ++i) {
    total_cnts[topk_ids[i]] += 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  int32_t* cnts = new int32_t[8]{};
  int32_t ids[4] = {0, 64, 1, 2};
  (void)moe_align_block_size(ids, 4, 8, cnts);
  delete[] cnts;
  return 0;
}
