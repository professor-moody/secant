#include <cstddef>
#include <cstdint>

struct Split { int varIdx; float c; };

int predictTrees(const Split& split, const int* cvidx, const float* psample,
                 size_t sample_total, size_t sstep, float* out_val) {
  int vi = split.varIdx;
  int ci = cvidx ? cvidx[vi] : vi;
  if (ci < 0 || static_cast<size_t>(ci) >= sample_total) {
    return -1;
  }
  float val = psample[ci * sstep];
  *out_val = val - split.c;
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  float* sample = new float[3]{1.5f, 2.5f, 3.5f};
  int map[2] = {0, 16};
  Split sp{1, 0.5f};
  float out = 0.0f;
  int rc = predictTrees(sp, map, sample, 3, 1, &out);
  volatile float keep = out + static_cast<float>(rc);
  (void)keep;
  delete[] sample;
  return 0;
}
