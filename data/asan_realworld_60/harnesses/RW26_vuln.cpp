#include <cstddef>
#include <cstdint>
#include <vector>

struct Alias { int64_t parameter_number; };
struct ComputationLayout {
  std::vector<int64_t> param_shapes_;
  int64_t parameter_count() const { return static_cast<int64_t>(param_shapes_.size()); }
  int64_t parameter_layout(int64_t i) const { return param_shapes_.data()[i]; }
};

int VerifyEntryComputationLayout(const ComputationLayout& layout, const Alias& alias,
                                 int64_t result_shape) {
  int64_t param_shape = layout.parameter_layout(alias.parameter_number);
  return param_shape == result_shape ? 0 : -1;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  ComputationLayout layout{{4}};
  Alias a{2 + argc};
  volatile int rc = VerifyEntryComputationLayout(layout, a, 4);
  (void)rc;
  return 0;
}
