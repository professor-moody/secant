#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  std::vector<int64_t> dims_;
  const std::vector<int64_t>& dimensions() const { return dims_; }
  int64_t dimensions(int64_t index) const { return dims_.data()[index]; }
};

int InferTopKShape(const Shape& operand_shape, int64_t k, int64_t* out_dim) {
  if (operand_shape.dimensions().empty()) {
    return -1;
  }
  int64_t last_dim = static_cast<int64_t>(operand_shape.dimensions().size()) - 1;
  if (k < 0 || operand_shape.dimensions(last_dim) < k) {
    return -1;
  }
  *out_dim = k;
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  Shape s;
  s.dims_.reserve(4);
  int64_t out = 0;
  volatile int rc = InferTopKShape(s, argc, &out);
  (void)rc;
  return 0;
}
