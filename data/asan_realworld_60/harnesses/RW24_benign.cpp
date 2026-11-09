#include <cstddef>
#include <cstdint>
#include <vector>

struct Shape {
  std::vector<int64_t> dims_;
  int64_t rank() const { return static_cast<int64_t>(dims_.size()); }
  int64_t dimensions(int64_t dim) const { return dims_.data()[dim]; }
};

int HandleSort(const Shape& key_shape, int64_t sort_dim, std::vector<int64_t>* out_inc) {
  int64_t rank = key_shape.rank();
  std::vector<int64_t> increment(rank, 1);
  if (sort_dim < 0 || sort_dim >= static_cast<int64_t>(increment.size())) {
    return -1;
  }
  int64_t sort_dim_elements = key_shape.dimensions(sort_dim);
  increment[sort_dim] = sort_dim_elements;
  *out_inc = increment;
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  Shape s{{2, 3}};
  std::vector<int64_t> inc;
  volatile int rc = HandleSort(s, -1, &inc);
  (void)rc;
  return 0;
}
