#include <cstddef>
#include <cstdint>
#include <cstring>

static constexpr size_t XNN_MAX_TENSOR_DIMS = 6;
struct xnn_shape { size_t num_dims; size_t dim[XNN_MAX_TENSOR_DIMS]; };
struct xnn_runtime_value { xnn_shape shape; };

int resize_unary_elementwise_output_tensor(const size_t* input_dims, size_t input_num_dims,
                                           xnn_runtime_value* output) {
  output->shape.num_dims = input_num_dims;
  std::memcpy(output->shape.dim, input_dims, input_num_dims * sizeof(size_t));
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  size_t in_dims[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  xnn_runtime_value* out = new xnn_runtime_value{};
  int rc = resize_unary_elementwise_output_tensor(in_dims, 11 + static_cast<size_t>(argc), out);
  volatile size_t keep = out->shape.dim[0] + static_cast<size_t>(rc);
  (void)keep;
  delete out;
  return 0;
}
