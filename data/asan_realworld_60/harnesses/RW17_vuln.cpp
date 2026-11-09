#include <cassert>
#include <cstddef>
#include <vector>

struct InferenceContext {
  std::vector<int> inputs_;
  size_t getNumInputs() const { return inputs_.size(); }
  int getInputType(size_t i) const { return inputs_.data()[i]; }
};

int LoopInferenceFunction(const InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  assert(num_inputs >= 2);
  auto num_loop_state_vars = num_inputs - 2;
  int acc = 0;
  for (size_t i = 0; i < num_loop_state_vars; ++i) {
    acc += ctx.getInputType(i + 2);
  }
  return acc;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  InferenceContext ctx{{1}};
  (void)LoopInferenceFunction(ctx);
  return 0;
}
