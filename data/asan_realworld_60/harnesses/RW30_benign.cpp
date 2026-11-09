#include <cstddef>
#include <vector>

struct SubGraph { int num_tensors; };
struct Model {
  const std::vector<SubGraph>* subgraphs_;
  const std::vector<SubGraph>* subgraphs() const { return subgraphs_; }
};

int BuildInterpreter(const Model* model, int* out_subgraph_count) {
  auto* subgraphs = model->subgraphs();
  if (!subgraphs || subgraphs->empty()) {
    return -1;
  }
  *out_subgraph_count = static_cast<int>(subgraphs->size());
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  Model m{nullptr};
  int cnt = 0;
  (void)BuildInterpreter(&m, &cnt);
  return 0;
}
