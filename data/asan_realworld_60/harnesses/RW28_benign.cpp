#include <cstdint>

struct NodeDef { int op_kind; };
struct PushDownCtx {
  NodeDef* left_leaf;
  NodeDef* right_leaf;
  bool left_leaf_is_const;
  bool right_leaf_is_const;
};

static inline bool IsReallyConstant(const NodeDef& n) { return n.op_kind == 1; }

bool PrepareConstantPushDown(NodeDef* left_in, NodeDef* right_in, PushDownCtx* ctx) {
  ctx->left_leaf = left_in;
  ctx->right_leaf = right_in;
  if (ctx->left_leaf == nullptr || ctx->right_leaf == nullptr) {
    return false;
  }
  ctx->left_leaf_is_const = IsReallyConstant(*ctx->left_leaf);
  ctx->right_leaf_is_const = IsReallyConstant(*ctx->right_leaf);
  return ctx->left_leaf_is_const || ctx->right_leaf_is_const;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  NodeDef b{1};
  PushDownCtx ctx{};
  (void)PrepareConstantPushDown(nullptr, &b, &ctx);
  return 0;
}
