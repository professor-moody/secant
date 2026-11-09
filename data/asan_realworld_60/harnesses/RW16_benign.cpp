#include <cstddef>
#include <cstdint>

struct LarkParser {
  const char* src_;
  size_t pos_ = 0;
  int group_depth_ = 0;
  static constexpr int kMaxGroupDepth = 200;

  long ParseAtom() {
    long total = 0;
    if (group_depth_ >= kMaxGroupDepth) return -1;
    while (src_[pos_] != '\0' && src_[pos_] != '}') {
      if (src_[pos_] == '{') {
        ++pos_;
        ++group_depth_;
        long sub = ParseAtom();
        --group_depth_;
        if (sub < 0) return -1;
        total += sub + 1;
        if (src_[pos_] == '}') ++pos_;
      } else {
        ++pos_;
        ++total;
      }
    }
    return total;
  }
};

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  size_t n = 1000000;
  char* deep = new char[n + 1];
  for (size_t i = 0; i < n; ++i) deep[i] = '{';
  deep[n] = '\0';
  LarkParser p{deep, 0, 0};
  volatile long rc = p.ParseAtom();
  (void)rc;
  delete[] deep;
  return 0;
}
