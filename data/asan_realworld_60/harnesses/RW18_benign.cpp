#include <cstdint>

int ReadDarknetConvGroups(int channels, int filters, int groups, int* out_in_per_group) {
  if (channels <= 0 || filters <= 0) return -1;
  if (groups <= 0) return -1;
  if (channels % groups != 0) return -1;
  *out_in_per_group = channels / groups;
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  int out = 0;
  (void)ReadDarknetConvGroups(6, 4, 0, &out);
  return 0;
}
