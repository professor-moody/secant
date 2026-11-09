#include <cstddef>
#include <cstdint>
#include <cstring>

struct LZ4F_dctx {
  uint64_t contentSize;
  uint64_t frameRemainingSize;
};

int LZ4F_decodeBlock(LZ4F_dctx* dctx, uint8_t* dst, const uint8_t* src, size_t sizeToCopy) {
  if (dctx->contentSize && static_cast<uint64_t>(sizeToCopy) > dctx->frameRemainingSize) {
    return -1;
  }
  if (dctx->contentSize) {
    dctx->frameRemainingSize -= static_cast<uint64_t>(sizeToCopy);
  }
  std::memcpy(dst, src, sizeToCopy);
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  LZ4F_dctx dctx = {16, 16};
  uint8_t* dst = new uint8_t[16]{};
  uint8_t src[32] = {};
  int rc = LZ4F_decodeBlock(&dctx, dst, src, 31 + static_cast<size_t>(argc));
  volatile uint8_t keep = dst[0] + static_cast<uint8_t>(rc);
  (void)keep;
  delete[] dst;
  return 0;
}
