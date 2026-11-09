#include <cstdint>

struct avifRWStream { uint32_t written; };
struct avifImage { uint16_t maxCLL; uint16_t maxPALL; };

int avifEncoderWriteHDRProperties(avifRWStream* dedupStream,
                                  avifRWStream* outputStream,
                                  const avifImage* imageMetadata) {
  (void)outputStream;
  if (imageMetadata->maxCLL || imageMetadata->maxPALL) {
    dedupStream->written += 4;
  }
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  avifRWStream out = {0};
  avifImage img = {1000, 400};
  (void)avifEncoderWriteHDRProperties(nullptr, &out, &img);
  return 0;
}
