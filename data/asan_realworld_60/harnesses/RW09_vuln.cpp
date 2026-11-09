#include <cstdint>
#include <cstdlib>

struct avifCodec { int maxThreads; };
struct avifTile { avifCodec* codec; };
struct avifDecoderData { avifTile tiles[2]; uint32_t tileCount; };

static inline void avifDecoderDataResetCodec(avifDecoderData* data) {
  for (uint32_t i = 0; i < data->tileCount; ++i) {
    delete data->tiles[i].codec;
    data->tiles[i].codec = nullptr;
  }
}

int avifDecoderNextImage(avifDecoderData* data, int fail_tile) {
  if (!data->tiles[0].codec) {
    for (uint32_t i = 0; i < data->tileCount; ++i) {
      if (static_cast<int>(i) == fail_tile) return -1;
      data->tiles[i].codec = new avifCodec{1};
    }
  }
  for (uint32_t i = 0; i < data->tileCount; ++i) {
    data->tiles[i].codec->maxThreads = 4;
  }
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  avifDecoderData d = {{{nullptr}, {nullptr}}, 2};
  (void)avifDecoderNextImage(&d, 1);
  (void)avifDecoderNextImage(&d, -1);
  avifDecoderDataResetCodec(&d);
  return 0;
}
