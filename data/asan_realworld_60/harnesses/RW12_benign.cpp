#include <cstddef>
#include <cstdint>
#include <vector>

struct ExtraChannelInfo { uint32_t bits_per_sample; };
struct PackedFrame { std::vector<uint8_t> extra_channels; };
struct PackedPixelFile {
  std::vector<ExtraChannelInfo> extra_channels_info;
  std::vector<PackedFrame> frames;
  bool Validate() const {
    for (const auto& frame : frames) {
      if (frame.extra_channels.size() != extra_channels_info.size()) return false;
    }
    return true;
  }
};

bool Encode(const PackedPixelFile& ppf, uint32_t* total_bits) {
  if (!ppf.Validate()) return false;
  *total_bits = 0;
  for (const auto& frame : ppf.frames) {
    for (size_t i = 0; i < frame.extra_channels.size(); ++i) {
      *total_bits += ppf.extra_channels_info.data()[i].bits_per_sample;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  PackedPixelFile ppf{{{8}}, {{{1, 2, 3, 4}}}};
  uint32_t bits = 0;
  volatile bool ok = Encode(ppf, &bits);
  (void)ok;
  return 0;
}
