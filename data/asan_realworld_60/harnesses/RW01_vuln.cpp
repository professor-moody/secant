#include <cstddef>
#include <cstdint>

struct CFF_VarRegion { int32_t axisList[2]; };
struct CFF_VarData { uint32_t regionIdxCount; const uint16_t* regionIndices; };
struct CFF_VStore { uint32_t axisCount; uint32_t regionCount; CFF_VarRegion* varRegionList; };

int cff_blend_build_vector(const CFF_VStore* vs, const CFF_VarData* varData,
                           uint32_t lenNDV, const int32_t* ndv, int32_t* blend_vec) {
  blend_vec[0] = 0x10000;
  for (uint32_t master = 1; master <= varData->regionIdxCount; ++master) {
    uint32_t idx = varData->regionIndices[master - 1];
    CFF_VarRegion* varRegion = &vs->varRegionList[idx];
    if (idx >= vs->regionCount) {
      return -1;
    }
    if (!lenNDV) {
      blend_vec[master] = 0;
      continue;
    }
    int32_t scalar = 0x10000;
    for (uint32_t axis = 0; axis < vs->axisCount; ++axis) {
      if (varRegion->axisList[axis] > ndv[axis]) scalar = 0;
    }
    blend_vec[master] = scalar;
  }
  return 0;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  uint16_t indices[1] = {1};
  CFF_VarData vd = {1, indices};
  CFF_VStore vs = {1, 0, nullptr};
  int32_t ndv[1] = {0x4000};
  int32_t out[2] = {0, 0};
  (void)cff_blend_build_vector(&vs, &vd, 1, ndv, out);
  return 0;
}
