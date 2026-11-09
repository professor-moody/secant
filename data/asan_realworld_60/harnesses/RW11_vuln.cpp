#include <cstddef>
#include <cstdint>
#include <vector>

struct FieldDef { uint16_t offset; bool deprecated; };
struct Table {
  const uint8_t* data;
  bool CheckField(uint16_t off) const { return off != 0; }
  uint8_t GetField(uint16_t off) const { return data[off]; }
};

int GenStruct(const std::vector<FieldDef>& fields, const Table* table) {
  int sum = 0;
  for (auto it = fields.begin(); it != fields.end(); ++it) {
    const FieldDef& fd = *it;
    bool is_present = table->CheckField(fd.offset);
    if (is_present) {
      sum += table->GetField(fd.offset);
    }
  }
  return sum;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  uint8_t* buf = new uint8_t[8]{10, 20, 30, 40, 0, 0, 0, 0};
  Table t{buf};
  std::vector<FieldDef> fields = {{1, false}, {24, true}};
  volatile int s = GenStruct(fields, &t);
  (void)s;
  delete[] buf;
  return 0;
}
