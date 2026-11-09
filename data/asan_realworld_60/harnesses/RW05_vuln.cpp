#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static constexpr uint32_t PNG_FLAG_ZSTREAM_ENDED = 0x0008U;
static constexpr uint32_t PNG_FLAG_PROCESS_DATA_PAUSED = 0x0010U;

struct png_struct {
  uint32_t flags;
  const uint8_t* next_in;
  uint32_t avail_in;
  uint8_t* save_buffer;
  uint8_t last_byte;
  void (*row_fn)(png_struct*);
};

static inline void png_push_process_row(png_struct* png_ptr) {
  if (png_ptr->row_fn) png_ptr->row_fn(png_ptr);
}

void png_process_IDAT_data(png_struct* png_ptr, const uint8_t* buffer, size_t buffer_length) {
  png_ptr->next_in = buffer;
  png_ptr->avail_in = static_cast<uint32_t>(buffer_length);
  while (png_ptr->avail_in > 0 && !(png_ptr->flags & PNG_FLAG_ZSTREAM_ENDED)) {
    png_ptr->last_byte ^= *png_ptr->next_in++;
    png_ptr->avail_in--;
    png_push_process_row(png_ptr);
  }
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  uint8_t* buf = new uint8_t[4]{1, 2, 3, 4};
  auto pause_cb = +[](png_struct* p) {
    p->flags |= PNG_FLAG_PROCESS_DATA_PAUSED;
    uint8_t* next = new uint8_t[16]{};
    delete[] p->save_buffer;
    p->save_buffer = next;
  };
  png_struct s = {0, nullptr, 0, buf, 0, pause_cb};
  png_process_IDAT_data(&s, s.save_buffer, 4);
  delete[] s.save_buffer;
  return 0;
}
