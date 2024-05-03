#include "libMkcfup.h"

int main(int argc, char *argv[]) {
    // The first param is the input dir, the second is the output txt dir.

    // Before execute this program, you should make sure:
    // images exist, `results_item.txt`, `groundtruth_rect.txt`, `frames.txt` exist.
    std::string input_dir = "";
    std::string output_dir = "";
    std::string item = "";
    if (argc > 1) {
        input_dir = std::string(argv[1]);
        output_dir = std::string(argv[2]);
        item = std::string(argv[3]);
    }
    std::cout << input_dir << std::endl;
    std::cout << output_dir << std::endl;
    std::cout << item << std::endl;

    inference(input_dir, output_dir, item);
    return 0;
}
