#include <iostream>
#include "include/argparse.hpp"
#include "include/LabelDet.h"

int run(std::string directory) {

    LabelDet app(directory);
    app.run();
    return 0;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("label det");
    program.add_argument("p")
        .default_value(std::string("."))
        .required()
        .help("specify the root.");
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    std::cout << program.get("p") << std::endl;
    std::string directory = program.get("p");
    run(directory);
}