#include "Halide.h"

namespace {

class Distributed : public Halide::Generator<Distributed> {
public:
    Input<Buffer<int>> input{"input", 1};
    Output<Buffer<int>> output{"output", 1};

    void generate() {
        output(x) = 2 * input(x) + 1;
    }

    void schedule() {
    	Var xo("xo"), xi("xi");
        output.split(x, xo, xi, 64)
              .distribute(xo);
    }

private:
    Var x{"x"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Distributed, distributed)
