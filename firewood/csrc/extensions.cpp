#include <torch/extension.h>
#include "biased_activations/extension.h"
#include "upfirdn2d/extension.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bias_act", &bias_act,
          py::arg("x"), py::arg("b"), py::arg("xref"), py::arg("yref"),
          py::arg("dy"), py::arg("grad"), py::arg("dim"), py::arg("act"),
          py::arg("alpha"), py::arg("gain"), py::arg("clamp"));
    m.def("upfirdn2d", &upfirdn2d,
          py::arg("x"), py::arg("f"), py::arg("upx"), py::arg("upy"),
          py::arg("downx"), py::arg("downy"), py::arg("padx0"), py::arg("padx1"),
          py::arg("pady0"), py::arg("pady1"), py::arg("flip"), py::arg("gain"));
}
