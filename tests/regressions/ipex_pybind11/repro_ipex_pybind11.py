import tempfile
import torch
from intel_extension_for_pytorch.xpu import cpp_extension as xpu_ext

print("torch:", torch.__version__)
print("has _PYBIND11_COMPILER_TYPE:", hasattr(torch._C, "_PYBIND11_COMPILER_TYPE"))

cpp = r"""#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "add");
}
"""


with tempfile.TemporaryDirectory(prefix="ipex-pybind11-repro-") as work_dir:
    src_path = f"{work_dir}/pyb11_repro.cpp"
    with open(src_path, "w", encoding="utf-8") as source:
        source.write(cpp)

    mod = xpu_ext.load(
        name="pyb11_repro",
        sources=[src_path],
        extra_cflags=["-O0"],
        build_directory=work_dir,
        verbose=True,
        is_python_module=True,
        keep_intermediates=False,
    )
    actual = mod.add(torch.ones(1), torch.ones(1))
    torch.testing.assert_close(actual, torch.full((1,), 2.0))
    print("PASS built module:", mod)
