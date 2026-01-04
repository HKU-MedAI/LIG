#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward_tensor);
    m.def("rasterize_backward", &rasterize_backward_tensor);
    m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
    m.def("project_gaussians_backward", &project_gaussians_backward_tensor);
    m.def("project_gaussians_forward_cholesky", &project_gaussians_forward_cholesky_tensor);
    m.def("project_gaussians_backward_cholesky", &project_gaussians_backward_cholesky_tensor);
    
    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor);
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);

    m.def("gradient_aware_upscale_forward", &gradient_aware_upscale_forward_tensor);
    m.def("gradient_aware_upscale_backward", &gradient_aware_upscale_backward_tensor);
}
