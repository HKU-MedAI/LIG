#include "cuda_runtime.h"
#include "forward.cuh"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &A);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &cov2d,
    torch::Tensor &means2d,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width
);

std::tuple<
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_cholesky_tensor(
    const int num_points,
    torch::Tensor &cholesky,
    torch::Tensor &means2d,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width
);

std::tuple<torch::Tensor, torch::Tensor>
project_gaussians_backward_cholesky_tensor(
    const int num_points,
    torch::Tensor &radii,
    torch::Tensor &cholesky,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
);

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds,
    const unsigned block_width
);

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    const torch::Tensor &isect_ids_sorted,
    const std::tuple<int, int, int> tile_bounds
);

std::tuple<
    torch::Tensor, // output img
    torch::Tensor, // output wsum
    torch::Tensor, // output dx
    torch::Tensor, // output dy
    torch::Tensor, // output dxy
    torch::Tensor // output final_idx
> rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    bool compute_upscale_gradients = true
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dxy_abs
        torch::Tensor, // dL_dconic
        torch::Tensor // dL_dcolors
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned block_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output,
        const torch::Tensor &v_render_wsum,
        const c10::optional<torch::Tensor> &v_output_dx,
        const c10::optional<torch::Tensor> &v_output_dy,
        const c10::optional<torch::Tensor> &v_output_dxy
    );
torch::Tensor gradient_aware_upscale_forward_tensor(
    const torch::Tensor &render,    // [H, W, 3]
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi  // x1, y1, x2, y2
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gradient_aware_upscale_backward_tensor(
    const torch::Tensor &grad_output,  // [dst_h, dst_w, 3]
    const torch::Tensor &render,       // [H, W, 3]
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &dxy,
    int dst_h,
    int dst_w,
    const std::tuple<float, float, float, float> &roi
);