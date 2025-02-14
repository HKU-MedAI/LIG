#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

namespace cg = cooperative_groups;

__global__ void compute_cov2d_bounds_kernel(
    const unsigned num_pts, const float* __restrict__ covs2d, float* __restrict__ conics, float* __restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    DEVICE_GUARD(covs2d);
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;

    compute_cov2d_bounds_kernel<<<blocks, N_THREADS>>>(
        num_pts,
        covs2d.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>()
    );
    return std::make_tuple(conics, radii);
}

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
) {
    DEVICE_GUARD(cov2d);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = int((img_width + block_width - 1) / block_width);
    tile_bounds_dim3.y = int((img_height + block_width - 1) / block_width);
    tile_bounds_dim3.z = 1;

    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, cov2d.options().dtype(torch::kFloat32));

    torch::Tensor radii_d =
        torch::zeros({num_points}, cov2d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, cov2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, cov2d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)cov2d.contiguous().data_ptr<float>(),
        (float2 *)means2d.contiguous().data_ptr<float>(),
        tile_bounds_dim3,
        block_width,
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, radii_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
){
    DEVICE_GUARD(conics);
    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, conics.options().dtype(torch::kFloat32));

    torch::Tensor v_mean2d =
        torch::zeros({num_points, 2}, conics.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_mean2d);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds,
    const unsigned block_width
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        block_width,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted, 
    const std::tuple<int, int, int> tile_bounds
) {
    DEVICE_GUARD(isect_ids_sorted);
    CHECK_INPUT(isect_ids_sorted);
    int num_tiles = std::get<0>(tile_bounds) * std::get<1>(tile_bounds);
    torch::Tensor tile_bins = torch::zeros(
        {num_tiles, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

std::tuple<torch::Tensor, torch::Tensor>
rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors //,
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>() //,
    );

    return std::make_tuple(out_img, final_idx);
}

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
        const torch::Tensor &v_output //, // dL_dout_color
    ) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + block_width - 1) / block_width,
        (img_height + block_width - 1) / block_width,
        1
    };
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_xy_abs = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());

    rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float2 *)v_xy_abs.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_xy_abs, v_conic, v_colors);
}
