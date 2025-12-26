#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// compute the 2d gaussian parameters from 3d gaussian parameters
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ cov2d,
    const float2* __restrict__ means2d,
    const dim3 tile_bounds,
    const unsigned block_width,
    float2* __restrict__ xys,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
);

// compute output color image from binned and sorted gaussians
__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    float* __restrict__ out_wsum
);

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
);

__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    int* __restrict__ final_index,
    float3* __restrict__ out_img
);

// compute output color image from binned and sorted gaussians
__global__ void gradient_aware_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,  // optional, can be nullptr
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    float* __restrict__ out_wsum,
    float3* __restrict__ out_dx,
    float3* __restrict__ out_dy,
    float3* __restrict__ out_dxy
);
