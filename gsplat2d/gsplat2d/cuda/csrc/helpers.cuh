#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>

// SnugBox: alpha threshold (1/255 default)
constexpr __device__ float SNUGBOX_ALPHA_INV = 255.f;

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float2 pix_extent,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max,
    const int block_size
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)block_size, pix_center.y / (float)block_size
    };
    float2 tile_extent = {
        pix_extent.x / (float)block_size, pix_extent.y / (float)block_size
    };
    get_bbox(tile_center, tile_extent, tile_bounds, tile_min, tile_max);
}

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float opacity, float3 &conic, float2 &extent) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // conic = Sigma^-1, upper triangular: [a, b; b, c]
    float a = cov2d.z * inv_det;
    float b = -cov2d.y * inv_det;
    float c = cov2d.x * inv_det;
    conic = {a, b, c};

    // SnugBox (arXiv:2412.00578): tight AABB based on opacity threshold alpha >= 1/SNUGBOX_ALPHA
    // t = 2*log(SNUGBOX_ALPHA_INV*opacity), ellipse threshold
    float t = 2.f * logf(SNUGBOX_ALPHA_INV * opacity);
    if (t <= 0.f) {
        extent = {0.f, 0.f};
        return true;  // opacity too low, Gaussian invisible
    }

    // Eq. 16:
    // extent.x = sqrt(t*c / det_conic)
    // extent.y = sqrt(t*a / det_conic)
    // det(conic) = a*c - b^2 = 1/det(cov2d) = inv_det
    extent.x = ceilf(sqrtf(t * c / inv_det));
    extent.y = ceilf(sqrtf(t * a / inv_det));

    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y / 2.f, v_conic.y / 2.f, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}


// Explicit float3 operations (duplicated from helper_math.h for visibility)
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}
