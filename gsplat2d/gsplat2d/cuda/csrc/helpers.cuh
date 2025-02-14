#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <iostream>

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
    const float pix_radius,
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
    float2 tile_radius = {
        pix_radius / (float)block_size, pix_radius / (float)block_size
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
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
