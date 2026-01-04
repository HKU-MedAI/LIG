#include <iostream>
#include <algorithm>
#include "forward.cuh"
#include "backward.cuh"
#include "helpers.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/**
 * Backward pass for weighted sum rasterization with gradient-aware upscaling
 * 
 * ============================================================================
 * MATHEMATICAL DERIVATION
 * ============================================================================
 * 
 * Forward outputs (all per-pixel):
 *   I(x,y)      = Σ c_i * α_i           -- color image
 *   ∂I/∂x       = Σ c_i * α_x,i         -- image x-gradient  
 *   ∂I/∂y       = Σ c_i * α_y,i         -- image y-gradient
 *   ∂²I/∂x∂y    = Σ c_i * α_xy,i        -- mixed partial
 *   W           = Σ α_i                 -- weight sum
 * 
 * where for each gaussian i:
 *   d = [px - μ_x, py - μ_y]            -- pixel-to-center offset
 *   σ = 0.5*(a*d_x² + c*d_y²) + b*d_x*d_y   -- quadratic form (conic = {a,b,c})
 *   α = exp(-σ)                         -- gaussian weight
 *   
 *   σ_x = ∂σ/∂x = a*d_x + b*d_y
 *   σ_y = ∂σ/∂y = b*d_x + c*d_y  
 *   σ_xy = ∂²σ/∂x∂y = b
 *   
 *   α_x  = ∂α/∂x = -α * σ_x
 *   α_y  = ∂α/∂y = -α * σ_y
 *   α_xy = ∂²α/∂x∂y = α * (σ_x * σ_y - σ_xy)
 * 
 * Incoming gradients from loss (via spline upscaler):
 *   v_I   = ∂L/∂I
 *   v_dx  = ∂L/∂(∂I/∂x)
 *   v_dy  = ∂L/∂(∂I/∂y)
 *   v_dxy = ∂L/∂(∂²I/∂x∂y)
 *   v_W   = ∂L/∂W
 * 
 * ============================================================================
 * GRADIENTS W.R.T. COLOR c_k
 * ============================================================================
 * 
 * ∂L/∂c_k = v_I * α_k + v_dx * α_x,k + v_dy * α_y,k + v_dxy * α_xy,k
 * 
 * ============================================================================
 * GRADIENTS W.R.T. GAUSSIAN PARAMETERS (via chain rule through α)
 * ============================================================================
 * 
 * Intermediate "virtual gradients" on α and its derivatives:
 *   v_α   = dot(v_I, c) + v_W
 *   v_αx  = dot(v_dx, c)
 *   v_αy  = dot(v_dy, c)
 *   v_αxy = dot(v_dxy, c)
 * 
 * ============================================================================
 * GRADIENTS W.R.T. POSITION μ
 * ============================================================================
 * 
 * Note: d = pixel - μ, so ∂d/∂μ = -1
 * 
 * ∂α/∂μ_x = α * σ_x           (since ∂σ/∂μ_x = -σ_x)
 * ∂α/∂μ_y = α * σ_y
 * 
 * ∂α_x/∂μ_x = α * (a - σ_x²)
 * ∂α_x/∂μ_y = α * (b - σ_x * σ_y)
 * ∂α_y/∂μ_x = α * (b - σ_x * σ_y)
 * ∂α_y/∂μ_y = α * (c - σ_y²)
 * 
 * ∂α_xy/∂μ_x = α * (σ_x² * σ_y - 2*b*σ_x - a*σ_y)
 * ∂α_xy/∂μ_y = α * (σ_x * σ_y² - 2*b*σ_y - c*σ_x)
 * 
 * Total:
 * ∂L/∂μ_x = v_α * α * σ_x 
 *         + v_αx * α * (a - σ_x²)
 *         + v_αy * α * (b - σ_x * σ_y)
 *         + v_αxy * α * (σ_x² * σ_y - 2*b*σ_x - a*σ_y)
 * 
 * ∂L/∂μ_y = v_α * α * σ_y
 *         + v_αx * α * (b - σ_x * σ_y)
 *         + v_αy * α * (c - σ_y²)
 *         + v_αxy * α * (σ_x * σ_y² - 2*b*σ_y - c*σ_x)
 * 
 * ============================================================================
 * GRADIENTS W.R.T. CONIC (inverse covariance) {a, b, c}
 * ============================================================================
 * 
 * ∂σ/∂a = 0.5*d_x²,  ∂σ/∂b = d_x*d_y,  ∂σ/∂c = 0.5*d_y²
 * ∂σ_x/∂a = d_x,     ∂σ_x/∂b = d_y,    ∂σ_x/∂c = 0
 * ∂σ_y/∂a = 0,       ∂σ_y/∂b = d_x,    ∂σ_y/∂c = d_y
 * ∂σ_xy/∂b = 1       (others = 0)
 * 
 * ∂α/∂a = -α * 0.5 * d_x²
 * ∂α/∂b = -α * d_x * d_y
 * ∂α/∂c = -α * 0.5 * d_y²
 * 
 * ∂α_x/∂a = α * (0.5 * d_x² * σ_x - d_x)
 * ∂α_x/∂b = α * (d_x * d_y * σ_x - d_y)
 * ∂α_x/∂c = α * (0.5 * d_y² * σ_x)
 * 
 * ∂α_y/∂a = α * (0.5 * d_x² * σ_y)
 * ∂α_y/∂b = α * (d_x * d_y * σ_y - d_x)
 * ∂α_y/∂c = α * (0.5 * d_y² * σ_y - d_y)
 * 
 * Let P = σ_x * σ_y - b (the term in α_xy = α * P)
 * ∂α_xy/∂a = α * (-0.5 * d_x² * P + d_x * σ_y)
 * ∂α_xy/∂b = α * (-d_x * d_y * P + d_y * σ_y + d_x * σ_x - 1)
 * ∂α_xy/∂c = α * (-0.5 * d_y² * P + d_y * σ_x)
 * 
 * Total for each conic component combines all four gradient paths.
 */

__global__ void gradient_aware_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const int* __restrict__ final_index,
    // Incoming gradients from loss (5 channels)
    const float3* __restrict__ v_output,      // ∂L/∂I
    const float3* __restrict__ v_output_dx,   // ∂L/∂(∂I/∂x)
    const float3* __restrict__ v_output_dy,   // ∂L/∂(∂I/∂y)
    const float3* __restrict__ v_output_dxy,  // ∂L/∂(∂²I/∂x∂y)
    const float* __restrict__ v_output_wsum,  // ∂L/∂W
    // Output gradients w.r.t. gaussian parameters
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);
    const bool inside = (i < img_size.y && j < img_size.x);
    const int bin_final = inside ? final_index[pix_id] : 0;

    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float2 xy_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float opacity_batch[MAX_BLOCK_SIZE];

    // Load all incoming gradients for this pixel
    const float3 v_I   = v_output[pix_id];
    const float3 v_dx  = v_output_dx[pix_id];
    const float3 v_dy  = v_output_dy[pix_id];
    const float3 v_dxy = v_output_dxy[pix_id];
    const float  v_W   = v_output_wsum[pix_id];

    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (int batch = 0; batch < num_batches; ++batch) {
        block.sync();

        // Load gaussians back-to-front
        const int batch_end = range.y - 1 - block_size * batch;
        const int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            opacity_batch[tr] = opacities ? opacities[g_id] : 1.0f;
        }
        block.sync();

        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside && (batch_end - t <= bin_final);
            
            // Gaussian parameters
            float alpha = 0.f;
            float vis = 0.f;
            float2 d = {0.f, 0.f};
            float3 conic = {0.f, 0.f, 0.f};
            float sigma_x = 0.f, sigma_y = 0.f;
            
            if (valid) {
                conic = conic_batch[t];
                const float2 xy = xy_batch[t];
                d = {px - xy.x, py - xy.y};
                
                const float sigma = 0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) 
                                  + conic.y * d.x * d.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opacity_batch[t] * vis);
                
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                } else {
                    // First derivatives of sigma (Eq. 37-38 derivatives)
                    sigma_x = conic.x * d.x + conic.y * d.y;  // ∂σ/∂x
                    sigma_y = conic.y * d.x + conic.z * d.y;  // ∂σ/∂y
                }
            }

            if (!warp.any(valid)) continue;

            // Initialize local gradient accumulators
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            if (valid) {
                const float3 c = rgbs_batch[t];
                const float a = conic.x, b = conic.y, cc = conic.z;
                
                // Mixed partial: σ_xy = b
                const float sigma_xy = b;
                
                // α derivatives (from forward pass formulas)
                const float alpha_x  = -alpha * sigma_x;
                const float alpha_y  = -alpha * sigma_y;
                const float alpha_xy = alpha * (sigma_x * sigma_y - sigma_xy);
                
                // ================================================================
                // GRADIENT W.R.T. COLOR (Section: Gradients w.r.t. color c_k)
                // ∂L/∂c = v_I * α + v_dx * α_x + v_dy * α_y + v_dxy * α_xy
                // ================================================================
                v_rgb_local.x = v_I.x * alpha + v_dx.x * alpha_x + v_dy.x * alpha_y + v_dxy.x * alpha_xy;

                v_rgb_local.y = v_I.y * alpha + v_dx.y * alpha_x + v_dy.y * alpha_y + v_dxy.y * alpha_xy;
                v_rgb_local.z = v_I.z * alpha + v_dx.z * alpha_x + v_dy.z * alpha_y + v_dxy.z * alpha_xy;
                
                // ================================================================
                // VIRTUAL GRADIENTS (Section: Gradients w.r.t. gaussian parameters)
                // ================================================================
                const float v_alpha = (v_I.x * c.x + v_I.y * c.y + v_I.z * c.z) + v_W;
                const float v_alpha_x = v_dx.x * c.x + v_dx.y * c.y + v_dx.z * c.z;
                const float v_alpha_y = v_dy.x * c.x + v_dy.y * c.y + v_dy.z * c.z;
                const float v_alpha_xy = v_dxy.x * c.x + v_dxy.y * c.y + v_dxy.z * c.z;
                
                // ================================================================
                // GRADIENT W.R.T. POSITION μ (Section: Gradients w.r.t. position)
                // Note: d = pixel - μ, so gradients w.r.t. μ have opposite sign
                // ================================================================
                const float sx = sigma_x, sy = sigma_y;
                const float sx2 = sx * sx, sy2 = sy * sy;
                const float sxsy = sx * sy;
                
                // ∂L/∂μ_x (note: result negated because d = px - μ)
                v_xy_local.x = alpha * (
                    v_alpha * sx
                    + v_alpha_x * (a - sx2)
                    + v_alpha_y * (b - sxsy)
                    + v_alpha_xy * (sx2 * sy - 2.f * b * sx - a * sy)
                );
                
                // ∂L/∂μ_y
                v_xy_local.y = alpha * (
                    v_alpha * sy
                    + v_alpha_x * (b - sxsy)
                    + v_alpha_y * (cc - sy2)
                    + v_alpha_xy * (sx * sy2 - 2.f * b * sy - cc * sx)
                );
                
                // ================================================================
                // GRADIENT W.R.T. CONIC {a, b, c} (Section: Gradients w.r.t. conic)
                // ================================================================
                const float dx = d.x, dy = d.y;
                const float dx2 = dx * dx, dy2 = dy * dy;
                const float dxdy = dx * dy;
                const float P = sxsy - b;  // term in α_xy = α * P
                
                // ∂L/∂a
                v_conic_local.x = alpha * (
                    v_alpha * (-0.5f * dx2)
                    + v_alpha_x * (0.5f * dx2 * sx - dx)
                    + v_alpha_y * (0.5f * dx2 * sy)
                    + v_alpha_xy * (-0.5f * dx2 * P + dx * sy)
                );
                
                // ∂L/∂b
                v_conic_local.y = alpha * (
                    v_alpha * (-dxdy)
                    + v_alpha_x * (dxdy * sx - dy)
                    + v_alpha_y * (dxdy * sy - dx)
                    + v_alpha_xy * (-dxdy * P + dy * sy + dx * sx - 1.f)
                );
                
                // ∂L/∂c
                v_conic_local.z = alpha * (
                    v_alpha * (-0.5f * dy2)
                    + v_alpha_x * (0.5f * dy2 * sx)
                    + v_alpha_y * (0.5f * dy2 * sy - dy)
                    + v_alpha_xy * (-0.5f * dy2 * P + dy * sx)
                );
                
                // v_opacity = d(alpha)/d(opacity) * v_alpha = vis * v_alpha
                v_opacity_local = vis * v_alpha;
            }
            
            // Warp-level reduction
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_xy_abs_ptr = (float*)(v_xy_abs);
                atomicAdd(v_xy_abs_ptr + 2*g + 0, fabsf(v_xy_local.x));
                atomicAdd(v_xy_abs_ptr + 2*g + 1, fabsf(v_xy_local.y));

                if (v_opacity) {
                    atomicAdd(v_opacity + g, v_opacity_local);
                }
            }
        }
    }
}

__global__ void gradient_aware_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    float* __restrict__ out_wsum,
    float3* __restrict__ out_dx,
    float3* __restrict__ out_dy,
    float3* __restrict__ out_dxy
) {

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j + 0.5;
    float py = (float)i + 0.5;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    // __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float2 xy_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float opacity_batch[MAX_BLOCK_SIZE];

    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    float3 pix_dx = {0.f, 0.f, 0.f};
    float3 pix_dy = {0.f, 0.f, 0.f};
    float3 pix_dxy = {0.f, 0.f, 0.f};
    float pix_wsum = 0.f;
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
            opacity_batch[tr] = opacities ? opacities[g_id] : 1.0f;
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float2 xy = xy_batch[t];
            // d = [x - μ_x, y - μ_y] (Eq. 38)
            const float2 d = {px - xy.x, py - xy.y};
            const float sigma = 0.5f * (conic.x * d.x * d.x +
                                        conic.z * d.y * d.y) +
                                conic.y * d.x * d.y;
            const float opacity = opacity_batch[t];
            const float alpha = min(0.999f, opacity * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            // First-order partial derivatives of sigma = -g (Eq. 37: g = -d^T * Σ^{-1} * d)
            // sigma = 0.5 * d^T * conic * d, where conic = Σ^{-1}
            // ∂sigma/∂x = conic.x * d.x + conic.y * d.y
            // ∂sigma/∂y = conic.y * d.x + conic.z * d.y
            const float d_sigma_dx = conic.x * d.x + conic.y * d.y;
            const float d_sigma_dy = conic.y * d.x + conic.z * d.y;

            // ∂²sigma/∂x∂y = conic.y (the off-diagonal element of the conic matrix)
            const float d2_sigma_dxdy = conic.y;

            // α = exp(-sigma) (Eq. 36), ∂α/∂x = -α * ∂sigma/∂x
            const float d_alpha_dx = -alpha * d_sigma_dx;
            const float d_alpha_dy = -alpha * d_sigma_dy;

            // ∂²α/∂x∂y = -α * (∂²sigma/∂x∂y - ∂sigma/∂x * ∂sigma/∂y)
            const float d2_alpha_dxdy = -alpha * (d2_sigma_dxdy - d_sigma_dx * d_sigma_dy);

            int32_t g = id_batch[t];
            const float3 c = colors[g];
            pix_out += c * alpha;

            // Accumulate gradients for weighted summation (simplified from Eqs. 46-49)
            // For weighted sum: I(x,y) = Σ c_i * α_i(x,y)
            // ∂I/∂x = Σ c_i * ∂α_i/∂x (no alpha-compositing terms)
            pix_dx += c * d_alpha_dx;

            // ∂I/∂y = Σ c_i * ∂α_i/∂y
            pix_dy += c * d_alpha_dy;

            // ∂²I/∂x∂y = Σ c_i * ∂²α_i/∂x∂y
            pix_dxy += c * d2_alpha_dxdy;

            // Accumulate weighted sum of alphas
            pix_wsum += alpha;

            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        out_img[pix_id] = pix_out;
        out_wsum[pix_id] = pix_wsum;
        out_dx[pix_id] = pix_dx;
        out_dy[pix_id] = pix_dy;
        out_dxy[pix_id] = pix_dxy;
    }
}