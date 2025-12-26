#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void gradient_aware_upscale_kernel(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float3* __restrict__ render,   // [H, W] of float3 (HWC)
    const float3* __restrict__ dx,
    const float3* __restrict__ dy,
    const float3* __restrict__ dxy,
    float3* __restrict__ output          // [dst_h, dst_w] of float3 (HWC)
);

__global__ void gradient_aware_upscale_backward_kernel(
    const int dst_h,
    const int dst_w,
    const int src_h,
    const int src_w,
    const float roi_x1,
    const float roi_y1,
    const float roi_x2,
    const float roi_y2,
    const float3* __restrict__ grad_output,  // [dst_H, dst_W] of float3
    float3* __restrict__ grad_render,        // [src_H, src_W] of float3
    float3* __restrict__ grad_dx,
    float3* __restrict__ grad_dy,
    float3* __restrict__ grad_dxy
);