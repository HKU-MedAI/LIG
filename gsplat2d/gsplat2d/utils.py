"""Python bindings for binning and sorting gaussians"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

import gsplat2d.cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    extents: Float[Tensor, "batch 2"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    block_size: int,
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        extents.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        block_size,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_intersects: int,
    isect_ids_sorted: Int[Tensor, "num_intersects 1"],
    tile_bounds: Tuple[int, int, int],
) -> Int[Tensor, "num_intersects 2"]:

    return _C.get_tile_bin_edges(
        num_intersects, isect_ids_sorted.contiguous(), tile_bounds
    )


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"],
    opacities: Optional[Float[Tensor, "batch"]] = None,
) -> Tuple[Float[Tensor, "batch 3"], Float[Tensor, "batch 2"]]:

    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous(),
        opacities.contiguous() if opacities is not None else None)


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:

    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    extents: Float[Tensor, "batch 2"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    block_size: int,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:

    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys,
        depths,
        extents,
        cum_tiles_hit,
        tile_bounds,
        block_size,
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
