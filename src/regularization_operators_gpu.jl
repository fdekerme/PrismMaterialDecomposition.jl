# ==============================================================================
# CUDA backend
# ------------------------------------------------------------------------------
# CUDA-specific kernels and host wrappers for the similarity operator (W and Wᵀ).
# GPUs via the CUDA.jl API.
# ==============================================================================

"""
W_cross_mul_gpu_kernel_cuda!(y, x, u, p, t)

CUDA device kernel that computes the forward gather-style similarity smoothing for a
single-channel flattened image. For each linear index `idx` the kernel computes an
exponential similarity to neighbouring pixels (within `radius`) using `img_top/img_bottom`
from the parameter tuple `p` and writes a normalized weighted average into `y[idx]`.

Parameters
- y, x : device arrays (flattened column-major) for output and input values.
- u, t  : unused placeholders (kept for SciMLOperators kernel API compatibility).
- p     : parameter tuple unpacked as
    (N::Int32, nrows::Int32, ncols::Int32, nothing, (img_top, img_bottom, h_top::Float32, h_bottom::Float32, radius::Int32)).

Notes
- Falls back to identity (y = x) when no valid neighbors contribute.
"""
function W_cross_mul_gpu_kernel_cuda!(y, x, u, p, t)
    # Unpack parameters
    N::Int32,
    nrows::Int32,
    ncols::Int32,
    nothing,
    (img_top, img_bottom, h_top::Float32, h_bottom::Float32, radius::Int32) = p

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx < 1 || idx > N
        return
    end

    # map linear idx -> (r,c) for column-major storage
    c = Int32(cld(idx, nrows))
    r = Int32(idx - (c - 1) * nrows)
    radius_f = Float32(radius)

    center_val_top = img_top[r, c]
    center_val_bottom = img_bottom[r, c]

    r0 = max(Int32(1), r - radius)
    r1 = min(nrows, r + radius)
    c0 = max(Int32(1), c - radius)
    c1 = min(ncols, c + radius)

    total_weight = 0.0f0
    weighted_sum = 0.0f0
    hh_top = h_top * h_top
    hh_bottom = h_bottom * h_bottom

    @inbounds for cc = c0:c1
        for rr = r0:r1
            if rr == r && cc == c
                continue
            end
            neighbor_val_top = img_top[rr, cc]
            neighbor_val_bottom = img_bottom[rr, cc]
            diff_top = center_val_top - neighbor_val_top
            diff_bottom = center_val_bottom - neighbor_val_bottom
            if (abs(diff_top) < 3.0f0 * h_top) && (abs(diff_bottom) < 3.0f0 * h_bottom)
                dx = Float32(r - rr)
                dy = Float32(c - cc)
                #distance_norm = 1.0f0 / sqrt(dx * dx + dy * dy)   # L2 inverse distance
                distance_norm = exp(-(dx * dx + dy * dy) / (radius_f * radius_f)) # Gaussian distance weight
                s =
                    (
                        exp(-(diff_top * diff_top) / hh_top) *
                        exp(-(diff_bottom * diff_bottom) / hh_bottom)
                    )
                lin = (cc - 1) * nrows + rr
                weighted_sum += s * distance_norm * x[lin]
                total_weight += s * distance_norm
            end
        end
    end

    if total_weight > 0.0f0
        y[idx] = weighted_sum / total_weight
    else
        y[idx] = x[idx]
    end
    return
end


"""
W_I_cross_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)

CUDA host wrapper that applies the forward similarity kernel to the stacked two-channel
vector (top then bottom). Launches `W_cross_mul_gpu_kernel_cuda!` separately for the top and
bottom halves and computes (W - I) * x in-place by subtracting `x_gpu` from the accumulated
result in `y_gpu`.

Parameters
- y_gpu, x_gpu : device arrays of length 2N (top concatenated with bottom).
- u, t         : unused placeholders for API compatibility.
- p            : same parameter tuple described for the kernel; first element `N` is used
                 to slice top/bottom views.
"""
function W_I_cross_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)
    N, _ = p

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @cuda threads = nthreads blocks = nblocks W_cross_mul_gpu_kernel_cuda!(
        top_y,
        top_x,
        nothing,
        p,
        nothing,
    )
    @cuda threads = nthreads blocks = nblocks W_cross_mul_gpu_kernel_cuda!(
        bottom_y,
        bottom_x,
        nothing,
        p,
        nothing,
    )
    y_gpu .= y_gpu .- x_gpu
end

"""
W_I_cross_mul_gpu_cuda!(x_gpu, u, p, t)

CUDA convenience variant that allocates a new device array and returns (W - I) * x.
This overload mirrors the allocate-to-return calling convention used by some SciMLOperators
FunctionOperator methods.

Parameters
- x_gpu : device array input (length 2N, top then bottom)
- u, p, t : unused/parameters consistent with the kernel API; `p` contains N used for
           slicing.
"""
function W_I_cross_mul_gpu_cuda!(x_gpu, u, p, t)
    N, _ = p
    y_gpu = CUDA.zeros(eltype(x_gpu), length(x_gpu))

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @cuda threads = nthreads blocks = nblocks W_cross_mul_gpu_kernel_cuda!(
        top_y,
        top_x,
        nothing,
        p,
        nothing,
    )

    @cuda threads = nthreads blocks = nblocks W_cross_mul_gpu_kernel_cuda!(
        bottom_y,
        bottom_x,
        nothing,
        p,
        nothing,
    )

    y_gpu .= y_gpu .- x_gpu
end


"""
Wt_cross_mul_gpu_kernel_cuda!(y, x, u, p, t)

CUDA device kernel that computes the transpose (scatter-style) of the similarity operator.
For each centre pixel the kernel computes the normalization `total_weight` and then adds
normalized contributions into neighbour positions in `y` using device atomic adds.

Parameters
- y, x : device arrays where `x` contains centre contributions and `y` is updated via atomics.
- p    : same parameter tuple layout as the forward kernel.

Notes
- If a pixel has no valid neighbours the kernel writes an identity contribution for that pixel.
"""
function Wt_cross_mul_gpu_kernel_cuda!(y, x, u, p, t)
    # Unpack parameters
    N::Int32,
    nrows::Int32,
    ncols::Int32,
    nothing,
    (img_top, img_bottom, h_top::Float32, h_bottom::Float32, radius::Int32) = p

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx < 1 || idx > N
        return
    end

    # map linear idx -> (r,c) for column-major storage
    c = Int32(cld(idx, nrows))
    r = Int32(idx - (c - 1) * nrows)
    radius_f = Float32(radius)

    center_val_top = img_top[r, c]
    center_val_bottom = img_bottom[r, c]

    r0 = max(Int32(1), r - radius)
    r1 = min(nrows, r + radius)
    c0 = max(Int32(1), c - radius)
    c1 = min(ncols, c + radius)

    # First pass: compute sum of unnormalized similarities for this row (center)
    total_weight = 0.0f0
    hh_top = h_top * h_top
    hh_bottom = h_bottom * h_bottom

    @inbounds for cc = c0:c1
        for rr = r0:r1
            if rr == r && cc == c
                continue
            end
            neighbor_val_top = img_top[rr, cc]
            neighbor_val_bottom = img_bottom[rr, cc]
            diff_top = center_val_top - neighbor_val_top
            diff_bottom = center_val_bottom - neighbor_val_bottom
            if (abs(diff_top) < 3.0f0 * h_top) && (abs(diff_bottom) < 3.0f0 * h_bottom)
                dx = Float32(r - rr)
                dy = Float32(c - cc)
                #distance_norm = 1.0f0 / sqrt(dx * dx + dy * dy)   # L2 inverse distance
                distance_norm = exp(-(dx * dx + dy * dy) / (radius_f * radius_f)) # Gaussian distance weight
                s =
                    (
                        exp(-(diff_top * diff_top) / hh_top) *
                        exp(-(diff_bottom * diff_bottom) / hh_bottom)
                    )
                total_weight += s * distance_norm
            end
        end
    end

    # Second pass: normalized contributions into neighbors (atomic adds)
    if total_weight > 0.0f0
        xi = x[idx]  # contribution scalar from center
        @inbounds for cc = c0:c1
            for rr = r0:r1
                if rr == r && cc == c
                    continue
                end
                neighbor_val_top = img_top[rr, cc]
                neighbor_val_bottom = img_bottom[rr, cc]
                diff_top = center_val_top - neighbor_val_top
                diff_bottom = center_val_bottom - neighbor_val_bottom
                if (abs(diff_top) < 3.0f0 * h_top) && (abs(diff_bottom) < 3.0f0 * h_bottom)
                    dx = Float32(r - rr)
                    dy = Float32(c - cc)
                    #distance_norm = 1.0f0 / sqrt(dx * dx + dy * dy)   # L2 inverse distance
                    distance_norm = exp(-(dx * dx + dy * dy) / (radius_f * radius_f)) # Gaussian distance weight

                    s =
                        (
                            exp(-(diff_top * diff_top) / hh_top) *
                            exp(-(diff_bottom * diff_bottom) / hh_bottom)
                        )
                    w = s * distance_norm / total_weight
                    lin = Int((cc - 1) * nrows + rr)   # linear index 1..N as Int
                    # Use CUDA.@atomic macro to compile to the appropriate device atomic intrinsic
                    CUDA.@atomic y[lin] += w * xi
                end
            end
        end
    else
        # Fallback: no neighbors -> identity contribution to itself
        CUDA.@atomic y[Int(idx)] += x[Int(idx)]
    end

    return
end

"""
W_I_t_cross_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)

CUDA host wrapper that launches `Wt_cross_mul_gpu_kernel_cuda!` for top and bottom channels and
computes (W - I)ᵀ * x in-place by subtracting `x_gpu` from the atomic-accumulated result
stored in `y_gpu`.

Parameters
- y_gpu, x_gpu : device arrays of length 2N (top then bottom).
- p            : parameter tuple with N used to slice halves.
"""
function W_I_t_cross_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)
    N, _ = p

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    fill!(y_gpu, Float32(0.0))
    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @cuda threads = nthreads blocks = nblocks Wt_cross_mul_gpu_kernel_cuda!(
        top_y,
        top_x,
        nothing,
        p,
        nothing,
    )
    @cuda threads = nthreads blocks = nblocks Wt_cross_mul_gpu_kernel_cuda!(
        bottom_y,
        bottom_x,
        nothing,
        p,
        nothing,
    )
    y_gpu .= y_gpu .- x_gpu
end

"""
W_I_t_cross_mul_gpu_cuda!(x_gpu, u, p, t)

CUDA convenience variant that allocates and returns a new device array containing
(W - I)ᵀ * x. Matches the allocate-and-return calling convention used elsewhere.
"""
function W_I_t_cross_mul_gpu_cuda!(x_gpu, u, p, t)
    y_gpu = CUDA.zeros(eltype(x_gpu), length(x_gpu))

    N, _ = p

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @cuda threads = nthreads blocks = nblocks Wt_cross_mul_gpu_kernel_cuda!(
        top_y,
        top_x,
        nothing,
        p,
        nothing,
    )
    @cuda threads = nthreads blocks = nblocks Wt_cross_mul_gpu_kernel_cuda!(
        bottom_y,
        bottom_x,
        nothing,
        p,
        nothing,
    )
    y_gpu .= y_gpu .- x_gpu
end

# ==============================================================================
# Metal backend
# ------------------------------------------------------------------------------
# Metal-specific kernels and host wrappers for the similarity operator (W and Wᵀ).
# GPUs via the Metal.jl API.
# ==============================================================================

"""
W_cross_mul_gpu_kernel_metal!(y, x, p)

Metal device kernel implementing the forward gather-style similarity smoothing for a
single-channel flattened image. Semantics and `p` layout match the CUDA forward kernel.

Parameters
- y, x : Metal device arrays (flattened column-major).
- p    : parameter tuple unpacked as in the CUDA kernel (N, nrows, ncols, _, (img_top, img_bottom, h_top, h_bottom, radius)).

Notes
- Falls back to identity (y = x) when no valid neighbours contribute.
"""
function W_cross_mul_gpu_kernel_metal!(y, x, p)
    # Unpack parameters
    N::Int32,
    nrows::Int32,
    ncols::Int32,
    _,
    (img_top, img_bottom, h_top::Float32, h_bottom::Float32, radius::Int32) = p

    idx = thread_position_in_grid_1d()
    if idx < 1 || idx > N
        return
    end

    # map linear idx -> (r,c) for column-major storage
    c = Int32(cld(idx, nrows))
    r = Int32(idx - (c - 1) * nrows)
    radius_f = Float32(radius)

    center_val_top = img_top[r, c]
    center_val_bottom = img_bottom[r, c]

    r0 = max(Int32(1), r - radius)
    r1 = min(nrows, r + radius)
    c0 = max(Int32(1), c - radius)
    c1 = min(ncols, c + radius)

    total_weight = 0.0f0
    weighted_sum = 0.0f0
    hh_top = h_top * h_top
    hh_bottom = h_bottom * h_bottom

    @inbounds for cc = c0:c1
        for rr = r0:r1
            if rr == r && cc == c
                continue
            end
            neighbor_val_top = img_top[rr, cc]
            neighbor_val_bottom = img_bottom[rr, cc]
            diff_top = center_val_top - neighbor_val_top
            diff_bottom = center_val_bottom - neighbor_val_bottom
            if (abs(diff_top) < 3.0f0 * h_top) && (abs(diff_bottom) < 3.0f0 * h_bottom)
                dx = Float32(r - rr)
                dy = Float32(c - cc)
                #distance_norm = 1.0f0 / sqrt(dx * dx + dy * dy)   # L2 inverse distance
                distance_norm = exp(-(dx * dx + dy * dy) / (radius_f * radius_f)) # Gaussian distance weight

                s =
                    (
                        exp(-(diff_top * diff_top) / hh_top) *
                        exp(-(diff_bottom * diff_bottom) / hh_bottom)
                    )
                lin = (cc - 1) * nrows + rr
                weighted_sum += s * distance_norm * x[lin]
                total_weight += s * distance_norm
            end
        end
    end

    if total_weight > 0.0f0
        y[idx] = weighted_sum / total_weight
    else
        y[idx] = x[idx]
    end
    return
end


"""
W_I_cross_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)

Metal host wrapper that launches `W_cross_mul_gpu_kernel_metal!` for top and bottom channels and
then computes (W - I) * x in-place (y_gpu .= accumulated - x_gpu). Mirrors the CUDA host wrapper
but targets Metal device arrays and the Metal kernel API.
"""
function W_I_cross_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)
    N, _ = p

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @metal threads = nthreads groups = nblocks W_cross_mul_gpu_kernel_metal!(
        top_y,
        top_x,
        p,
    )
    @metal threads = nthreads groups = nblocks W_cross_mul_gpu_kernel_metal!(
        bottom_y,
        bottom_x,
        p,
    )
    y_gpu .= y_gpu .- x_gpu #(W - I) * x
end

"""
W_I_cross_mul_gpu_metal!(x_gpu, u, p, t)

Metal convenience variant that allocates and returns a new device array holding (W - I) * x.
"""
function W_I_cross_mul_gpu_metal!(x_gpu, u, p, t)
    N, _ = p
    y_gpu = Metal.zeros(eltype(x_gpu), length(x_gpu))

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @metal threads = nthreads groups = nblocks W_cross_mul_gpu_kernel_metal!(
        top_y,
        top_x,
        p,
    )

    @metal threads = nthreads groups = nblocks W_cross_mul_gpu_kernel_metal!(
        bottom_y,
        bottom_x,
        p,
    )

    y_gpu .= y_gpu .- x_gpu #(W - I) * x
end

"""
Wt_cross_mul_gpu_kernel_metal!(y, x, p)

Metal device kernel that computes the transpose (scatter) action of the similarity operator.
It mirrors the logic of `Wt_cross_mul_gpu_kernel_cuda!` but uses `Metal.@atomic` for neighbour writes.

Parameters
- y, x : Metal device arrays where `x` supplies centre contributions and `y` is updated via atomics.
- p    : same parameter tuple layout as the CUDA/Metal forward kernels.
"""
function Wt_cross_mul_gpu_kernel_metal!(y, x, p)
    # Unpack parameters
    N::Int32,
    nrows::Int32,
    ncols::Int32,
    nothing,
    (img_top, img_bottom, h_top::Float32, h_bottom::Float32, radius::Int32) = p

    idx = thread_position_in_grid_1d()
    if idx < 1 || idx > N
        return
    end

    # map linear idx -> (r,c) for column-major storage
    c = Int32(cld(idx, nrows))
    r = Int32(idx - (c - 1) * nrows)
    radius_f = Float32(radius)

    center_val_top = img_top[r, c]
    center_val_bottom = img_bottom[r, c]

    r0 = max(Int32(1), r - radius)
    r1 = min(nrows, r + radius)
    c0 = max(Int32(1), c - radius)
    c1 = min(ncols, c + radius)

    # First pass: compute sum of unnormalized similarities for this row (center)
    total_weight = 0.0f0
    hh_top = h_top * h_top
    hh_bottom = h_bottom * h_bottom

    @inbounds for cc = c0:c1
        for rr = r0:r1
            if rr == r && cc == c
                continue
            end
            neighbor_val_top = img_top[rr, cc]
            neighbor_val_bottom = img_bottom[rr, cc]
            diff_top = center_val_top - neighbor_val_top
            diff_bottom = center_val_bottom - neighbor_val_bottom
            if (abs(diff_top) < 3.0f0 * h_top) && (abs(diff_bottom) < 3.0f0 * h_bottom)
                dx = Float32(r - rr)
                dy = Float32(c - cc)
                #distance_norm = 1.0f0 / sqrt(dx * dx + dy * dy)   # L2 inverse distance
                distance_norm = exp(-(dx * dx + dy * dy) / (radius_f * radius_f)) # Gaussian distance weight

                s =
                    (
                        exp(-(diff_top * diff_top) / hh_top) *
                        exp(-(diff_bottom * diff_bottom) / hh_bottom)
                    )
                total_weight += s * distance_norm
            end
        end
    end

    # Second pass: normalized contributions into neighbors (atomic adds)
    if total_weight > 0.0f0
        xi = x[idx]  # contribution scalar from center
        @inbounds for cc = c0:c1
            for rr = r0:r1
                if rr == r && cc == c
                    continue
                end
                neighbor_val_top = img_top[rr, cc]
                neighbor_val_bottom = img_bottom[rr, cc]
                diff_top = center_val_top - neighbor_val_top
                diff_bottom = center_val_bottom - neighbor_val_bottom
                if (abs(diff_top) < 3.0f0 * h_top) && (abs(diff_bottom) < 3.0f0 * h_bottom)
                    dx = Float32(r - rr)
                    dy = Float32(c - cc)
                    #distance_norm = 1.0f0 / sqrt(dx * dx + dy * dy)   # L2 inverse distance
                    distance_norm = exp(-(dx * dx + dy * dy) / (radius_f * radius_f)) # Gaussian distance weight

                    s =
                        (
                            exp(-(diff_top * diff_top) / hh_top) *
                            exp(-(diff_bottom * diff_bottom) / hh_bottom)
                        )
                    w = s * distance_norm / total_weight
                    lin = Int((cc - 1) * nrows + rr)   # linear index 1..N as Int
                    # Use CUDA.@atomic macro to compile to the appropriate device atomic intrinsic
                    Metal.@atomic y[lin] += w * xi
                end
            end
        end
    else
        # Fallback: no neighbors -> identity contribution to itself
        Metal.@atomic y[Int(idx)] += x[Int(idx)]
    end

    return
end

"""
W_I_t_cross_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)

Metal host wrapper that launches the Metal transpose kernel for top and bottom channels
and computes (W - I)ᵀ * x in-place by subtracting `x_gpu` from the accumulated atomic result.
"""
function W_I_t_cross_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)
    N, _ = p

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    fill!(y_gpu, Float32(0.0))
    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @metal threads = nthreads groups = nblocks Wt_cross_mul_gpu_kernel_metal!(
        top_y,
        top_x,
        p,
    )
    @metal threads = nthreads groups = nblocks Wt_cross_mul_gpu_kernel_metal!(
        bottom_y,
        bottom_x,
        p,
    )
    y_gpu .= y_gpu .- x_gpu #(W - I)ᵀ * x
end

"""
W_I_t_cross_mul_gpu_metal!(x_gpu, u, p, t)

Metal convenience variant that allocates and returns a new device array containing
(W - I)ᵀ * x.
"""
function W_I_t_cross_mul_gpu_metal!(x_gpu, u, p, t)
    y_gpu = Metal.zeros(eltype(x_gpu), length(x_gpu))

    N, _ = p

    top_x = view(x_gpu, 1:N)
    bottom_x = view(x_gpu, N+1:2N)

    top_y = view(y_gpu, 1:N)
    bottom_y = view(y_gpu, N+1:2N)

    nthreads = 512
    nblocks = cld(N, nthreads)

    @metal threads = nthreads groups = nblocks Wt_cross_mul_gpu_kernel_metal!(
        top_y,
        top_x,
        p,
    )
    @metal threads = nthreads groups = nblocks Wt_cross_mul_gpu_kernel_metal!(
        bottom_y,
        bottom_x,
        p,
    )
    y_gpu .= y_gpu .- x_gpu #(W - I)ᵀ * x
end
