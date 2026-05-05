"""
        _noise_variance_diag_inv(dli_images, background_mask)

Compute a simple diagonal inverse-variance estimator from a rectangular
background region. The implementation follows the normalization strategy
from the referenced paper and returns a 2N-element vector containing the
inverse-variance for the top and bottom images concatenated.

Arguments
- `dli_images::DLI` : dual-energy images container with `.top` and `.bottom`.
- `background_mask` : an hyper-rectangle used to extract background pixels.

Returns
- `V_inv::Vector{Float64}` : length 2N vector containing inverse variance values
    (first N entries for the high-energy image, next N for the low-energy image).
"""
function _noise_variance_diag_inv(dli_images::DLI, background_mask)
    # low energy is top, high energy is bottom
    noise_bottom = extract_pixels(dli_images.bottom, background_mask)
    noise_top = extract_pixels(dli_images.top, background_mask)

    var_noise_bottom = var(noise_bottom)
    var_noise_top = var(noise_top)

    var_noise_bottom_norm = var_noise_bottom / (var_noise_bottom + var_noise_top)
    var_noise_top_norm = var_noise_top / (var_noise_bottom + var_noise_top)

    var_noise_bottom_vec = fill(1 / var_noise_bottom_norm, length(dli_images.bottom))
    var_noise_top_vec = fill(1 / var_noise_top_norm, length(dli_images.top))

    # top is the low energy spectra, bottom is the high energy spectra
    V_inv = vcat(var_noise_top_vec, var_noise_bottom_vec)
    return V_inv
end

"""
    Vinv_mat_cpu(dli_images, background_mask)

Construct a `MatrixOperator` representing the diagonal inverse-variance matrix
on the CPU. The diagonal is built from `_noise_variance_diag_inv` and returned
as a sparse diagonal matrix wrapped in `MatrixOperator`.
"""
function Vinv_mat_cpu(dli_images::DLI, background_mask)::MatrixOperator
    V_inv_diag = _noise_variance_diag_inv(dli_images, background_mask)
    V⁻¹ = spdiagm(V_inv_diag)
    return MatrixOperator(V⁻¹)
end

"""
    Vinv_mat_gpu_cuda(dli_images, background_mask)

Build a GPU-backed sparse diagonal inverse-variance matrix. Converts the CPU
diagonal to a `CuSparseMatrixCSC{Float32}` and wraps it in `MatrixOperator`.
"""
function Vinv_mat_gpu_cuda(dli_images::DLI, background_mask)::MatrixOperator
    V⁻¹_cpu = Vinv_mat_cpu(dli_images, background_mask).A
    V⁻¹_gpu = CuSparseMatrixCSC(Float32.(V⁻¹_cpu))
    return MatrixOperator(V⁻¹_gpu)
end

"""
    Vinv_mul!(w, v, u, p, t)

In-place multiplication implementing a simple diagonal inverse-variance
operator for use with `FunctionOperator`.

Arguments
- `w` : preallocated output vector (length 2N)
- `v` : input vector (length 2N)
- `u` : unused placeholder (kept for operator API compatibility)
- `p` : tuple `(N, var_high_inv, var_low_inv)` providing per-channel inverse variance scalars
- `t` : unused placeholder (kept for operator API compatibility)
"""
function Vinv_mul!(w, v, u, p, t)
    N, var_high_inv, var_low_inv = p
    for i = 1:N
        w[i] = var_high_inv * v[i]
        w[N+i] = var_low_inv * v[N+i]
    end
    nothing
end

"""
    Vinv_mul!(v, u, p, t)

Non-inplace convenience wrapper returning the result vector.
"""
function Vinv_mul!(v, u, p, t)
    w = zeros(eltype(v), length(v))
    Vinv_mul!(w, v, u, p, t)
    w
end

"""
    Vinv_op_cpu(dli_images, background_mask)

Build a `FunctionOperator` implementing the diagonal inverse-variance operator
on the CPU. The operator uses background statistics to estimate per-channel
inverse-variance scalars and returns a `FunctionOperator` configured for in-place
linear application.
"""
function Vinv_op_cpu(dli_images::DLI, background_mask)::FunctionOperator
    # low energy is top, high energy is bottom
    noise_bottom = extract_pixels(dli_images.bottom, background_mask)
    noise_top = extract_pixels(dli_images.top, background_mask)
    N = length(dli_images.top)

    var_noise_bottom = var(noise_bottom)
    var_noise_top = var(noise_top)

    var_noise_bottom_norm = var_noise_bottom / (var_noise_bottom + var_noise_top)
    var_noise_top_norm = var_noise_top / (var_noise_bottom + var_noise_top)

    p_V = (N, 1 / var_noise_top_norm, 1 / var_noise_bottom_norm)
    op_V = FunctionOperator(
        Vinv_mul!,
        zeros(Float64, 2N),
        zeros(Float64, 2N);
        p = p_V,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_V
end


# Matrix Free Variance-covariance on Metal GPU

"""
    Vinv_mul_gpu_kernel_metal!(y_gpu, x_gpu, p)

Metal kernel body applying per-channel scalar inverse-variance multiplication.
Each thread multiplies two entries corresponding to the high/low channels.
"""
function Vinv_mul_gpu_kernel_metal!(y_gpu, x_gpu, p)
    N, var_high_inv, var_low_inv = p
    idx = thread_position_in_grid_1d()
    y_gpu[idx] = var_high_inv * x_gpu[idx]
    y_gpu[N+idx] = var_low_inv * x_gpu[N+idx]
    nothing
end

"""
    Vinv_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)

Launch the Metal kernel to compute `y_gpu = V⁻¹ * x_gpu` in-place.
`p` is expected to be `(N, var_high_inv, var_low_inv)`.
"""
function Vinv_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)
    N, _ = p
    nthreads = 512
    nblocks = cld(N, nthreads)
    @metal threads = nthreads groups = nblocks Vinv_mul_gpu_kernel_metal!(y_gpu, x_gpu, p)
end

"""
    Vinv_mul_gpu_metal!(x_gpu, u, p, t)

Non-inplace variant for Metal: allocates `y_gpu`, launches the kernel and returns it.
"""
function Vinv_mul_gpu_metal!(x_gpu, u, p, t)
    y_gpu = Metal.zeros(eltype(x_gpu), length(x_gpu))
    Vinv_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)
    y_gpu
end

"""
    Vinv_op_gpu_metal(dli_images, background_mask)

Build a Metal-backed `FunctionOperator` for the inverse-variance operator.
Estimates per-channel inverse variances from the background mask and configures
the operator with a GPU prototype vector.
"""
function Vinv_op_gpu_metal(dli_images::DLI, background_mask)::FunctionOperator
    # low energy is top, high energy is bottom
    noise_bottom = extract_pixels(dli_images.bottom, background_mask)
    noise_top = extract_pixels(dli_images.top, background_mask)

    var_noise_bottom = var(noise_bottom)
    var_noise_top = var(noise_top)

    var_noise_bottom_norm = var_noise_bottom / (var_noise_bottom + var_noise_top)
    var_noise_top_norm = var_noise_top / (var_noise_bottom + var_noise_top)

    N = Int32(length(dli_images.top))
    prototype_gpu = MtlVector(zeros(Float32, 2N))

    p_V = (N, Float32(1 / var_noise_top_norm), Float32(1 / var_noise_bottom_norm))
    op_V_gpu_metal = FunctionOperator(
        Vinv_mul_gpu_metal!,
        prototype_gpu,
        prototype_gpu;
        p = p_V,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_V_gpu_metal
end

"""
    Vinv_mul_gpu_kernel_cuda!(y_gpu, x_gpu, p)

CUDA kernel body applying per-channel inverse variance to two vector entries per thread.
"""
function Vinv_mul_gpu_kernel_cuda!(y_gpu, x_gpu, p)
    N, var_high_inv, var_low_inv = p
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx < 1 || idx > N
        return
    end
    y_gpu[idx] = var_high_inv * x_gpu[idx]
    y_gpu[N+idx] = var_low_inv * x_gpu[N+idx]
    nothing
end

"""
    Vinv_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)

Launch the CUDA kernel to compute `y_gpu = V⁻¹ * x_gpu` in-place.
"""
function Vinv_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)
    N, _ = p
    nthreads = 512
    nblocks = cld(N, nthreads)
    @cuda threads = nthreads blocks = nblocks Vinv_mul_gpu_kernel_cuda!(y_gpu, x_gpu, p)
end

"""
    Vinv_mul_gpu_cuda!(x_gpu, u, p, t)

Non-inplace CUDA variant: allocates `y_gpu`, launches kernel and returns it.
"""
function Vinv_mul_gpu_cuda!(x_gpu, u, p, t)
    y_gpu = CUDA.zeros(eltype(x_gpu), length(x_gpu))
    Vinv_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)
    y_gpu
end

"""
    Vinv_op_gpu_cuda(dli_images, background_mask)

Build a CUDA-backed `FunctionOperator` for the inverse-variance operator.
"""
function Vinv_op_gpu_cuda(dli_images::DLI, background_mask)::FunctionOperator
    noise_bottom = extract_pixels(dli_images.bottom, background_mask)
    noise_top = extract_pixels(dli_images.top, background_mask)

    var_noise_bottom = var(noise_bottom)
    var_noise_top = var(noise_top)

    var_noise_bottom_norm = var_noise_bottom / (var_noise_bottom + var_noise_top)
    var_noise_top_norm = var_noise_top / (var_noise_bottom + var_noise_top)

    N = Int32(length(dli_images.top))
    prototype_gpu = CuArray(zeros(Float32, 2N))

    p_V = (N, Float32(1 / var_noise_top_norm), Float32(1 / var_noise_bottom_norm))
    op_V_gpu_cuda = FunctionOperator(
        Vinv_mul_gpu_cuda!,
        prototype_gpu,
        prototype_gpu;
        p = p_V,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_V_gpu_cuda
end
