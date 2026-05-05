#The file is use to build the A matrix in CPU (as a SparseMatrix or a FunctionOperator) or GPU (as a FunctionOperator)

"""
    A_mat_cpu(N, M)

Construct a sparse block mixing matrix for the material decomposition on CPU.
This builds a 2N × 2N sparse block matrix composed of scaled identity blocks:

    [M[1,1] I   M[1,2] I
     M[2,1] I   M[2,2] I]

Arguments
- `N::Integer` : number of pixels per material (nrows * ncols)
- `M::Matrix{Float64}` : 2×2 mixing matrix mapping materials to measurements

Returns
- `MatrixOperator` wrapping the constructed sparse matrix.
"""
function A_mat_cpu(N::Integer, M::Matrix{Float64})::MatrixOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2"
    # Create an N x N sparse identity matrix to use as a template
    Id_N = sparse(I, N, N)

    # Construct the four N x N blocks by scaling the identity matrix
    A11 = M[1, 1] * Id_N
    A12 = M[1, 2] * Id_N
    A21 = M[2, 1] * Id_N
    A22 = M[2, 2] * Id_N

    # Assemble the final 2N x 2N sparse matrix from the blocks
    A = [A11 A12; A21 A22]

    return MatrixOperator(A)
end

"""
    At_mat_cpu(N, M)

Construct the transpose (adjoint) of the block mixing matrix on CPU.

Arguments
- `N::Integer` : number of pixels per material
- `M::Matrix{Float64}` : 2×2 mixing matrix; the function uses `transpose(M)` internally.

Returns
- `MatrixOperator` wrapping the constructed sparse adjoint matrix.
"""
function At_mat_cpu(N::Integer, M::Matrix{Float64})::MatrixOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2"
    # Create an N x N sparse identity matrix to use as a template
    Mᵀ = transpose(M)
    Id_N = sparse(I, N, N)

    # Construct the four N x N blocks by scaling the identity matrix
    A11 = Mᵀ[1, 1] * Id_N
    A12 = Mᵀ[1, 2] * Id_N
    A21 = Mᵀ[2, 1] * Id_N
    A22 = Mᵀ[2, 2] * Id_N

    # Assemble the final 2N x 2N sparse matrix from the blocks
    Aᵀ = [A11 A12; A21 A22]

    return MatrixOperator(Aᵀ)
end

"""
    A_mat_gpu_cuda(N, M)

Build a GPU-backed sparse matrix representation of the block mixing matrix.

This function converts the small 2×2 `M` to CPU floats, constructs the sparse
block matrix via `A_mat_cpu`, then converts to a `CuSparseMatrixCSC`.

Arguments
- `N` : number of pixels per material (accepts integer or Int32)
- `M::CuMatrix{Float32}` : 2×2 mixing matrix on the GPU

Returns
- `MatrixOperator` wrapping a `CuSparseMatrixCSC{Float32}`.
"""
function A_mat_gpu_cuda(N, M::CuMatrix{Float32})::MatrixOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2 (CuMatrix)"
    M_cpu = Matrix{Float64}(M)
    A_cpu = A_mat_cpu(N, M_cpu).A
    A_gpu = CuSparseMatrixCSC(Float32.(A_cpu))
    return MatrixOperator(A_gpu)
end

"""
    At_mat_gpu_cuda(N, M)

GPU-backed adjoint sparse matrix for the mixing operator.

This mirrors `A_mat_gpu_cuda` but builds the transpose adjoint on the GPU.
"""
function At_mat_gpu_cuda(N, M::CuMatrix{Float32})::MatrixOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2 (CuMatrix)"
    M_cpu = Matrix{Float64}(M)
    Aᵀ_cpu = At_mat_cpu(N, M_cpu).A
    Aᵀ_gpu = CuSparseMatrixCSC(Float32.(Aᵀ_cpu))
    return MatrixOperator(Aᵀ_gpu)
end

"""
    A_mul!(w, v, u, p, t)

In-place multiplication for the block mixing operator used by `FunctionOperator`.

Arguments
- `w` : output vector (preallocated), length 2N
- `v` : input vector (length 2N)
- `u` : unused placeholder (kept for operator API compatibility)
- `p` : tuple `(N, M)` where `N` is integer and `M` is 2×2 mixing matrix
- `t` : unused placeholder (kept for operator API compatibility)

Behavior
- Writes the product `w = A * v` in-place using scaled identity blocks.
"""
function A_mul!(w, v, u, p, t)
    N, M = p
    @assert length(w) == length(v) "Output and input must have same length"
    @assert length(v) == 2 * N "Input vector length must be 2N"
    @assert size(M) == (2, 2) "Mixing matrix M must be 2×2 in p"
    for i = 1:N
        w[i] = M[1, 1] * v[i] + M[1, 2] * v[N+i]
        w[N+i] = M[2, 1] * v[i] + M[2, 2] * v[N+i]
    end
    nothing
end

"""
    A_mul!(v, u, p, t)

Convenience non-inplace variant returning the result vector. Uses the in-place
`A_mul!` internally and allocates the output.
"""
function A_mul!(v, u, p, t)
    w = zeros(eltype(v), length(v))
    A_mul!(w, v, u, p, t)
    w
end

"""
    A_op_cpu(N, M)

Return a `FunctionOperator` wrapping the block-mixing multiplication for CPU use.

Arguments
- `N::Int` : number of pixels per material
- `M::Matrix{Float64}` : 2×2 mixing matrix

Returns
- `FunctionOperator` configured for in-place linear application.
"""
function A_op_cpu(N::Int, M::Matrix{Float64})::FunctionOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "M must be 2×2"
    op_A = FunctionOperator(
        A_mul!,
        zeros(Float64, 2N),
        zeros(Float64, 2N);
        p = (N, M),
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_A
end

"""
    At_op_cpu(N, M)

Return a `FunctionOperator` wrapping the adjoint (transpose) block-mixing operator
for CPU use.
"""
function At_op_cpu(N::Int, M::Matrix{Float64})::FunctionOperator
    @assert N > 0 "N must be positive"
    @assert size(M) == (2, 2) "M must be 2×2"
    op_At = FunctionOperator(
        A_mul!,
        zeros(Float64, 2N),
        zeros(Float64, 2N);
        p = (N, transpose(M)),
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_At
end


"""
    A_mul_gpu_kernel_metal!(y_gpu, x_gpu, p)

Metal kernel body: each thread computes two output entries for the block operator.
`thread_position_in_grid_1d()` is used to index into the vectors.
"""
function A_mul_gpu_kernel_metal!(y_gpu, x_gpu, p)
    N, M = p
    idx = thread_position_in_grid_1d()
    y_gpu[idx] = M[1, 1] * x_gpu[idx] + M[1, 2] * x_gpu[N+idx]
    y_gpu[N+idx] = M[2, 1] * x_gpu[idx] + M[2, 2] * x_gpu[N+idx]
    nothing
end

"""
    A_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)

Launches the Metal kernel to compute `y_gpu = A * x_gpu` in-place.
`p` is expected to be `(N, M)` where `M` is a 2×2 mixing matrix (Metal-backed).
"""
function A_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)
    N, _ = p
    nthreads = 512
    nblocks = cld(N, nthreads)
    @metal threads = nthreads groups = nblocks A_mul_gpu_kernel_metal!(y_gpu, x_gpu, p)
end

"""
    A_mul_gpu_metal!(x_gpu, u, p, t)

Non-inplace variant for Metal: allocates `y_gpu`, launches kernel and returns it.
"""
function A_mul_gpu_metal!(x_gpu, u, p, t)
    y_gpu = Metal.zeros(eltype(x_gpu), length(x_gpu))
    A_mul_gpu_metal!(y_gpu, x_gpu, u, p, t)
    y_gpu
end

"""
    A_op_gpu_metal(N, M)

Return a `FunctionOperator` wrapping the Metal-backed block-mixing operator.
"""
function A_op_gpu_metal(N::Int32, M::MtlMatrix{Float32})::FunctionOperator
    op_A_gpu_metal = FunctionOperator(
        A_mul_gpu_metal!,
        zeros(Float32, 2N),
        zeros(Float32, 2N);
        p = (N, M),
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_A_gpu_metal
end

"""
    At_op_gpu_metal(N, M)

Return a `FunctionOperator` wrapping the Metal-backed adjoint block-mixing operator.
"""
function At_op_gpu_metal(N::Int32, M::MtlMatrix{Float32})::FunctionOperator
    op_At_gpu_metal = FunctionOperator(
        A_mul_gpu_metal!,
        zeros(Float32, 2N),
        zeros(Float32, 2N);
        p = (N, transpose(M)),
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_At_gpu_metal
end


"""
    A_mul_gpu_kernel_cuda!(y_gpu, x_gpu, p)

CUDA kernel body computing the block-mixing product for a single thread index.
The kernel guards for out-of-range thread indices.
"""
function A_mul_gpu_kernel_cuda!(y_gpu, x_gpu, p)
    N, M = p
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx < 1 || idx > N
        return
    end
    y_gpu[idx] = M[1, 1] * x_gpu[idx] + M[1, 2] * x_gpu[N+idx]
    y_gpu[N+idx] = M[2, 1] * x_gpu[idx] + M[2, 2] * x_gpu[N+idx]
    nothing
end

"""
    A_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)

Launches the CUDA kernel to compute `y_gpu = A * x_gpu` in-place.
`p` should be `(N, M)` where `M` is a small 2×2 CuMatrix.
"""
function A_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)
    N, _ = p
    nthreads = 512
    nblocks = cld(N, nthreads)
    @cuda threads = nthreads blocks = nblocks A_mul_gpu_kernel_cuda!(y_gpu, x_gpu, p)
end

"""
    A_mul_gpu_cuda!(x_gpu, u, p, t)

Non-inplace CUDA variant: allocates `y_gpu`, launches the kernel and returns it.
"""
function A_mul_gpu_cuda!(x_gpu, u, p, t)
    y_gpu = CUDA.zeros(eltype(x_gpu), length(x_gpu))
    A_mul_gpu_cuda!(y_gpu, x_gpu, u, p, t)
    y_gpu
end

"""
    A_op_gpu_cuda(N, M)

Return a `FunctionOperator` wrapping the CUDA-backed block-mixing operator.
"""
function A_op_gpu_cuda(N::Int32, M::CuMatrix{Float32})::FunctionOperator
    op_A_gpu_cuda = FunctionOperator(
        A_mul_gpu_cuda!,
        zeros(Float32, 2N),
        zeros(Float32, 2N);
        p = (N, M),
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_A_gpu_cuda
end

"""
    At_op_gpu_cuda(N, M)

Return a `FunctionOperator` wrapping the CUDA-backed adjoint block-mixing operator.
"""
function At_op_gpu_cuda(N::Int32, M::CuMatrix{Float32})::FunctionOperator
    op_At_gpu_cuda = FunctionOperator(
        A_mul_gpu_cuda!,
        zeros(Float32, 2N),
        zeros(Float32, 2N);
        p = (N, transpose(M)),
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_At_gpu_cuda
end
