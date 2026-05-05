# Constructor of quadratic regularization gradient, both matrix and lazy operator version
# All constructor must have the same signature constructor(parameters, prototype)

"""
    ∇R_quad_op(params, prototype)

Return a `FunctionOperator` implementing the gradient of a quadratic regularizer.

Arguments
- `params` : tuple holding regularizer parameters (see specific regularizer utilities)
- `prototype::AbstractArray` : prototype vector used to size the operator

Parameters tuple expected (for the matrix-free / operator version):
`params = (N, nrows, ncols, linear_ind)` where
 - `N` : number of pixels per image (nrows * ncols)
 - `nrows`, `ncols` : image dimensions
 - `linear_ind` : `LinearIndices((nrows,ncols))` used to map CartesianIndices to linear indices

Returns
- `FunctionOperator` ready to be used as `∇R` in the linear system.
"""
function ∇R_quad_op(params, prototype::AbstractArray)::FunctionOperator
    op_∇R = FunctionOperator(
        ∇R_quadratic_mul!,
        prototype,
        prototype;
        p = params,
        isconstant = true,
        isinplace = true,
    )
    return op_∇R
end

"""
    ∇R_quad_mat(params, prototype)

Return the sparse matrix representation of the quadratic regularizer when
an explicit matrix is available.

Parameters tuple expected for the matrix form:
`params = (N, nrows, ncols, linear_ind, quadratic_regularization_matrix)`
The 5th element is the sparse quadratic regularization matrix (2N×2N).
"""
function ∇R_quad_mat(params, prototype::AbstractArray)::MatrixOperator
    _, _, _, _, quadratic_regularization_matrix = params
    return MatrixOperator(quadratic_regularization_matrix)
end

# Constructor of edge-weighted quadratic regularization gradient, both matrix and lazy operator version

"""
    ∇R_quad_ew_op(params, prototype)

Return a `FunctionOperator` for an edge-weighted quadratic regularizer. The
operator typically uses edge-maps to weight smoothing and preserve edges.

Parameters tuple expected:
`params = (N, nrows, ncols, linear_ind, (combined_edges, edge_val, non_edge_val))` where
 - `combined_edges` : a BitMatrix or boolean mask (nrows×ncols) indicating edge pixels
 - `edge_val` : numeric weight used at/near edges (typically small, < 1)
 - `non_edge_val` : numeric weight used on non-edge pixels (typically 1.0)
"""
function ∇R_quad_ew_op(params, prototype::AbstractArray)::FunctionOperator
    op_∇R = FunctionOperator(
        ∇R_quad_ew_mul!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    return op_∇R
end

"""
    ∇R_quad_ew_mat(params, prototype)

Return the sparse matrix representation of the edge-weighted quadratic regularizer
extracted from `params`.

Parameters tuple expected for the matrix form:
`params = (N, nrows, ncols, linear_ind, ew_quadratic_regularization_matrix)`
where the 5th element is the precomputed 2N×2N sparse matrix.
"""
function ∇R_quad_ew_mat(params, prototype::AbstractArray)::MatrixOperator
    _, _, _, _, ew_quadratic_regularization_matrix = params
    return MatrixOperator(ew_quadratic_regularization_matrix)
end

# Constructor of similarity-based regularization gradient, both matrix and lazy operator version

"""
    ∇R_similarity_op(params, prototype)

Build a composed `∇R` operator for similarity-based regularization. The
operator implements (Wᵀ - I)(W - I) where `W` is a data-dependent similarity
operator. Returns a cached composed operator.

Parameters tuple expected (operator / matrix-free path):
`params = (N, nrows, ncols, linear_ind, (dli_images, h_top, h_bottom, half_size))` where
 - `dli_images` : container with `.top` and `.bottom` images (used to compute similarities)
 - `h_top`, `h_bottom` : similarity bandwidth parameters for top and bottom images
 - `half_size` : search half-window size (integer)

The `FunctionOperator` `W_mul!` and `Wt_mul!` implement the gather/scatter
behavior of a similarity matrix without ever assembling it explicitly.
"""
function ∇R_similarity_op(params, prototype::AbstractArray)::SciMLOperators.ComposedOperator
    W = FunctionOperator(
        W_mul!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    W_tranposed = FunctionOperator(
        Wt_mul!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    Id = IdentityOperator(length(prototype))
    op_∇R = cache_operator((W_tranposed - Id) * (W - Id), prototype)
    return op_∇R
end

#p = (N_gpu, nrows, ncols, nothing, (img_top_gpu, img_bottom_gpu, h_top_gpu, h_bottom_gpu, radius_gpu))
"""
    ∇R_cross_similarity_op_gpu_metal(params, prototype)

GPU (Metal) variant of `∇R_similarity_op`.

Input/expected parameters:
 - Caller may pass `params = (N, nrows, ncols, nothing, (img_top, img_bottom, h_top, h_bottom, radius))` where
     `img_top`/`img_bottom` are CPU arrays. The constructor converts these to `MtlArray` and
     rewrites `params` to `(Int32(N), Int32(nrows), Int32(ncols), nothing, (MtlArray(img_top_gpu), ... , Int32(radius)))`.

After conversion the GPU operators expect `params = (N32, nrows32, ncols32, nothing, (img_top_mtl, img_bottom_mtl, h_top_f32, h_bottom_f32, radius_i32))`.
"""
function ∇R_cross_similarity_op_gpu_metal(
    params,
    prototype::MtlVector,
)::SciMLOperators.ComposedOperator
    N, nrows, ncols, _, (img_top, img_bottom, h_top, h_bottom, radius) = params
    #Convert to appropriate types for Metal
    params = (
        Int32(N),
        Int32(nrows),
        Int32(ncols),
        nothing,
        (
            MtlArray(Float32.(img_top)),
            MtlArray(Float32.(img_bottom)),
            Float32(h_top),
            Float32(h_bottom),
            Int32(radius),
        ),
    )
    op_sim_gpu = FunctionOperator(
        W_I_cross_mul_gpu_metal!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    op_sim_transposed_gpu = FunctionOperator(
        W_I_t_cross_mul_gpu_metal!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    op_∇R_gpu = cache_operator((op_sim_transposed_gpu) * (op_sim_gpu), prototype)
    return op_∇R_gpu
end

"""
    ∇R_cross_similarity_op_gpu_cuda(params, prototype)

GPU (CUDA) variant of `∇R_similarity_op`.

Input/expected parameters:
 - Caller may pass `params = (N, nrows, ncols, nothing, (img_top, img_bottom, h_top, h_bottom, radius))` where
     `img_top`/`img_bottom` are CPU arrays. The constructor converts these to `CuArray` and
     rewrites `params` to `(Int32(N), Int32(nrows), Int32(ncols), nothing, (CuArray(img_top), ... , Int32(radius)))`.

After conversion the GPU operators expect `params = (N32, nrows32, ncols32, nothing, (img_top_cu, img_bottom_cu, h_top_f32, h_bottom_f32, radius_i32))`.
"""
function ∇R_cross_similarity_op_gpu_cuda(
    params,
    prototype::CuVector,
)::SciMLOperators.ComposedOperator
    N, nrows, ncols, _, (img_top, img_bottom, h_top, h_bottom, radius) = params
    #Convert to appropriate types for CUDA
    params = (
        Int32(N),
        Int32(nrows),
        Int32(ncols),
        nothing,
        (
            CuArray(Float32.(img_top)),
            CuArray(Float32.(img_bottom)),
            Float32(h_top),
            Float32(h_bottom),
            Int32(radius),
        ),
    )
    op_sim_gpu = FunctionOperator(
        W_I_cross_mul_gpu_cuda!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    op_sim_transposed_gpu = FunctionOperator(
        W_I_t_cross_mul_gpu_cuda!,
        prototype,
        prototype;
        p = params,
        islinear = true,
        isconstant = true,
        isinplace = true,
    )
    op_∇R_gpu = cache_operator((op_sim_transposed_gpu) * (op_sim_gpu), prototype)
    return op_∇R_gpu
end


"""
    ∇R_similarity_mat(params, prototype)

Return a matrix-based composed operator for similarity regularization. Expects
`sim_mat` inside `params` and computes (W - I)ᵀ(W - I) explicitly using sparse
matrices wrapped in `MatrixOperator`. Can be both "similarity" and "cross-similarity", depending on the input matrix.

Parameters tuple expected:
`params = (N, nrows, ncols, linear_ind, (sim_mat,))` where `sim_mat` is the
2N×2N block-diagonal similarity matrix (usually `blockdiag(sim_top, sim_bottom)`).
"""
function ∇R_similarity_mat(
    params,
    prototype::AbstractArray,
)::SciMLOperators.ComposedOperator
    N, _, _, _, (sim_mat,) = params

    Id_N = sparse(I, 2N, 2N)
    W_I = sim_mat - Id_N
    W_I_tranpose = transpose(W_I)

    op_W_I = MatrixOperator(W_I)
    op_W_I_tranpose = MatrixOperator(W_I_tranpose)

    # The gradient operator matrix is (W-I)ᵀ(W-I).
    op_∇R = op_W_I_tranpose * op_W_I
    return op_∇R
end

"""
    ∇R_similarity_mat_gpu_cuda(params, prototype)

CUDA GPU matrix-based variant: converts a sparse similarity matrix to a
`CuSparseMatrixCSC` and returns the composed operator (W-I)ᵀ(W-I) using GPU
matrix operators.  Can be both "similarity" and "cross-similarity", depending on the input matrix.

Parameters tuple expected:
`params = (N, nrows, ncols, linear_ind, (sim_mat,))` where `sim_mat` is a 2N×2N sparse
similarity matrix on CPU; this function converts it to `CuSparseMatrixCSC` internally.
"""
function ∇R_similarity_mat_gpu_cuda(
    params,
    prototype::AbstractArray,
)::SciMLOperators.ComposedOperator
    N, _, _, _, (sim_mat,) = params

    Id_N = sparse(I, 2N, 2N)
    W_I = sim_mat - Id_N
    W_I_tranpose = transpose(W_I)

    op_W_I = MatrixOperator(CuSparseMatrixCSC(Float32.(W_I)))
    op_W_I_tranpose = MatrixOperator(CuSparseMatrixCSC(Float32.(W_I_tranpose)))

    # The gradient operator matrix is (W-I)ᵀ(W-I).
    op_∇R = op_W_I_tranpose * op_W_I
    return op_∇R
end

#= # Constructor of regularization functions (not used, but could be useful for testing)'

# Contructor of q-GGMRF regularization gradient
function ∇R_qGGMRF(params, prototype::AbstractArray)
     op_∇R  = FunctionOperator(∇R_qGGMRF_mul!, prototype, prototype;  p = params, isconstant=true, isinplace=true)
    return op_∇R
end

 function R_qGGMRF(params, prototype::AbstractArray)
     op_∇R  = FunctionOperator(R_qGGMRF_mul!, prototype, prototype;  p = params, isconstant=true, isinplace=true)
    return op_∇R
end

function R_quad(params, prototype::AbstractArray)
     op_∇R  = FunctionOperator(R_quadratic_mul!, prototype, prototype;  p = params, isconstant=true, isinplace=true)
    return op_∇R
end

function R_quad_ew(params, prototype::AbstractArray)
    op_∇R = FunctionOperator(R_quad_ew_mul!, prototype, prototype;  p = params, islinear = true, isconstant=true, isinplace=true)    
    return op_∇R
end =#
