# This file provides functions to create explicit sparse matrices for various image regularization terms. This
# contrasts with the FunctionOperator approach, where the operation is defined
# only by its action on a vector (lazy operations).

"""
Generates the sparse matrix '∇R' for a quadratic regularization term.
If `edge_map` is provided, it generates the edge-weighted matrix.
Otherwise, it generates the standard unweighted matrix.
"""
function generate_quadratic_regularization_matrix(
    img
)
    nrows, ncols = size(img)
    N = nrows * ncols

    return spdiagm(
        N,
        N,
        -nrows => fill(-1.0, N - nrows),
        nrows => fill(-1.0, N - nrows),
        0 => fill(4.0, N),
        -1 => fill(-1.0, N - 1),
        1 => fill(-1.0, N - 1),
    )
end

"""
Generates the sparse matrix '∇R' for a quadratic regularization term.
If `edge_map` is provided, it generates the edge-weighted matrix.
Otherwise, it generates the standard unweighted matrix.
"""
function generate_ew_quadratic_regularization_matrix(
    img;
    combined_edges,
    edge_val = 0.2,
    non_edge_val = 1.0,
)
    nrows, ncols = size(img)
    N = nrows * ncols
    linear_ind = LinearIndices((nrows, ncols))
    cartesian_ind = CartesianIndices((nrows, ncols))

    rows = Vector{Int64}(undef, 5N)
    cols = Vector{Int64}(undef, 5N)
    vals = Vector{Float64}(undef, 5N)

    for (i, center_ind_cart) in enumerate(cartesian_ind)
        center_ind_linear = linear_ind[center_ind_cart]
        neighbor_indices, weights = _calculate_quad_ew_row(
            linear_ind,
            center_ind_cart,
            nrows,
            ncols,
            combined_edges,
            edge_val,
            non_edge_val,
        )

        rows[5*(i-1)+1:5*i] .= fill(center_ind_linear, 5)
        cols[5*(i-1)+1:5*i] .= neighbor_indices
        vals[5*(i-1)+1:5*i] .= weights
    end

    return SparseArrays.sparse(rows, cols, vals, N, N)
end


# ##############################################################################
# 2. SIMILARITY-BASED REGULARIZER
# ##############################################################################
# This regularizer is R(x) = ||(W-I)x||². This is a quadratic form, and its
# gradient is the linear operator ∇R(x) = 2(W-I)ᵀ(W-I)x. We can explicitly
# build the matrix M = 2(W-I)ᵀ(W-I).

function generate_similarity_matrix_W(
    img, 
    h, 
    half_size::Int;
    distance_metric::Function = no_distance_penalty
)
    nrows, ncols = size(img)
    linear_ind = LinearIndices((nrows, ncols))
    cartesian_ind = CartesianIndices((nrows, ncols))
    N = nrows * ncols
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for center_ind_cart in cartesian_ind
        center_ind_linear = linear_ind[center_ind_cart]
        i, j = center_ind_cart[1], center_ind_cart[2]

        ni_x = max(1, i - half_size):min(nrows, i + half_size)
        ni_y = max(1, j - half_size):min(ncols, j + half_size)

        neighbor_indices, weights = _calculate_similarity_row(
            img,
            linear_ind,
            cartesian_ind,
            center_ind_cart,
            h,
            ni_x,
            ni_y,
            nrows,
            ncols,
            half_size,
            distance_metric = distance_metric
        )

        append!(vals, weights)
        append!(cols, neighbor_indices)
        append!(rows, fill(center_ind_linear, length(neighbor_indices)))
    end
    return SparseArrays.sparse(rows, cols, vals, N, N)
end

#@time W = generate_similarity_matrix_W(vec(dli_images.top), nrows, ncols, linear_ind, 0.5, 20)

function generate_cross_similarity_matrix_W(
    dli_images::DLI,
    h_top,
    h_bottom,
    half_size::Int;
    distance_metric::Function = gaussian_spatial_distance
)
    nrows, ncols = size(dli_images.top)
    linear_ind = LinearIndices((nrows, ncols))
    cartesian_ind = CartesianIndices((nrows, ncols))
    N = nrows * ncols
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for center_ind_cart in cartesian_ind
        center_ind_linear = linear_ind[center_ind_cart]
        i, j = center_ind_cart[1], center_ind_cart[2]

        ni_x = max(1, i - half_size):min(nrows, i + half_size)
        ni_y = max(1, j - half_size):min(ncols, j + half_size)

        # in the following function, use the cartesian index "ind" and not the linear "center_ind"
        neighbor_indices, weights = _calculate_cross_similarity_row(
            dli_images,
            linear_ind,
            cartesian_ind,
            center_ind_cart,
            h_top,
            h_bottom,
            ni_x,
            ni_y,
            nrows,
            ncols,
            half_size,
            distance_metric = distance_metric
        )

        append!(vals, weights)
        append!(cols, neighbor_indices)
        append!(rows, fill(center_ind_linear, length(neighbor_indices)))
    end
    return SparseArrays.sparse(rows, cols, vals, N, N)
end
