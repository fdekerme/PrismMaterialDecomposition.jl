function R_quadratic_mul!(R, v, u, p, t)
    N, nrows, ncols, linear_ind = p
    top_image = view(v, 1:N)
    bottom_image = view(v, N+1:2N)

    R[] = zero(eltype(R[])) # Initialize accumulator

    for ind in CartesianIndices((nrows, ncols)) #dont use Threads here because thread race
        #v is a Vector, but it is much easier to deals with 2D image. It is however slow to reshape v.
        # We use linear_ind to convert 2D CartesianIndex to Vector linear indices. 
        l_ind_center = linear_ind[ind]
        l_ind_left = linear_ind[left(ind, nrows)]
        l_ind_right = linear_ind[right(ind, nrows)]
        l_ind_top = linear_ind[top(ind, ncols)]
        l_ind_bottom = linear_ind[bottom(ind, ncols)]

        R[] += quad(top_image[l_ind_center] - top_image[l_ind_left])
        R[] += quad(top_image[l_ind_center] - top_image[l_ind_right])
        R[] += quad(top_image[l_ind_center] - top_image[l_ind_top])
        R[] += quad(top_image[l_ind_center] - top_image[l_ind_bottom])

        R[] += quad(bottom_image[l_ind_center] - bottom_image[l_ind_left])
        R[] += quad(bottom_image[l_ind_center] - bottom_image[l_ind_right])
        R[] += quad(bottom_image[l_ind_center] - bottom_image[l_ind_top])
        R[] += quad(bottom_image[l_ind_center] - bottom_image[l_ind_bottom])
    end

end

function R_quadratic_mul!(v, u, p, t)
    R = Ref(zero(eltype(v))) #R in not a FLoat in the LinearSovle "solve" function
    R_quadratic_mul!(R, v, u, p, t)
    R[]
end

function ∇R_quadratic_mul!(R, v, u, p_prime, t)
    N, nrows, ncols, linear_ind = p_prime
    top_image = view(v, 1:N)
    bottom_image = view(v, N+1:2N)
    
    fill!(R, zero(eltype(R))) # Initialize accumulator

    for ind in CartesianIndices((nrows, ncols)) #dont use Threads here because thread race
        #v is a Vector, but it is much easier to deals with 2D image. It is however slow to reshape v.
        # We use linear_ind to convert 2D CartesianIndex to Vector linear indices. 
        l_ind_center = linear_ind[ind]
        l_ind_left = linear_ind[left(ind, nrows)]
        l_ind_right = linear_ind[right(ind, nrows)]
        l_ind_top = linear_ind[top(ind, ncols)]
        l_ind_bottom = linear_ind[bottom(ind, ncols)]

        #Top image
        R[l_ind_center] += prime_quad(top_image[l_ind_center] - top_image[l_ind_left])
        R[l_ind_center] += prime_quad(top_image[l_ind_center] - top_image[l_ind_right])
        R[l_ind_center] += prime_quad(top_image[l_ind_center] - top_image[l_ind_top])
        R[l_ind_center] += prime_quad(top_image[l_ind_center] - top_image[l_ind_bottom])

        #Bottom dli_images
        R[l_ind_center+N] +=
            prime_quad(bottom_image[l_ind_center] - bottom_image[l_ind_left])
        R[l_ind_center+N] +=
            prime_quad(bottom_image[l_ind_center] - bottom_image[l_ind_right])
        R[l_ind_center+N] += prime_quad(bottom_image[l_ind_center] - bottom_image[l_ind_top])
        R[l_ind_center+N] +=
            prime_quad(bottom_image[l_ind_center] - bottom_image[l_ind_bottom])
    end
end

function ∇R_quadratic_mul!(v, u, p_prime, t)
    R = zeros(eltype(v), length(v))
    ∇R_quadratic_mul!(R, v, u, p_prime, t)
    R
end

### Quadratic Edge-Weighted Regularization, as in http://dx.doi.org/10.1118/1.4866386

function R_quad_ew_mul!(R, v, u, p, t)
    N, nrows, ncols, linear_ind, (combined_edges, edge_val, non_edge_val) = p
    top_image = view(v, 1:N)
    bottom_image = view(v, N+1:2N)

    R[] = zero(eltype(R[])) # Initialize accumulator

    for ind in CartesianIndices((nrows, ncols)) #dont use Threads here because thread race
        #v is a Vector, but it is much easier to deals with 2D image. It is however slow to reshape v.
        # We use linear_ind to convert 2D CartesianIndex to Vector linear indices. 
        ind_left, ind_right, ind_top, ind_bottom =
            left(ind, nrows), right(ind, nrows), top(ind, ncols), bottom(ind, ncols)

        is_center_pixel_edge = combined_edges[ind]

        # the edge-detection weight is a small value if either i or k is 
        # the index of an edge pixel in the image and one otherwise
        center_left_weight =
            is_center_pixel_edge || combined_edges[ind_left] ? edge_val : non_edge_val
        center_right_weight =
            is_center_pixel_edge || combined_edges[ind_right] ? edge_val : non_edge_val
        center_top_weight =
            is_center_pixel_edge || combined_edges[ind_top] ? edge_val : non_edge_val
        center_bottom_weight =
            is_center_pixel_edge || combined_edges[ind_bottom] ? edge_val : non_edge_val

        l_ind_center = linear_ind[ind]
        l_ind_left = linear_ind[ind_left]
        l_ind_right = linear_ind[ind_right]
        l_ind_top = linear_ind[ind_top]
        l_ind_bottom = linear_ind[ind_bottom]

        #Top image
        R[] += quad_ew(top_image[l_ind_center] - top_image[l_ind_left], center_left_weight)
        R[] +=
            quad_ew(top_image[l_ind_center] - top_image[l_ind_right], center_right_weight)
        R[] += quad_ew(top_image[l_ind_center] - top_image[l_ind_top], center_top_weight)
        R[] += quad_ew(top_image[l_ind_center] - top_image[l_ind_bottom], center_bottom_weight)

        #Bottom image
        R[] += quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_left],
            center_left_weight,
        )
        R[] += quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_right],
            center_right_weight,
        )
        R[] +=
            quad_ew(bottom_image[l_ind_center] - bottom_image[l_ind_top], center_top_weight)
        R[] += quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_bottom],
            center_bottom_weight,
        )

    end

end

function R_quad_ew_mul!(v, u, p, t)
    R = Ref(zero(eltype(v))) #R in not a FLoat in the LinearSovle "solve" function
    R_quad_ew_mul!(R, v, u, p, t)
    R[]
end

function ∇R_quad_ew_mul!(R, v, u, p_prime, t)
    N, nrows, ncols, linear_ind, (combined_edges, edge_val, non_edge_val) = p_prime
    top_image = view(v, 1:N)
    bottom_image = view(v, N+1:2N)

    fill!(R, zero(eltype(R))) # Initialize accumulator

    for ind in CartesianIndices((nrows, ncols)) #dont use Threads here because thread race
        ind_left, ind_right, ind_top, ind_bottom =
            left(ind, nrows), right(ind, nrows), top(ind, ncols), bottom(ind, ncols)

        is_center_pixel_edge = combined_edges[ind]

        # the edge-detection weight is a small value if either i or k is 
        # the index of an edge pixel in the image and one otherwise
        center_left_weight =
            is_center_pixel_edge || combined_edges[ind_left] ? edge_val : non_edge_val
        center_right_weight =
            is_center_pixel_edge || combined_edges[ind_right] ? edge_val : non_edge_val
        center_top_weight =
            is_center_pixel_edge || combined_edges[ind_top] ? edge_val : non_edge_val
        center_bottom_weight =
            is_center_pixel_edge || combined_edges[ind_bottom] ? edge_val : non_edge_val

        l_ind_center = linear_ind[ind]
        l_ind_left = linear_ind[ind_left]
        l_ind_right = linear_ind[ind_right]
        l_ind_top = linear_ind[ind_top]
        l_ind_bottom = linear_ind[ind_bottom]

        #Top image
        R[l_ind_center] += prime_quad_ew(
            top_image[l_ind_center] - top_image[l_ind_left],
            center_left_weight,
        )
        R[l_ind_center] += prime_quad_ew(
            top_image[l_ind_center] - top_image[l_ind_right],
            center_right_weight,
        )
        R[l_ind_center] +=
            prime_quad_ew(top_image[l_ind_center] - top_image[l_ind_top], center_top_weight)
        R[l_ind_center] += prime_quad_ew(
            top_image[l_ind_center] - top_image[l_ind_bottom],
            center_bottom_weight,
        )

        #Bottom image
        R[l_ind_center+N] += prime_quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_left],
            center_left_weight,
        )
        R[l_ind_center+N] += prime_quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_right],
            center_right_weight,
        )
        R[l_ind_center+N] += prime_quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_top],
            center_top_weight,
        )
        R[l_ind_center+N] += prime_quad_ew(
            bottom_image[l_ind_center] - bottom_image[l_ind_bottom],
            center_bottom_weight,
        )
    end

end

function ∇R_quad_ew_mul!(v, u, p_prime, t)
    R = zeros(eltype(v), length(v)) #R must have the sme type than v, with is not a Vecotr{Float64} in the LinearSovle "solve" function
    ∇R_quad_ew_mul!(R, v, u, p_prime, t)
    R
end

### Regularization based on similiarity matrix (fixed version, as in the paper)
"""
The "Gather" operation for `z = W * v`, with W the similarity matrix 
For each output element `z[i]`, it gathers weighted values from input neighbors `v[k]`.
"""
function W_mul!(R, v, u, p, t)
    # Unpack parameters
    N, nrows, ncols, linear_ind, (dli_images, h_top, h_bottom, half_size) = p
    cartesian_ind = CartesianIndices((nrows, ncols))

    fill!(R, zero(eltype(R))) # Initialize accumulator

    # Loop over output elements
    for ind in cartesian_ind
        center_ind = linear_ind[ind]
        i, j = ind[1], ind[2]

        ni_x = max(1, i - half_size):min(nrows, i + half_size)
        ni_y = max(1, j - half_size):min(ncols, j + half_size)

        neighbor_indices_top, weights_top = _calculate_similarity_row(
            dli_images.top,
            linear_ind,
            cartesian_ind,
            ind,
            h_top,
            ni_x,
            ni_y,
            nrows,
            ncols,
            half_size
        )
        neighbor_indices_bottom, weights_bottom = _calculate_similarity_row(
            dli_images.bottom,
            linear_ind,
            cartesian_ind,
            ind,
            h_bottom,
            ni_x,
            ni_y,
            nrows,
            ncols,
            half_size
        )

        # Weighted sum of NEIGHBOR values from v
        #Top image
        dot_product_top = 0.0
        for (idx, k) in enumerate(neighbor_indices_top)
            dot_product_top += weights_top[idx] * v[k]
        end
        R[center_ind] = dot_product_top

        #Bottom image
        dot_product_bottom = 0.0
        for (idx, k) in enumerate(neighbor_indices_bottom)
            dot_product_bottom += weights_bottom[idx] * v[k+N]
        end
        R[center_ind+N] = dot_product_bottom
    end
end

function W_mul!(v, u, p, t)
    R = zeros(eltype(v), length(v))
    W_mul!(R, v, u, p, t)
    R
end

"""
The "Scatter" operation for `z = Wᵀ * v`, with W the similarity matrix 
For each input element `v[i]`, it scatters its value to its neighbors in the output `z`.
"""
function Wt_mul!(R, v, u, p, t)
    # Unpack parameters
    N, nrows, ncols, linear_ind, (dli_images, h_top, h_bottom, half_size) = p
    cartesian_ind = CartesianIndices((nrows, ncols))

    fill!(R, zero(eltype(R))) # Initialize accumulator

    # Loop over output elements
    for ind in cartesian_ind
        center_ind = linear_ind[ind]
        i, j = ind[1], ind[2]

        ni_x = max(1, i - half_size):min(nrows, i + half_size)
        ni_y = max(1, j - half_size):min(ncols, j + half_size)

        neighbor_indices_top, weights_top = _calculate_similarity_row(
            dli_images.top,
            linear_ind,
            cartesian_ind,
            ind,
            h_top,
            ni_x,
            ni_y,
            nrows,
            ncols,
            half_size
        )
        neighbor_indices_bottom, weights_bottom = _calculate_similarity_row(
            dli_images.bottom,
            linear_ind,
            cartesian_ind,
            ind,
            h_bottom,
            ni_x,
            ni_y,
            nrows,
            ncols,
            half_size
        )

        # Add the contribution of the CURRENT value v[i] to its neighbors in z
        #Top image
        for (idx, k) in enumerate(neighbor_indices_top)
            R[k] += weights_top[idx] * v[center_ind] # Note the indices: z[k] and v[i]
        end

        #Bottom image
        for (idx, k) in enumerate(neighbor_indices_bottom)
            R[k+N] += weights_bottom[idx] * v[center_ind+N] # Note the indices: z[k] and v[i]
        end
    end
end

function Wt_mul!(v, u, p, t)
    R = zeros(eltype(v), length(v))
    Wt_mul!(R, v, u, p, t)
    R
end

#qGGMRF is not used 
function R_qGGMRF_mul!(R, v, u, p, t)
    N, nrows, ncols, linear_ind, (p2, q, c) = p
    top_image = view(v, 1:N)
    bottom_image = view(v, N+1:2N)

    R[] = zero(eltype(R[])) # Initialize accumulator

    for ind in CartesianIndices((nrows, ncols)) #dont use Threads here because thread race
        #v is a Vector, but it is much easier to deals with 2D image. It is however slow to reshape v.
        # We use linear_ind to convert 2D CartesianIndex to Vector linear indices. 
        l_ind_center = linear_ind[ind]
        l_ind_left = linear_ind[left(ind, nrows)]
        l_ind_right = linear_ind[right(ind, nrows)]
        l_ind_top = linear_ind[top(ind, ncols)]
        l_ind_bottom = linear_ind[bottom(ind, ncols)]

        R[] += qggmrf(top_image[l_ind_center] - top_image[l_ind_left], p2, q, c)
        R[] += qggmrf(top_image[l_ind_center] - top_image[l_ind_right], p2, q, c)
        R[] += qggmrf(top_image[l_ind_center] - top_image[l_ind_top], p2, q, c)
        R[] += qggmrf(top_image[l_ind_center] - top_image[l_ind_bottom], p2, q, c)

        R[] += qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_left], p2, q, c)
        R[] += qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_right], p2, q, c)
        R[] += qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_top], p2, q, c)
        R[] += qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_bottom], p2, q, c)
    end

end

function R_qGGMRF_mul!(v, u, p, t)
    R = Ref(zero(eltype(v))) #R in not a FLoat in the LinearSovle "solve" function
    R_qGGMRF_mul!(R, v, u, p, t)
    R[]
end

#= L = vcat(vec(dli_images.top), vec(dli_images.bottom))
nrows, ncols = size(dli_images.top)
N = nrows*ncols
linear_ind = LinearIndices((nrows, ncols))

p_qggmrf = (nrows, ncols, N, linear_ind, qggmrf, (2.0, 2.0, 1.0))
p_huber = (nrows, ncols, N, linear_ind, huber, 1.0)
R_op!(L, nothing, p_qggmrf, nothing)
R_op!(L, nothing, p_huber, nothing) =#


#mfop_qggmrf = FunctionOperator(R_op!, similar(L), [1.0];  p = p_qggmrf)
#mfop_huber = FunctionOperator(R_op!, similar(L), [1.0];  p = p_huber)

#mfop_qggmrf * L
#mfop_huber * L

function ∇R_qGGMRF_mul!(R, v, u, p_prime, t)
    N, nrows, ncols, linear_ind, (p2, q, c) = p_prime
    top_image = view(v, 1:N)
    bottom_image = view(v, N+1:2N)

    fill!(R, zero(eltype(R))) # Initialize accumulator

    for ind in CartesianIndices((nrows, ncols)) #dont use Threads here because thread race
        #v is a Vector, but it is much easier to deals with 2D image. It is however slow to reshape v.
        # We use linear_ind to convert 2D CartesianIndex to Vector linear indices. 
        l_ind_center = linear_ind[ind]
        l_ind_left = linear_ind[left(ind, nrows)]
        l_ind_right = linear_ind[right(ind, nrows)]
        l_ind_top = linear_ind[top(ind, ncols)]
        l_ind_bottom = linear_ind[bottom(ind, ncols)]

        #Top image
        R[l_ind_center] +=
            prime_qggmrf(top_image[l_ind_center] - top_image[l_ind_left], p2, q, c)
        R[l_ind_center] +=
            prime_qggmrf(top_image[l_ind_center] - top_image[l_ind_right], p2, q, c)
        R[l_ind_center] +=
            prime_qggmrf(top_image[l_ind_center] - top_image[l_ind_top], p2, q, c)
        R[l_ind_center] +=
            prime_qggmrf(top_image[l_ind_center] - top_image[l_ind_bottom], p2, q, c)

        #Bottom dli_images
        R[l_ind_center+N] +=
            prime_qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_left], p2, q, c)
        R[l_ind_center+N] +=
            prime_qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_right], p2, q, c)
        R[l_ind_center+N] +=
            prime_qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_top], p2, q, c)
        R[l_ind_center+N] +=
            prime_qggmrf(bottom_image[l_ind_center] - bottom_image[l_ind_bottom], p2, q, c)
    end

end

function ∇R_qGGMRF_mul!(v, u, p_prime, t)
    R = zeros(eltype(v), length(v))
    ∇R_qGGMRF_mul!(R, v, u, p_prime, t)
    R
end