# --- Edges detection ---

canny_edge(img, upper_threshold_ratio, lower_threshold_ratio; sigma = 1.4) = canny(
    img,
    (Percentile(100 * upper_threshold_ratio), Percentile(100 * lower_threshold_ratio)),
    sigma,
)
#prewitt_edge(img) = imfilter(img, Kernel.prewitt())
#sobel_edge(img) = imfilter(img, Kernel.sobel())

"""
    compute_edge_map(high_energy_img, low_energy_img; upper=0.8, lower=0.2)

Detects edges on high and low energy images using the Canny method and
combines them into a single binary edge map. 
Use the edges weights as in http://dx.doi.org/10.1118/1.4866386

# Arguments
- `high_energy_img::Matrix`: The high energy CT image.
- `low_energy_img::Matrix`: The low energy CT image.
- `upper::Real`: The high threshold for the Canny edge detector.
- `lower::Real`: The low threshold for the Canny edge detector.

# Returns
- `BitMatrix`: A binary matrix of the same size as the input images, where
  `true` indicates the presence of an edge.
"""
function compute_edge_map(dli_images::DLI, upper::Real = 0.8, lower::Real = 0.2)
    # 1. Detect edges on the high energy CT image using the Canny method.
    edges_top = canny_edge(dli_images.top, upper, lower)

    # 2. Detect edges on the low energy CT image.
    edges_bottom = canny_edge(dli_images.bottom, upper, lower)

    # 3. Combine the edge maps. A pixel is considered an edge if it's
    # detected in either image. This is a robust approach.
    combined_edges = edges_top .| edges_bottom

    return combined_edges
end

# Function that deals with border effects. It returns the same index if it is on the border, so that the difference is zero.
left(ind::CartesianIndex, nrows) = ind[1] == 1 ? ind : CartesianIndex(ind[1] - 1, ind[2])
right(ind::CartesianIndex, nrows) =
    ind[1] == nrows ? ind : CartesianIndex(ind[1] + 1, ind[2])
bottom(ind::CartesianIndex, ncols) =
    ind[2] == ncols ? ind : CartesianIndex(ind[1], ind[2] + 1)
top(ind::CartesianIndex, ncols) = ind[2] == 1 ? ind : CartesianIndex(ind[1], ind[2] - 1)

# ---Quadratic regularization ---
quad(δ::Real) = 0.5 * δ^2
prime_quad(δ::Real) = δ

function _calculate_quad_ew_row(
    linear_ind,
    center_ind,
    nrows,
    ncols,
    combined_edges,
    edge_val,
    non_edge_val,
)

    ind_left, ind_right, ind_top, ind_bottom = left(center_ind, nrows),
    right(center_ind, nrows),
    top(center_ind, ncols),
    bottom(center_ind, ncols)

    is_center_pixel_edge = combined_edges[center_ind] #check if the center pixel i is an edge pixel in the image

    # the edge-detection weight is a small value if either i or k is 
    # the index of an edge pixel in the image and one otherwise
    center_weight = is_center_pixel_edge ? edge_val : non_edge_val
    center_left_weight =
        is_center_pixel_edge || combined_edges[ind_left] ? edge_val : non_edge_val
    center_right_weight =
        is_center_pixel_edge || combined_edges[ind_right] ? edge_val : non_edge_val
    center_top_weight =
        is_center_pixel_edge || combined_edges[ind_top] ? edge_val : non_edge_val
    center_bottom_weight =
        is_center_pixel_edge || combined_edges[ind_bottom] ? edge_val : non_edge_val

    #We then set the weights to zero for border pixels. We can do that because of the definition of left, right, top and bottom functions.
    #For instance, if the left pixel if out of the image, then left(ind) = ind
    center_left_weight = ind_left == center_ind ? 0.0 : center_left_weight
    center_right_weight = ind_right == center_ind ? 0.0 : center_right_weight
    center_top_weight = ind_top == center_ind ? 0.0 : center_top_weight
    center_bottom_weight = ind_bottom == center_ind ? 0.0 : center_bottom_weight

    weights = [
        center_weight,
        center_left_weight,
        center_right_weight,
        center_top_weight,
        center_bottom_weight,
    ]
    indices = [
        linear_ind[center_ind],
        linear_ind[ind_left],
        linear_ind[ind_right],
        linear_ind[ind_top],
        linear_ind[ind_bottom],
    ]
    return indices, weights
end


# ---Quadratic Edge-Weighted Regularization (http://dx.doi.org/10.1118/1.4866386) ---

"""
    quad_ew(δ, weight_at_pixel)

Potential function for quadratic edge-weighted regularization for a single difference.
ρ(δ, wᵢ) = 0.5 * wᵢ * δ²
"""
quad_ew(δ::Real, weight_at_pixel::Real) = 0.5 * weight_at_pixel * δ^2

"""
    prime_rho_quad_ew(δ, weight_at_pixel)

Derivative of the potential function for quadratic edge-weighted regularization.
ρ'(δ, wᵢ) = wᵢ * δ
"""
prime_quad_ew(δ::Real, weight_at_pixel::Real) = weight_at_pixel * δ

# ---Similarity-based Regularization (http://dx.doi.org/10.1118/1.4947485) ---

gaussian(δ, h::Float64) = exp(-δ^2 / h^2)

#Distance matrics with the signature distance_metric(ind1::CartesianIndex, ind2::CartesianIndex, σ_spat)
euclidean_distance(ind1::CartesianIndex, ind2::CartesianIndex, σ_spat) = norm(Tuple(ind1) .- Tuple(ind2), 2) #σ_spat to keep the signature consistent 
gaussian_spatial_distance(ind1::CartesianIndex, ind2::CartesianIndex, σ_spat) =
    exp(-euclidean_distance(ind1, ind2, σ_spat)^2 / σ_spat^2)
no_distance_penalty(ind1::CartesianIndex, ind2::CartesianIndex, σ_spat) = 1.0  #σ_spat to keep the signature consistent 

"""
    _calculate_similarity_row(
        img::Matrix{Float64}, 
        linear_ind::LinearIndices, 
        cartesian_ind::CartesianIndices, 
        center_ind_cart::CartesianIndex, 
        h::Float64, 
        ni_x,
        ni_y, 
        nrows::Int, 
        ncols::Int; 
        n_iter::Int = 1, 
        n_neighbors::Int = 200
    )

Compute the similarity weights between a center pixel and its neighbors within a specified window in an image.

# Arguments
- `img`: 2D array representing the image data.
- `linear_ind`: LinearIndices object for mapping Cartesian indices to linear indices.
- `cartesian_ind`: CartesianIndices object for mapping linear indices to Cartesian indices.
- `center_ind_cart`: CartesianIndex of the center pixel.
- `h`: Bandwidth parameter for the Gaussian similarity function.
- `ni_x`: Range of row indices for the neighborhood window.
- `ni_y`: Range of column indices for the neighborhood window.
- `nrows`: Number of rows in the image.
- `ncols`: Number of columns in the image.

# Keyword Arguments
- `n_iter`: Current iteration count for expanding the neighborhood window (default: 1).
- `n_neighbors`: Minimum number of neighbors to select (default: 0).

# Returns
- `neighbor_indices`: Vector of linear indices of selected neighbor pixels.
- `normalized_weights`: Vector of normalized similarity weights for each neighbor.

# Notes
- The function recursively expands the neighborhood window if the number of selected neighbors is less than `n_neighbors`, up to a maximum of 3 iterations.
- Similarity between pixels is computed using a Gaussian function, and only neighbors with intensity difference less than `3h` are considered.
"""
function _calculate_similarity_row(
    img::Matrix{Float64},
    linear_ind::LinearIndices,
    cartesian_ind::CartesianIndices,
    center_ind_cart::CartesianIndex,
    h::Float64,
    ni_x,
    ni_y,
    nrows::Int,
    ncols::Int,
    half_size::Int;
    n_iter::Int = 1,
    n_neighbors::Int = 0, #deactiveted by default. It is 200 in the Harms paper
    distance_metric::Function = no_distance_penalty,
) :: Tuple{Vector{Int}, Vector{Float64}}
    window_indices_cart = view(cartesian_ind, ni_x, ni_y) #neighborhood of the pixel caracterized as center_val
    center_val = img[center_ind_cart]

    neighbor_indices_linear = Int[]
    unnormalized_weights = Float64[]
    distances = Float64[]

    nb_selected_pixel = 0
    for neighbor_ind_cart in window_indices_cart
        # Skip the center pixel itself
        neighbor_ind_cart == center_ind_cart && continue

        neighbor_ind_linear = linear_ind[neighbor_ind_cart]
        neighbor_val = img[neighbor_ind_cart] # == img[neighbor_ind_linear]

        # Check if the neighbor is similar based on pixel value difference
        if abs(center_val - neighbor_val) < 3h
            distance = distance_metric(center_ind_cart, neighbor_ind_cart, half_size)
            s_ik = gaussian(center_val - neighbor_val, h)

            push!(neighbor_indices_linear, neighbor_ind_linear)
            push!(unnormalized_weights, s_ik)
            push!(distances, distance)
            nb_selected_pixel += 1
        end
    end

    # Recursive loop to ensure enough similar pixel have been selectred. 
    # The mini number of similar pixel selected is n_neighbors = 200 in the Harms et al. paper. 
    # This function is desactivated by default here (n_neighbors = 0)
    if nb_selected_pixel < n_neighbors && n_iter <= 3
        ni_x_extended = max(1, ni_x[1] - 10):min(nrows, ni_x[end] + 10)
        ni_y_extended = max(1, ni_y[1] - 10):min(ncols, ni_y[end] + 10)
        return _calculate_similarity_row(
            img,
            linear_ind,
            cartesian_ind,
            center_ind_cart,
            h,
            ni_x_extended,
            ni_y_extended,
            nrows,
            ncols,
            half_size;
            n_iter = n_iter + 1,
        )
    end

    # Normalize the weights.

    if isempty(unnormalized_weights)
        # If no valid neighbors were found, use the center pixel as the only neighbor.
        neighbor_indices_linear = [linear_ind[center_ind_cart]]
        normalized_weights = [1.0]
    else
        weights = unnormalized_weights .* distances
        norma = sum(weights)
        normalized_weights = weights ./ norma
    end

    return neighbor_indices_linear, normalized_weights
end

"""
    _calculate_cross_similarity_row(
        dli_images::DLI, 
        linear_ind::LinearIndices, 
        cartesian_ind::CartesianIndices, 
        center_ind_cart::CartesianIndex, 
        h_top::Float64, 
        h_bottom::Float64,
        ni_x,
        ni_y, 
        nrows::Int, 
        ncols::Int,
        half_size::Int; 
        n_iter::Int = 1, 
        n_neighbors::Int = 200
    )

Compute the cross similarity weights between a center pixel and its neighbors within a specified window in an image.

# Arguments
- `dli_images`: DLI object containing top and bottom images.
- `linear_ind`: LinearIndices object for mapping Cartesian indices to linear indices.
- `cartesian_ind`: CartesianIndices object for mapping linear indices to Cartesian indices.
- `center_ind_cart`: CartesianIndex of the center pixel.
- `h`: Bandwidth parameter for the Gaussian similarity function.
- `ni_x`: Range of row indices for the neighborhood window.
- `ni_y`: Range of column indices for the neighborhood window.
- `nrows`: Number of rows in the image.
- `ncols`: Number of columns in the image.
- `half_size`: Half size of the neighborhood window.

# Keyword Arguments
- `n_iter`: Current iteration count for expanding the neighborhood window (default: 1).
- `n_neighbors`: Minimum number of neighbors to select (default: 0).

# Returns
- `neighbor_indices`: Vector of linear indices of selected neighbor pixels.
- `normalized_weights`: Vector of normalized similarity weights for each neighbor.

# Notes
- The function recursively expands the neighborhood window if the number of selected neighbors is less than `n_neighbors`, up to a maximum of 3 iterations.
- Similarity between pixels is computed using a Gaussian function, and only neighbors with intensity difference less than `3h` are considered.
"""
function _calculate_cross_similarity_row(
    dli_images::DLI,
    linear_ind::LinearIndices,
    cartesian_ind::CartesianIndices,
    center_ind_cart::CartesianIndex,
    h_top::Float64,
    h_bottom::Float64,
    ni_x,
    ni_y,
    nrows::Int,
    ncols::Int,
    half_size::Int;
    n_iter::Int = 1,
    n_neighbors::Int = 0, #deactiveted by default. It is 200 in the Harms paper
    distance_metric::Function = gaussian_spatial_distance # function with signature distance_metric(ind1::CartesianIndex, ind2::CartesianIndex, half_size::Int)
) :: Tuple{Vector{Int}, Vector{Float64}}
    window_indices_cart = view(cartesian_ind, ni_x, ni_y)
    top_img = dli_images.top
    bottom_img = dli_images.bottom

    center_val_top = top_img[center_ind_cart]
    center_val_bottom = bottom_img[center_ind_cart]

    neighbor_indices_linear = Int[]
    unnormalized_weights = Float64[]
    distances = Float64[]

    nb_selected_pixel = 0
    for neighbor_ind_cart in window_indices_cart
        neighbor_ind_cart == center_ind_cart && continue

        neighbor_ind_linear = linear_ind[neighbor_ind_cart]
        neighbor_val_top = top_img[neighbor_ind_cart]
        neighbor_val_bottom = bottom_img[neighbor_ind_cart]

        if (abs(center_val_top - neighbor_val_top) < 3h_top) &&
           (abs(center_val_bottom - neighbor_val_bottom) < 3h_bottom)
            distance = distance_metric(center_ind_cart, neighbor_ind_cart, half_size)
            s_ik = 
                    gaussian(center_val_top - neighbor_val_top, h_top) *
                gaussian(center_val_bottom - neighbor_val_bottom, h_bottom)
                
            push!(neighbor_indices_linear, neighbor_ind_linear)
            push!(unnormalized_weights, s_ik)
            push!(distances, distance)
            nb_selected_pixel += 1
        end
    end

    # Recursive loop to ensure enough similar pixel have been selectred. 
    # The mini number of similar pixel selected is n_neighbors = 200 in the Harms et al. paper. 
    # This function is desactivated by default here (n_neighbors = 0)
    if nb_selected_pixel < n_neighbors && n_iter <= 3 
        ni_x_extended = max(1, ni_x[1] - 10):min(nrows, ni_x[end] + 10)
        ni_y_extended = max(1, ni_y[1] - 10):min(ncols, ni_y[end] + 10)
        return _calculate_cross_similarity_row(
            dli_images,
            linear_ind,
            cartesian_ind,
            center_ind_cart,
            h_top,
            h_bottom,
            ni_x_extended,
            ni_y_extended,
            nrows,
            ncols,
            half_size;
            n_iter = n_iter + 1,
            n_neighbors = n_neighbors,
        )
    end

    if isempty(unnormalized_weights)
        # If no valid neighbors were found, use the center pixel as the only neighbor.
        neighbor_indices_linear = [linear_ind[center_ind_cart]]
        normalized_weights = [1.0]
    else
        weights = unnormalized_weights .* distances
        norma = sum(weights)
        normalized_weights = weights ./ norma
    end

    return neighbor_indices_linear, normalized_weights
end

# ---Huber loss ---
"""
    huber(δ, α)

Calculates the value of the Huber loss function.
ρ(δ; α) = 0.5 * δ^2 if |δ| ≤ α, else α * (|δ| - 0.5 * α).
"""
function huber(δ::Real, α::Real)
    abs_δ = abs(δ)
    if abs_δ <= α
        return 0.5 * δ^2
    else
        return α * (abs_δ - 0.5 * α)
    end
end



#= # --- q-GGMRF regularization (not used, from http://dx.doi.org/10.1118/1.2789499) ---

"""
    qggmrf(δ, p, q, c)

Calculates the value of the q-GGMRF potential function ρ(δ).

# Arguments
- `δ::Real`: The difference in value between adjacent pixels.
- `p::Real`: Power parameter for regions near the origin (low contrast).
- `q::Real`: Power parameter for regions distant from the origin (high contrast).
- `c::Real`: Threshold parameter for transitioning between low and high contrast.

# Returns
- The scalar result of the potential function.
"""
function qggmrf(δ::Real, p::Real, q::Real, c::Real)
    abs_δ = abs(δ)
    # Add a small epsilon for numerical stability if δ and c are zero.
    denominator = 1.0 + (abs_δ / abs(c))^(p - q)
    return (abs_δ^p) / denominator
end

function plot_qggmrf(δ, p, q, c)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "q-GGMRF regularization", xlabel = "δ", ylabel = "ρ(δ)")
    lines!(ax, δ, qggmrf.(δ, [p], [q], [c]), label = "p=$p, q=$q, c=$c")
    axislegend(ax)
    return fig
end

function plot_qggmrf!(ax::Axis, δ, p, q, c)
    lines!(ax, δ, qggmrf.(δ, [p], [q], [c]), label = "p=$p, q=$q, c=$c")
end

"""
    prime_qggmrf(δ, p, q, c)

Calculates the derivative of the q-GGMRF potential function, ρ'(δ).
This derivative is known as the influence function.
This is required for calculating the gradient of the regularization term.

# Arguments
- `δ::Real`: The difference in value between adjacent pixels.
- `p::Real`: Power parameter for regions near the origin.
- `q::Real`: Power parameter for regions distant from the origin.
- `c::Real`: Threshold parameter for transition.

# Returns
- The scalar result of the derivative of the potential function.
"""
function prime_qggmrf(δ::Real, p::Real, q::Real, c::Real)
    # Handle the case where δ is zero to avoid NaN from sign(0) * ∞
    if δ == 0.0
        return 0.0
    end

    abs_δ = abs(δ)
    c_stable = c + eps(Float64) # Avoid division by zero if c is 0

    # Pre-calculate common terms
    abs_δ_over_c_pq = (abs_δ / c_stable)^(p - q)
    denominator = (1.0 + abs_δ_over_c_pq)^2

    # Numerator calculation based on the analytical derivative
    # ρ'(δ) = sgn(δ) * [ p*|δ|^(p-1) + q*|δ|^(2p-q-1)/c^(p-q) ] / ( 1 + (|δ|/c)^(p-q) )^2
    term1 = p * abs_δ^(p - 1)
    term2 = q * (abs_δ^(2p - q - 1)) / (c_stable^(p - q))

    numerator = term1 + term2

    return sign(δ) * numerator / denominator
end

function plot_prime_qggmrf(δ, p, q, c)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Derivative of q-GGMRF Potential Function",
        xlabel = "δ",
        ylabel = "ρ'(δ)",
    )
    lines!(ax, δ, prime_qggmrf.(δ, [p], [q], [c]), label = "p=$p, q=$q, c=$c")
    axislegend(ax)
    return fig
end

function plot_prime_qggmrf!(ax::Axis, δ, p, q, c)
    lines!(ax, δ, prime_qggmrf.(δ, [p], [q], [c]), label = "p = $p, q = $q, c = $c")
end

 =#