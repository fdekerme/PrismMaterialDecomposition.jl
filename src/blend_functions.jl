"""
Change the vector v to the basis defined by the vectors μ₁ and μ₂
"""
function change_of_basis(v, μ₁::μ, μ₂::μ)
    M = [μ₁.low μ₂.low; μ₁.high μ₂.high]
    return M \ v #much better than inv(M) * v
end

"""
Change the vector v to the basis defined by the vectors μ₁ and μ₂
"""
function change_of_basis(v::μ, μ₁::μ, μ₂::μ)
    M = [μ₁.low μ₂.low; μ₁.high μ₂.high]
    res = M \ [v.low; v.high] #much better than inv(M) * v
    return μ(v.name * "_new_basis", res[1], res[2])
end

"""
This version is used for the optimization problem
"""
function change_of_basis(v, A_vec)
    M = reshape(A_vec, 2, 2)
    return M \ v #much better than inv(M) * v
end

function linear_blend_function(x, α)
    return clamp(α, 0, 1)
end

function linear_blend(dli_images, k)
    α = linear_blend_function.(dli_images.top, k)
    return Blend(
        "linear",
        α .* dli_images.top .+ (1 .- α) .* dli_images.bottom,
        α .* ones(size(dli_images.bottom)),
    )
end

function log_substraction(dli_images, w)
    return Blend(
        "linear",
        dli_images.bottom - w .* dli_images.top,
        w .* ones(size(dli_images.bottom)),
    )
end

function binary_blend_function(x, threshold)
    return x .>= threshold
end

function binary_blend(dli_images, threshold; reference_image = nothing)
    reference = reference_image == nothing ? dli_images.top : reference_image
    α = binary_blend_function.(reference, threshold) #booelan mask
    return Blend("binary", α .* dli_images.top .+ (1 .- α) .* dli_images.bottom, α)
end

function slope_blend_function(x, level, width)
    return clamp(width * (x - level), 0, 1)
end

function slope_blend(dli_images, level, width; reference_image = nothing)
    reference = reference_image == nothing ? dli_images.top : reference_image
    α = slope_blend_function.(reference, level, width) # level = - b / width => b = - level * width
    return Blend("slope", α .* dli_images.top .+ (1 .- α) .* dli_images.bottom, matrix(α))
end

function moidal_blend_function(x, level, width, offset)
    return clamp(1 / (1 + exp.(width * (x - level))) + offset, 0, 1)
end

function moidal_blend(dli_images, level, width, offset; reference_image = nothing)
    reference = reference_image == nothing ? dli_images.top : reference_image
    α = moidal_blend_function.(reference, level, width, offset)
    return Blend("moidal", α .* dli_images.top .+ (1 .- α) .* dli_images.bottom, matrix(α))
end

function gaussian_blend_function(x, level, width, offset)
    return clamp(exp.(width * (x - level) .^ 2) + offset, 0, 1)
end

function gaussian_blend(dli_images, level, width, offset; reference_image = nothing)
    reference = reference_image == nothing ? dli_images.top : reference_image
    α = gaussian_blend_function.(reference, level, width, offset)
    return Blend(
        "gaussian",
        α .* dli_images.top .+ (1 .- α) .* dli_images.bottom,
        matrix(α),
    )
end

"""
    material_decomposition(dli_images, μ₁, μ₂)

Solve the two-by-two material decomposition problem for every pixel using the
attenuation coefficients `μ₁` and `μ₂`.
Returns an `MI` container with the two material maps.
"""
function material_decomposition(dli_images::DLI, μ₁::μ, μ₂::μ)
    top_flat = reshape(dli_images.top, 1, :)
    bottom_flat = reshape(dli_images.bottom, 1, :)
    decomposed = change_of_basis(vcat(top_flat, bottom_flat), μ₁, μ₂)

    decomposed_mat1 = reshape(decomposed[1, :], size(dli_images.top))
    decomposed_mat2 = reshape(decomposed[2, :], size(dli_images.bottom))

    return MI(μ₁, μ₂, decomposed_mat1, decomposed_mat2)
end


"""
This version is used for the optimization problem
"""
function material_decomposition(dli_images::DLI, A_vec::Vector)
    top_flat = reshape(dli_images.top, 1, :)
    bottom_flat = reshape(dli_images.bottom, 1, :)
    decomposed = change_of_basis(vcat(top_flat, bottom_flat), A_vec)

    decomposed_mat1 = reshape(decomposed[1, :], size(dli_images.top))
    decomposed_mat2 = reshape(decomposed[2, :], size(dli_images.bottom))

    return (decomposed_mat1, decomposed_mat2)
end


"""
    VMI_blend(material_images, E, MAC_function_1, MAC_function_2)

Compute a virtual monoenergetic image at energy `E` from material images using
the provided mass-attenuation coefficient callbacks.
"""
function VMI_blend(material_images::MI, E, MAC_function_1, MAC_function_2)
    VMI =
        MAC_function_1(E) * material_images.mat1 .+
        MAC_function_2(E) .* material_images.mat2
    return Blend("VMI", VMI, ones(size(material_images.mat1)))
end
