"""
    DLI{T}

Container for paired dual-layer images acquired at two energy levels.

# Fields
- `top::T`: Low-energy / first-layer image data.
- `bottom::T`: High-energy / second-layer image data.
"""
struct DLI{T<:AbstractArray}
    top::T
    bottom::T
end

"""
    μ

Mass attenuation coefficients for a material at two effective energies.

# Fields
- `name::String`: Material identifier.
- `low::Float64`: Attenuation coefficient at the low-energy spectrum.
- `high::Float64`: Attenuation coefficient at the high-energy spectrum.
"""
struct μ
    name::String
    low::Float64
    high::Float64

    function μ(name::String, low::Float64, high::Float64)
        new(name, low, high)
    end

    function μ(name::String, low::Unitful.Quantity, high::Unitful.Quantity)
        new(name, val(low), val(high))
    end
end

"""
    Blend

Result of combining dual-layer images together with the blending weights.

# Fields
- `type::String`: Short label describing the blending strategy.
- `image::AbstractArray`: Blended image.
- `α::AbstractArray`: Weight map used during blending.
"""
struct Blend
    type::String
    image::AbstractArray
    α::AbstractArray
end

"""
    MI{T}

Material decomposition result storing two material density maps.

# Fields
- `μ₁::μ`: Attenuation model of the first material.
- `μ₂::μ`: Attenuation model of the second material.
- `mat1::T`: First material map.
- `mat2::T`: Second material map.
"""
struct MI{T<:AbstractArray}
    μ₁::μ
    μ₂::μ
    mat1::T
    mat2::T
end

"""
    Spectra{T}

Top and bottom spectral distributions for each pixel.

# Fields
- `top::T`: Low-energy spectra 
- `bottom::T`: High-energy spectra
- `is_weighted::Bool`: Flag indicating whether the spectra are energy-weighted i.e. I(E) (non weighted) vs I(E) * E (weighted).
"""
struct Spectra{T<:AbstractArray}
    top::T
    bottom::T
    is_weighted::Bool
end

"""
        Regularization(name, constructor, params)

Container that describes a regularizer used by the decomposition pipeline.

Fields
- `name::String` : human readable name of the regularizer (e.g. "Similarity").
- `constructor::Function` : a function that, given the problem shape and a prototype
    vector, returns a regularization operator (matrix or `FunctionOperator`) suitable
    for inclusion in the linear system. The constructor signature is
    `constructor(params, prototype)`. All regularization constructors are defined in 
    `regularization_constructors.jl`.
- `params::Tuple` : regularizer parameter tuple passed to the constructor.

Usage
- Instances of this struct are passed to the problem constructors (CPU/CUDA/Metal)
    which call `reg.constructor(...)` to build `∇R` and set `regularization_name`.

Notes
- The `constructor` must produce operators compatible with the `SciMLOperators` and
    `LinearSolve` framework used elsewhere in the repository. For GPU usage the
    constructor should return GPU-backed operators (Cu/Mtl types).
"""
struct Regularization
    name::String
    constructor::Function
    params::Tuple
end


"""
    ESF

Edge spread function sampled along a one-dimensional profile.

# Fields
- `x::AbstractVector`: Profile positions.
- `y::AbstractVector`: Measured intensities.
"""
struct ESF
    x::AbstractVector
    y::AbstractVector
end

"""
    LSF

Line spread function obtained by differentiating an ESF.

# Fields
- `x::AbstractVector`: Profile positions.
- `y::AbstractVector`: Gradient magnitudes.
"""
struct LSF
    x::AbstractVector
    y::AbstractVector
end

"""
    Circ

Circle geometry helper used for edge detection and ESF extraction.

# Fields
- `center::Tuple{Float64,Float64}`: Circle centre `(row, column)` in pixels.
- `radius::Float64`: Circle radius in pixels.
"""
struct Circ
    center::Tuple{Float64,Float64}
    radius::Float64
end
