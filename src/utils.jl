# Utilities and dispatches for DLI and MI types

"""
    cond(μ₁, μ₂)

Condition number of the mixing matrix defined by two attenuation models.
"""
function cond(μ₁::μ, μ₂::μ)
    mixing = [μ₁.low μ₂.low; μ₁.high μ₂.high]
    return LinearAlgebra.cond(mixing)
end

"""
    imrotate(dli, angle)

Rotate both layers of a `DLI` container by `angle` radians.
"""
function imrotate(dli::DLI, angle)
    rotated_top = Matrix(imrotate(dli.top, angle))
    rotated_bottom = Matrix(imrotate(dli.bottom, angle))
    return DLI(rotated_top, rotated_bottom)
end

"""
    imfilter(dli, kernel)

Apply an image filter to both layers of a `DLI` container.
"""
function imfilter(dli::DLI, kernel)
    filtered_top = Matrix(imfilter(dli.top, kernel))
    filtered_bottom = Matrix(imfilter(dli.bottom, kernel))
    return DLI(filtered_top, filtered_bottom)
end

"""
    imfilter(mi, kernel)

Apply an image filter independently to each material map in an `MI` container.
"""
function imfilter(materials::MI, kernel)
    filtered_mat1 = Matrix(imfilter(materials.mat1, kernel))
    filtered_mat2 = Matrix(imfilter(materials.mat2, kernel))
    return MI(materials.μ₁, materials.μ₂, filtered_mat1, filtered_mat2)
end

"""
    adjust_histogram(image, algorithm)

Apply an Images.jl histogram adjustment algorithm to an array and return the
adjusted data.
"""
function adjust_histogram(image, algorithm::AbstractHistogramAdjustmentAlgorithm)
    return adjust_histogram(image, algorithm)
end

"""
    adjust_histogram(dli, algorithm)

Adjust the histogram of both channels in a dual-layer image container.
"""
function adjust_histogram(dli::DLI, algorithm::AbstractHistogramAdjustmentAlgorithm)
    adjusted_top = adjust_histogram(dli.top, algorithm)
    adjusted_bottom = adjust_histogram(dli.bottom, algorithm)
    return DLI(adjusted_top, adjusted_bottom)
end

"""
    adjust_histogram(mi, algorithm)

Adjust the histogram of both material maps in a decomposition result.
"""
function adjust_histogram(materials::MI, algorithm::AbstractHistogramAdjustmentAlgorithm)
    adjusted_mat1 = adjust_histogram(materials.mat1, algorithm)
    adjusted_mat2 = adjust_histogram(materials.mat2, algorithm)
    return MI(materials.μ₁, materials.μ₂, adjusted_mat1, adjusted_mat2)
end


"""
    normalize(dli)

Normalise both layers of a dual-layer image to `[0, 1]`.
"""
function normalize(dli::DLI)
    return DLI(normalize(dli.top), normalize(dli.bottom))
end

"""
    normalize(mi)

Normalise both material maps to `[0, 1]`.
"""
function normalize(materials::MI)
    return MI(
        materials.μ₁,
        materials.μ₂,
        normalize(materials.mat1),
        normalize(materials.mat2),
    )
end

"""
    matrix(x)

Convert images or material decompositions to dense `Array{Float64}` storage.
"""
matrix(dli::DLI) = DLI(Array{Float64}(dli.top), Array{Float64}(dli.bottom))
matrix(materials::MI) = MI(
    materials.μ₁,
    materials.μ₂,
    Array{Float64}(materials.mat1),
    Array{Float64}(materials.mat2),
)
matrix(arr::AbstractArray) = Array{Float64}(arr)

"""
    rectangle(xmin, xmax, ymin, ymax)

Create a `Makie.Rect` representing a rectangular region of interest.
"""
function rectangle(xmin::Int, xmax::Int, ymin::Int, ymax::Int)
    width = xmax - xmin
    height = ymax - ymin
    return Rect(xmin, ymin, width, height)
end

"""
    extract_pixels(image, rect)

Extract a sub-matrix from `image` using the bounds of `rect`.
"""
function extract_pixels(image::AbstractMatrix, rect::Rect)
    xmin = rect.origin[1]
    ymin = rect.origin[2]
    xmax = xmin + rect.widths[1]
    ymax = ymin + rect.widths[2]

    if xmin < 1 || ymin < 1 || xmax > size(image, 1) || ymax > size(image, 2)
        error("Rectangle is out of bounds")
    end

    return image[xmin:xmax, ymin:ymax]
end

"""
    extract_pixels(dli, rect)

Extract corresponding regions from both channels of a dual-layer image.
"""
function extract_pixels(dli::DLI, rect::Rect)
    return DLI(extract_pixels(dli.top, rect), extract_pixels(dli.bottom, rect))
end

"""
    extract_pixels(material_images, rect)

Extract corresponding regions from both channels of a dual-layer image.
"""
function extract_pixels(material_images::MI, rect::Rect)
    return MI(
        material_images.μ₁,
        material_images.μ₂,
        extract_pixels(material_images.mat1, rect),
        extract_pixels(material_images.mat2, rect),
    )
end
#TODO: Implement a generic function that extracts pixels for any shape

"""
     compute_mean_energy(spectra, is_weighted)
    
    Compute the global mean energy of the top and bottom spectra, optionally applying energy weighting.
"""
function compute_mean_energy(spectra::Spectra)
    energy_bins = collect(axes(spectra.top, 3))
    spectra_top = deepcopy(spectra.top)
    spectra_bottom = deepcopy(spectra.bottom)

    if spectra.is_weighted # the spectrum is I(E) x E
        spectra_top ./= reshape(energy_bins, 1, 1, :)
        spectra_bottom ./= reshape(energy_bins, 1, 1, :)
    end

    mean_weights_top = dropdims(mean(spectra_top, dims = (1, 2)), dims = (1, 2))
    mean_weights_bottom = dropdims(mean(spectra_bottom, dims = (1, 2)), dims = (1, 2))

    mean_energy_top = sum(mean_weights_top .* energy_bins) / sum(mean_weights_top) * u"keV"
    mean_energy_bottom = sum(mean_weights_bottom .* energy_bins) / sum(mean_weights_bottom) * u"keV"

    return (top = mean_energy_top, bottom = mean_energy_bottom)
end

function compute_mean_energy(spectra::Spectra, row::Integer, col::Integer)

    spectra_top = deepcopy(spectra.top[row, col, :])
    spectra_bottom = deepcopy(spectra.bottom[row, col, :])
    energy_bins = collect(axes(spectra_top, 1))

    if spectra.is_weighted
        spectra_top ./= energy_bins
        spectra_bottom ./= energy_bins
    end

    mean_energy_top = sum(spectra_top .* energy_bins) / sum(spectra_top) * u"keV"
    mean_energy_bottom = sum(spectra_bottom .* energy_bins) / sum(spectra_bottom) * u"keV"

    return (
        top = mean_energy_top,
        bottom = mean_energy_bottom,
    )
end

function effective_mass_attenuations(spectra::Spectra, material)
    energy_bins = axes(spectra.top, 3)
    energies = collect(energy_bins) .* u"keV"
    spectra_top = deepcopy(spectra.top)
    spectra_bottom = deepcopy(spectra.bottom)

    if spectra.is_weighted
        spectra_top ./= reshape(energy_bins, 1, 1, :)
        spectra_bottom ./= reshape(energy_bins, 1, 1, :)
    end

    mean_weights_top = dropdims(mean(spectra_top, dims = (1, 2)), dims = (1, 2))
    mean_weights_bottom = dropdims(mean(spectra_bottom, dims = (1, 2)), dims = (1, 2))

    denominator_top = sum(mean_weights_top .* energy_bins)
    denominator_bottom = sum(mean_weights_bottom .* energy_bins)

    numerator_top = sum(mass_attenuation_coeff(material, energies) .* mean_weights_top .* energy_bins)
    numerator_bottom = sum(mass_attenuation_coeff(material, energies) .* mean_weights_bottom .* energy_bins)

    μ_eff_top = numerator_top / denominator_top
    μ_eff_bottom = numerator_bottom / denominator_bottom

    return μ(material.name, μ_eff_top, μ_eff_bottom)
end

function effective_linear_attenuations(spectra::Spectra, material)
    energy_bins = axes(spectra.top, 3)
    energies = collect(energy_bins) .* u"keV"
    spectra_top = deepcopy(spectra.top)
    spectra_bottom = deepcopy(spectra.bottom)

    if spectra.is_weighted
        spectra_top ./= reshape(energy_bins, 1, 1, :)
        spectra_bottom ./= reshape(energy_bins, 1, 1, :)
    end

    mean_weights_top = dropdims(mean(spectra_top, dims = (1, 2)), dims = (1, 2))
    mean_weights_bottom = dropdims(mean(spectra_bottom, dims = (1, 2)), dims = (1, 2))

    denominator_top = sum(mean_weights_top .* energy_bins)
    denominator_bottom = sum(mean_weights_bottom .* energy_bins)

    numerator_top = sum(linear_attenuation_coeff(material, energies) .* mean_weights_top .* energy_bins)
    numerator_bottom = sum(linear_attenuation_coeff(material, energies) .* mean_weights_bottom .* energy_bins)

    μ_eff_top = numerator_top / denominator_top
    μ_eff_bottom = numerator_bottom / denominator_bottom

    return μ(material.name, μ_eff_top, μ_eff_bottom)
end

struct S{T<:AbstractArray}
           top::T
           bottom::T
end

S1 = S(rand(10,10), rand(10,10))

function fun(S1::S)
    s_top = S1.top
    s_bottom = S1.bottom

    s_top .*= 100
    s_bottom .*= 100
    
    return s_top .+ s_bottom
end

fun(S1)


"""
    callback(state, cost)

Logging callback used during optimisation routines.
"""
function callback(state, cost)
    println("Current cost: ", cost)
    return false
end

"""
    fit_circle_on_zoomed(img, zoom_range)

Fit a circle to the edge map inside `zoom_range` and convert the result to full
image coordinates.
"""
function fit_circle_on_zoomed(img, zoom_range)
    row_range, col_range = zoom_range
    sub_img = img[row_range, col_range]
    edge_indices = findall(sub_img)
    if isempty(edge_indices)
        error("No edges found in the zoomed region.")
    end
    x = [i[1] for i in edge_indices]
    y = [i[2] for i in edge_indices]
    fit_result = fit(CircleFit.Circle, x, y)
    center_zoom = coef(fit_result)[1:2]
    radius = coef(fit_result)[3]
    center_full =
        (center_zoom[1] + first(row_range) - 1, center_zoom[2] + first(col_range) - 1)
    return center_full, radius, fit_result
end


"""
    compute_contrast_std(Is, Ib, std_Is, std_Ib)

Computes the standard deviation for the fractional contrast formula C = (Is - Ib) / Ib.
"""
function compute_contrast_std(Is::Real, Ib::Real, std_Is::Real, std_Ib::Real)
    if Ib == 0
        throw(ArgumentError("Background intensity (Ib) cannot be zero."))
    end
    
    # Apply the error propagation formula
    variance_term = std_Is^2 + (Is / Ib)^2 * std_Ib^2
    std_C = (1.0 / abs(Ib)) * sqrt(variance_term)
    
    return std_C
end

"""
    compute_contrast_stats(Is, Ib, std_Is, std_Ib)

Returns both the fractional contrast and its standard deviation as a NamedTuple.
"""
function compute_contrast_stats(Is::Real, Ib::Real, std_Is::Real, std_Ib::Real)
    C = (Is - Ib) / Ib
    std_C = compute_contrast_std(Is, Ib, std_Is, std_Ib)
    
    return (contrast = C, std = std_C)
end