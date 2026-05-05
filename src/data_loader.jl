"""
    log_norm(top_image, bottom_image)

Apply a numerically safe negative logarithm to paired images.
Zero-valued pixels are clamped to the smallest positive value before taking
the logarithm to avoid infinities.
"""
function log_norm(top_image::Matrix{Float64}, bottom_image::Matrix{Float64})
    if all(top_image .== 0.0) || all(bottom_image .== 0.0)
        error("Images contain only zeros; cannot apply log normalization.")
    end

    top_image[top_image.==0.0] .= minimum(top_image[top_image.>0.0])
    bottom_image[bottom_image.==0.0] .= minimum(bottom_image[bottom_image.>0.0])
    return -log.(top_image), -log.(bottom_image)
end

using HDF5

"""
    load_png_images(folder_path, image_name)

Read paired PNG images from `TopLayer` and `BottomLayer` subfolders and wrap
them into a `DLI` container. Pixel values are returned as `Gray{Float64}` and
are assumed to be pre-normalized.
"""
function load_png_images(folder_path::String, image_name::String)
    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    top_image = load(joinpath(top_folder, image_name))
    bottom_image = load(joinpath(bottom_folder, image_name))

    top_image = Gray{Float64}.(top_image)
    bottom_image = Gray{Float64}.(bottom_image)

    return DLI(top_image, bottom_image)
end

"""
    load_jld2_images(folder_path, image_name; log_normalization=true)

Load dual-layer images stored as JLD2 files named `TopLayer` and `BottomLayer`.
Optionally apply logarithmic normalization via [`log_norm`](@ref).
"""
function load_jld2_images(folder_path::String, image_name::String, log_normalization::Bool = true)::DLI

    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    top_file = load(joinpath(top_folder, image_name))
    bottom_file = load(joinpath(bottom_folder, image_name))
    @assert haskey(top_file, "TopLayer") "Top JLD2 file missing key 'TopLayer'"
    @assert haskey(bottom_file, "BottomLayer") "Bottom JLD2 file missing key 'BottomLayer'"
    top_image::Matrix{Float64} = Float64.(top_file["TopLayer"])
    bottom_image::Matrix{Float64} = Float64.(bottom_file["BottomLayer"])

    size(top_image) == size(bottom_image) || error("The images are not the same size")

    top_image, bottom_image =
        log_normalization ? log_norm(top_image, bottom_image) : (top_image, bottom_image)

    # Check for Inf values
    if any(isinf.(top_image)) || any(isinf.(bottom_image))
        error("Inf values detected in dli_images.")
    end

    # Check for NaN values
    if any(isnan.(top_image)) || any(isnan.(bottom_image))
        error("NaN values detected in dli_images.")
    end

    return DLI(top_image, bottom_image)
end

"""
    load_jld2 (folder_path, image_name; log_normalization=true)

Load dual-layer images stored as JLD2 files named `TopLayer` and `BottomLayer`, along with metadata
Optionally apply logarithmic normalization via [`log_norm`](@ref).
"""
function load_jld2(folder_path::String, image_name::String, log_normalization = true)

    # Load images
    dli_images = load_jld2_images(folder_path, image_name, log_normalization)

    # Load metadata from .jld2 (HDF5) files and convert to Dict{String,Any}
    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    top_file = load(joinpath(top_folder, image_name))
    bottom_file = load(joinpath(bottom_folder, image_name))

    metadata_top = Dict{String,Any}()

    for key in keys(top_file)
        if startswith(key, "metadata/")
            key_name = key[10:end]  # Remove "metadata/" prefix
            metadata_top[key_name] = top_file[key]
        end
    end

    metadata_bottom = Dict{String,Any}()
    for key in keys(bottom_file)
        if startswith(key, "metadata/")
            key_name = key[10:end]  # Remove "metadata/" prefix
            metadata_bottom[key_name] = bottom_file[key]
        end
    end

    return dli_images, metadata_top, metadata_bottom

end

"""
    load_images(folder_path, image_name, log_normalization)

Dispatch helper that loads either PNG or JLD2 dual-layer images based on the
file extension, applying logarithmic normalization when requested.
"""
function load_images(folder_path::String, image_name::String, log_normalization::Bool)
    @assert (endswith(image_name, ".png") || endswith(image_name, ".jld2")) "image_name must end with .png or .jld2"
    if endswith(image_name, ".png")
        return load_png_images(folder_path, image_name)
    elseif endswith(image_name, ".jld2")
        return load_jld2_images(folder_path, image_name, log_normalization)
    end
    error("Unsupported image extension for $(image_name). Expected .png or .jld2.")
end


"""
    load_spectra(folder_path, spectra_name)

Load paired spectral data arrays from JLD2 files into a `Spectra` container.
"""
function load_spectra(folder_path::String, spectra_name::String; is_weighted::Bool = true)::Spectra
    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    top_file = load(joinpath(top_folder, spectra_name))
    bottom_file = load(joinpath(bottom_folder, spectra_name))
    @assert haskey(top_file, "TopLayer") "Spectra file missing 'TopLayer' key"
    @assert haskey(bottom_file, "BottomLayer") "Spectra file missing 'BottomLayer' key"
    top_spectra = top_file["TopLayer"]
    bottom_spectra = bottom_file["BottomLayer"]

    return Spectra(top_spectra, bottom_spectra, is_weighted)
end
