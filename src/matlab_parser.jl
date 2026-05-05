"""
Helper utilities for converting paired TopLayer/BottomLayer .mat files into
jld2/png formats expected by the DLI pipeline.
"""

"""
mat_to_JLD2(folder_path::String, file_name::String)

Read paired .mat files from `TopLayer` and `BottomLayer` (key "data") and write
each time-frame to separate JLD2 files named `frame_i.jld2` in their respective
subfolders. Each saved file contains a dictionary with the key `TopLayer` or
`BottomLayer` mapping to the 2D image for that frame.

Parameters
- folder_path : path to the parent folder containing `TopLayer` and `BottomLayer`.
- file_name   : .mat filename to read (same name present in both subfolders).

"""
function mat_to_JLD2(folder_path::String, file_name::String)
    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    @assert isfile(joinpath(top_folder, file_name)) "File $file_name not found in $top_folder"
    @assert isfile(joinpath(bottom_folder, file_name)) "File $file_name not found in $bottom_folder"

    mat_file_top = matread(joinpath(top_folder, file_name))
    mat_file_bottom = matread(joinpath(bottom_folder, file_name))

    @assert haskey(mat_file_top, "data") "The .mat file $file_name has no field 'data'"
    @assert haskey(mat_file_bottom, "data") "The .mat file $file_name has no field 'data'"

    top_data = mat_file_top["data"]
    bottom_data = mat_file_bottom["data"]

    for i in axes(top_data, 3)
        top_img = top_data[:, :, i]
        bottom_img = bottom_data[:, :, i]

        save(joinpath(top_folder, "frame_$i.jld2"), Dict("TopLayer" => top_img))
        save(joinpath(bottom_folder, "frame_$i.jld2"), Dict("BottomLayer" => bottom_img))

    end
end

"""
mat_to_png(folder_path::String, file_name::String)

Convert paired .mat files into PNG images for visualization. The function reads
the `data` variable from TopLayer/BottomLayer, applies a negative logarithm to
pixel intensities, performs a linear histogram stretch to normalize into [0,1],
and saves each frame as `frame_i.png` under the respective subfolder.

Parameters
- folder_path : path containing `TopLayer` and `BottomLayer` subfolders.
- file_name   : .mat filename to read (same name present in both subfolders).

Warning
- This function modifies intensities (applies -log) and normalizes images; use
    only for visualization or when this preprocessing is desired.
"""
function mat_to_png(folder_path::String, file_name::String)
    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    @assert isfile(joinpath(top_folder, file_name)) "File $file_name not found in $top_folder"
    @assert isfile(joinpath(bottom_folder, file_name)) "File $file_name not found in $bottom_folder"

    mat_file_top = matread(joinpath(top_folder, file_name))
    mat_file_bottom = matread(joinpath(bottom_folder, file_name))

    @assert haskey(mat_file_top, "data") "The .mat file $file_name has no field 'data'"
    @assert haskey(mat_file_bottom, "data") "The .mat file $file_name has no field 'data'"

    top_data = mat_file_top["data"]
    bottom_data = mat_file_bottom["data"]

    for i in axes(top_data, 3)
        top_img = top_data[:, :, i]
        bottom_img = bottom_data[:, :, i]

        top_img = -log.(top_img)
        bottom_img = -log.(bottom_img)

        images_adjust_histogram!(top_img, LinearStretching())
        images_adjust_histogram!(bottom_img, LinearStretching())

        save(joinpath(folder_path, "TopLayer", "frame_$i.png"), top_img)
        save(joinpath(folder_path, "BottomLayer", "frame_$i.png"), bottom_img)
    end
end

"""
mat_spectra_to_JLD2(folder_path::String, file_name::String)

Load detected spectral matrices from paired .mat files (expected variable `S`)
and save them as `detectedSpectra.jld2` under each layer's subfolder. The saved
file contains a dictionary mapping `TopLayer`/`BottomLayer` to the spectrum array.

Parameters
- folder_path : path containing `TopLayer` and `BottomLayer`.
- file_name   : .mat filename (expected to contain variable `S`).
"""
function mat_spectra_to_JLD2(folder_path::String, file_name::String)
    top_folder = joinpath(folder_path, "TopLayer")
    bottom_folder = joinpath(folder_path, "BottomLayer")

    @assert isfile(joinpath(top_folder, file_name)) "File $file_name not found in $top_folder"
    @assert isfile(joinpath(bottom_folder, file_name)) "File $file_name not found in $bottom_folder"

    mat_file_top = matread(joinpath(top_folder, file_name))
    mat_file_bottom = matread(joinpath(bottom_folder, file_name))

    @assert haskey(mat_file_top, "S") "The .mat file $file_name has no field 'S'"
    @assert haskey(mat_file_bottom, "S") "The .mat file $file_name has no field 'S'"

    top_data = mat_file_top["S"]
    bottom_data = mat_file_bottom["S"]

    save(joinpath(top_folder, "detectedSpectra.jld2"), Dict("TopLayer" => top_data))
    save(
        joinpath(bottom_folder, "detectedSpectra.jld2"),
        Dict("BottomLayer" => bottom_data),
    )
end
