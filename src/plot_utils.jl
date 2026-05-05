"""
    enable_plot_theme!()

Apply the default CairoMakie theme used by this package. Calling this function
is optional so loading the module does not mutate global Makie state.
"""
function enable_plot_theme!()
    #set_theme!(theme_latexfonts())
    Axis_theme = Theme(
    Axis = (
        xminorticksvisible = true,
        xminorgridvisible = true,
        xminortickalign=1.0,
        yminortickalign=1.0,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5)
    )
    )
    package_theme = merge(Axis_theme, theme_latexfonts())
    set_theme!(package_theme)
    update_theme!(fontsize=20)
    return nothing
end

# Use CairoMakie.scatter! directly to avoid name conflicts with CUSPARSE.scatter!

function plot_mean_spectra(spectra::Spectra)
    energy_bins = collect(axes(spectra.top, 3))
    spectra_top = deepcopy(spectra.top)
    spectra_bottom = deepcopy(spectra.bottom)

    if spectra.is_weighted # the spectrum is I(E) x E
        spectra_top ./= reshape(energy_bins, 1, 1, :)
        spectra_bottom ./= reshape(energy_bins, 1, 1, :)
    end

    mean_weights_top = dropdims(mean(spectra_top, dims = (1, 2)), dims = (1, 2))
    mean_weights_bottom = dropdims(mean(spectra_bottom, dims = (1, 2)), dims = (1, 2))

    mean_energy_top = sum(mean_weights_top .* energy_bins) / sum(mean_weights_top)
    mean_energy_bottom = sum(mean_weights_bottom .* energy_bins) / sum(mean_weights_bottom) 

    fig = Figure()
    if spectra.is_weighted
        title = "Spectra (correct one)"
        ylabel = "I(E)"
    else
        title = "Spectra (wrong one)"
        ylabel = "I(E) x E"
    end
    ax = Axis(fig[1, 1], xlabel = "Energy (keV)", ylabel = ylabel, title = title)

    lines!(ax, energy_bins, mean_weights_top, label = "Top")
    lines!(ax, energy_bins, mean_weights_bottom, label = "Bottom")
    vlines!(
        ax,
        [mean_energy_top],
        label = ["E high = $(round(mean_energy_top, digits = 1)) keV"],
        color = :black,
        linestyle = :dash,
    )
    vlines!(
        ax,
        [mean_energy_bottom],
        label = ["E low = $(round(mean_energy_bottom, digits = 1)) keV"],
        color = :black,
        linestyle = :dash,
    )
    axislegend(ax)

    return fig
end


function plot_spectra(spectra::Spectra, row::Integer, col::Integer)

    spectra_top = spectra.top[row, col, :]
    spectra_bottom = spectra.bottom[row, col, :]
    energy_bins = collect(axes(spectra_top, 1))

    if spectra.is_weighted # the spectrum is I(E) x E
        spectra_top ./= energy_bins
        spectra_bottom ./= energy_bins
    end

    mean_energy_top = sum(spectra_top .* energy_bins) / sum(spectra_top)
    mean_energy_bottom = sum(spectra_bottom .* energy_bins) / sum(spectra_bottom)

    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Energy (keV)",
        ylabel = "Intensity",
        title = "Pixel ($row, $col)",
    )

    lines!(ax, energy_bins, spectra_top, label = "Top")
    lines!(ax, energy_bins, spectra_bottom, label = "Bottom")
    vlines!(
        ax,
        [mean_energy_top],
        label = ["E high = $(round(mean_energy_top, digits = 1)) keV"],
        color = :black,
        linestyle = :dash,
    )
    vlines!(
        ax,
        [mean_energy_bottom],
        label = ["E low = $(round(mean_energy_bottom, digits = 1)) keV"],
        color = :black,
        linestyle = :dash,
    )
    axislegend(ax)

    return fig
end


"""
    plot_attenuation(E_range, material_1, material_2; E_low=nothing, E_high=nothing)

Plot mass attenuation coefficient curves for two materials over the supplied
energy range and optionally highlight reference energies.
"""
function plot_attenuation(E_range, material_1, material_2; E_low = nothing, E_high = nothing)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Mass Attenuation Coefficients",
        xlabel = "Energy (keV)",
        xtickformat = "{:.2f}",
        ylabel = "MAC (cm²/g)",
        xscale = log10,
        yscale = log10,
    )
    lines!(ax, val.(E_range), val.(μᵨ(material_1, E_range)), label = material_1.name)
    lines!(ax, val.(E_range), val.(μᵨ(material_2, E_range)), label = material_2.name)

    if !isnothing(E_low)
        E_low = val(E_low) # Extract numeric value if E_low is a Quantity
        vlines!(
            [E_low],
            label = "E low = $(round(E_low, digits = 1)) keV",
            color = :black,
            linestyle = :dash,
        )
    end
    if !isnothing(E_high)
        E_high = val(E_high) # Extract numeric value if E_high is a Quantity
        vlines!(
            [E_high],
            label = "E high = $(round(E_high, digits = 1)) keV",
            color = :black,
            linestyle = :dash,
        )
    end
    axislegend(ax) # Add a legend to the axis
    return fig
end

"""
    plot_attenuation(E_range, material_1; E_low=nothing, E_high=nothing)

Plot the mass attenuation coefficient curve for a material and optionally mark
reference energies.
"""
function plot_attenuation(E_range, material_1; E_low = nothing, E_high = nothing)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Mass Attenuation Coefficients",
        xlabel = "Energy (keV)",
        xtickformat = "{:.2f}",
        ylabel = "MAC (cm²/g)",
        xscale = log10,
        yscale = log10,
    )
    lines!(ax, val.(E_range), val.(μᵨ(material_1, E_range)), label = material_1.name)
    if !isnothing(E_high)
        E_high = val(E_high) # Extract numeric value if E_high is a Quantity
        vlines!(
            [E_high],
            label = "E high = $(round(E_high, digits = 1)) keV",
            color = :black,
            linestyle = :dash,
        )
    end
    if !isnothing(E_low)
        E_low = val(E_low) # Extract numeric value if E_low is a Quantity
        vlines!(
            [E_low],
            label = "E low = $(round(E_low, digits = 1)) keV",
            color = :black,
            linestyle = :dash,
        )
    end
    axislegend(ax) # Add a legend to the axis
    return fig
end

"""
    plot_energy_map(dli_images)

Scatter the joint top/bottom intensities to visualise the energy relationship
between detector layers.
"""
function plot_energy_map(dli_images::DLI)
    top_flat = vec(dli_images.top)
    bottom_flat = vec(dli_images.bottom)

    # Subsample for performance
    indices = rand(1:length(top_flat), min(10000, length(top_flat)))

    fig = Figure(size = (600, 400))
    ax = Axis(
        fig[1, 1],
        title = "Energy Map",
        xlabel = "Top Image Intensity",
        ylabel = "Bottom Image Intensity",
    )

    CairoMakie.scatter!(
        ax,
        top_flat[indices],
        bottom_flat[indices],
        markersize = 1,
        strokewidth = 0,
    )
    return fig
end

"""
    plot_material_map(material_images)

Scatter the two material images against each other to inspect correlations in
the decomposition.
"""
function plot_material_map(material_images::MI)
    mat1_flat = vec(material_images.mat1)
    mat2_flat = vec(material_images.mat2)

    indices = rand(1:length(mat1_flat), min(10000, length(mat1_flat)))

    fig = Figure(size = (600, 400))
    ax = Axis(
        fig[1, 1],
        title = "Material Map",
        xlabel = "$(material_images.μ₁.name) intensity",
        ylabel = "$(material_images.μ₂.name) intensity",
    )

    CairoMakie.scatter!(
        ax,
        mat1_flat[indices],
        mat2_flat[indices],
        markersize = 1,
        strokewidth = 0,
    )
    return fig
end

"""
    plot_histogram(material_images)

Plot histograms of the two material images using probability density scaling.
"""
function plot_histogram(material_images::MI)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Histograms")

    hist!(
        ax,
        vec(material_images.mat1),
        bins = 256,
        normalization = :pdf,
        label = material_images.μ₁.name,
        color = (:blue, 0.5),
    )
    hist!(
        ax,
        vec(material_images.mat2),
        bins = 256,
        normalization = :pdf,
        label = material_images.μ₂.name,
        color = (:red, 0.5),
    )

    axislegend(ax)
    return fig
end

"""
    plot_histogram(dli_images)

Plot histograms of the top and bottom detector images.
"""
function plot_histogram(dli_images::DLI)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Histograms")

    hist!(
        ax,
        vec(dli_images.top),
        bins = 256,
        normalization = :pdf,
        label = "Top Image",
        color = (:blue, 0.5),
    )
    hist!(
        ax,
        vec(dli_images.bottom),
        bins = 256,
        normalization = :pdf,
        label = "Bottom Image",
        color = (:red, 0.5),
    )

    axislegend(ax)
    return fig
end

"""
    plot_heatmap(dli_images)

Render heatmaps for the top and bottom detector layers with shared colour
scaling.
"""
function plot_heatmap(dli_images::DLI)
    fig = Figure(size = (800, 420))

    ax1 = Axis(fig[1, 1], title = "Top Image", aspect = DataAspect())
    ax2 = Axis(fig[1, 2], title = "Bottom Image", aspect = DataAspect())

    # 2. Apply the unified `colorrange` to both heatmaps.
    heatmap!(ax1, dli_images.top, colormap = :grays)
    hm2 = heatmap!(ax2, dli_images.bottom, colormap = :grays)

    linkaxes!(ax1, ax2)

    # 3. The colorbar is now accurate because both plots share the same scale.
    Colorbar(fig[1, 3], hm2, label = "log(Intensity)")

    return fig
end

"""
    plot_heatmap(dli_images, crange)

Render heatmaps for the dual detector images using a fixed colour range.
"""
function plot_heatmap(dli_images::DLI, crange)
    fig = Figure(size = (1100, 420))

    ax1 = Axis(fig[1, 1], title = "Top Image", aspect = DataAspect())
    ax2 = Axis(fig[1, 2], title = "Bottom Image", aspect = DataAspect())

    # 2. Apply the unified `colorrange` to both heatmaps.
    heatmap!(ax1, dli_images.top, colormap = :grays, colorrange = crange)
    hm2 = heatmap!(ax2, dli_images.bottom, colormap = :grays, colorrange = crange)

    linkaxes!(ax1, ax2)

    # 3. The colorbar is now accurate because both plots share the same scale.
    Colorbar(fig[1, 3], hm2, label = "log(Intensity)")

    return fig
end

"""
    plot_heatmap(material_images)

Plot heatmaps for both material images with individual colourbars.
"""
function plot_heatmap(material_images::MI)
    fig = Figure(size = (800, 420))

    ax1 = Axis(fig[1, 1], title = material_images.μ₁.name, aspect = DataAspect())
    hm1 = heatmap!(ax1, material_images.mat1, colormap = :grays)
    Colorbar(fig[1, 2], hm1, label = "log(Intensity)")

    ax2 = Axis(fig[1, 3], title = material_images.μ₂.name, aspect = DataAspect())
    hm2 = heatmap!(ax2, material_images.mat2, colormap = :grays)
    Colorbar(fig[1, 4], hm2, label = "log(Intensity)")
    linkaxes!(ax1, ax2)

    # 3. The colorbar is now accurate because both plots share the same scale.

    return fig
end

"""
    plot_heatmap(material_images, crange)

Plot material heatmaps with a shared colour range.
"""
function plot_heatmap(material_images::MI, crange)
    fig = Figure(size = (800, 420))

    ax1 = Axis(fig[1, 1], title = material_images.μ₁.name, aspect = DataAspect())
    hm1 = heatmap!(ax1, material_images.mat1, colormap = :grays, colorrange = crange)
    Colorbar(fig[1, 2], hm1, label = "log(Intensity)")

    ax2 = Axis(fig[1, 3], title = material_images.μ₂.name, aspect = DataAspect())
    hm2 = heatmap!(ax2, material_images.mat2, colormap = :grays, colorrange = crange)
    Colorbar(fig[1, 4], hm2, label = "log(Intensity)")
    linkaxes!(ax1, ax2)


    return fig
end


"""
    plot_heatmap(dli_images, material_images)

Display the dual-layer images alongside material reconstructions in a grid of
heatmaps.
"""
function plot_heatmap(dli_images::DLI, material_images::MI)
    # Adjust figure size to comfortably fit the plots and their colorbars
    fig = Figure(size = (1000, 800))

    # --- Top Left Plot ---
    ax11 = Axis(fig[1, 1], title = "Top Image", aspect = DataAspect())
    hm11 = heatmap!(ax11, dli_images.top, colormap = :grays)
    Colorbar(fig[1, 2], hm11) # Colorbar for the top-left plot

    # --- Top Right Plot ---
    ax12 = Axis(fig[1, 3], title = "Bottom Image", aspect = DataAspect())
    hm12 = heatmap!(ax12, dli_images.bottom, colormap = :grays)
    Colorbar(fig[1, 4], hm12) # Colorbar for the top-right plot

    # --- Bottom Left Plot ---
    ax21 = Axis(fig[2, 1], title = material_images.μ₁.name, aspect = DataAspect())
    hm21 = heatmap!(ax21, material_images.mat1, colormap = :grays)
    Colorbar(fig[2, 2], hm21) # Colorbar for the bottom-left plot

    # --- Bottom Right Plot ---
    ax22 = Axis(fig[2, 3], title = material_images.μ₂.name, aspect = DataAspect())
    hm22 = heatmap!(ax22, material_images.mat2, colormap = :grays)
    Colorbar(fig[2, 4], hm22) # Colorbar for the bottom-right plot

    # Link axes if you want them to zoom and pan together
    linkaxes!(ax11, ax12, ax21, ax22)

    return fig
end

"""
    plot_heatmap(image; crange=nothing, background_mask=nothing, signal_mask=nothing)

Heatmap helper for single images with optional overlay masks.
"""
function plot_heatmap(
    image::Matrix;
    crange = nothing,
    background_mask = nothing,
    signal_mask = nothing,
)
    if isnothing(crange)
        # If no range is given, find the global min/max across BOTH images.
        crange = extrema(vec(image))
    end

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = DataAspect())

    # 1. Capture the plot object returned by heatmap!
    hm = heatmap!(ax, image, colormap = :grays, colorrange = crange)

    !isnothing(background_mask) &&
        poly!(ax, background_mask, color = (:green, 0.5), label = "Signal Mask")
    !isnothing(signal_mask) &&
        poly!(ax, signal_mask, color = (:red, 0.5), label = "Background Mask")

    # 2. Pass the plot object `hm` to the Colorbar
    Colorbar(fig[1, 2], hm, label = "log(Intensity)")

    return fig
end

"""
    plot_materials_grid(materials, row_names; crange_mat1, crange_mat2, crange_vnbc, figsize=(1000, 800), material_names=nothing, zoom_region=nothing)

Arrange multiple material reconstructions in a grid for side-by-side
comparison.
"""
function plot_materials_grid(
    materials_1::Vector{MI{Matrix{Float64}}},
    materials_2::Vector{MI{Matrix{Float64}}},
    row_names::Vector{String};
    crange_mat1,
    crange_mat2,
    crange_vnbc,
    figsize = (1000, 800),
    material_names = nothing,
    zoom_region = nothing,
    zoom_region_mask = nothing,
    background_mask = nothing,
)

    n1 = length(materials_1)
    n2 = length(materials_2)
    @assert n1 == length(row_names) "materials and row_names must have same length"
    @assert n2 == length(row_names) "materials and row_names must have same length"
    n1 == 0 && error("materials 1 list is empty")
    n2 == 0 && error("materials 2 list is empty")

    @assert n1 == n2 "The 2 materials list have not the same size"
    n = n1 

    # Determine default material column names
    mat_name_1 = isnothing(material_names) ? materials_1[1].μ₁.name : material_names[1]
    mat_name_2 = isnothing(material_names) ? materials_2[1].μ₂.name : material_names[2]

    # Create figure. If figsize is nothing, omit the size keyword to avoid passing `Nothing` to Figure.
    fig =
        isnothing(figsize) ? Figure(backgroundcolor = :black) :
        Figure(backgroundcolor = :black, size = figsize)

    # Column titles on top (row 0)
    Label(fig[0, 1], mat_name_1, color = :white, halign = :center, tellwidth = false)
    Label(fig[0, 2], mat_name_2, color = :white, halign = :center, tellwidth = false)
    #Label(fig[0, 3], "VNCB", color = :white, halign = :center, tellwidth = false)

    # Rows with images and optional colorbars
    for i = 1:n
        Label(
            fig[i, 0],
            row_names[i],
            color = :white,
            tellheight = false,
            halign = :center, #right
            rotation = pi / 2,
        )
        mat1 = materials_1[i].mat1
        mat2 = materials_2[i].mat2

        if !isnothing(zoom_region)
            zoom_x, zoom_y = zoom_region
            mat1 = @view materials_1[i].mat1[zoom_x, zoom_y]
            mat2 = @view materials_2[i].mat2[zoom_x, zoom_y]
        end

        ax1 = Axis(fig[i, 1], aspect = DataAspect())
        hm1 = heatmap!(ax1, mat1, colormap = :grays, colorrange = crange_mat1)
        hidedecorations!(ax1)
        hidespines!(ax1)

        ax2 = Axis(fig[i, 2], aspect = DataAspect())
        hm2 = heatmap!(ax2, mat2, colormap = :grays, colorrange = crange_mat2)
        hidedecorations!(ax2)
        hidespines!(ax2)

#=         ax3 = Axis(fig[i, 3], aspect = DataAspect())
        hm3 = heatmap!(ax3, mat1 .+ mat2, colormap = :grays, colorrange = crange_vnbc)
        hidedecorations!(ax3)
        hidespines!(ax3) =#

        if i==1 && !isnothing(zoom_region_mask)
            poly!(ax1, zoom_region_mask, color = :transparent, 
                        strokewidth = 2,
                        label = "Zoom Region")
        end

        if i==1 && !isnothing(background_mask)
            poly!(ax1, background_mask, color = (:green, 0.5),
                        label = "Background Mask")
        end

    end

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 5)
    return fig
end


function plot_heatmap(image_vec::Vector, nrows, ncols, mat = 0)
    N = nrows * ncols
    image1 = reshape(@view(image_vec[1:N]), nrows, ncols)
    image2 = reshape(@view(image_vec[N+1:end]), nrows, ncols)

    if mat == 0
        return plot_heatmap(DLI(image1, image2))
    elseif mat == 1
        return plot_heatmap(image1)
    elseif mat == 2
        return plot_heatmap(image2)
    else
        error("mat must be 0, 1, or 2")
    end
end

"""
    plot_esf_lsf(material_image, circle, esf, lsf; crange=nothing, ref_mat=:mat1)

Plot a material image with circle overlay and accompanying ESF/LSF profiles.
"""
function plot_esf_lsf(
    material_image::MI,
    circle::Circ,
    esf::ESF,
    lsf::LSF;
    crange = nothing,
    ref_mat = :mat1,
    background_mask = nothing,
)
    fig = Figure(size = (1200, 400), fontsize=18)
    image = getfield(material_image, ref_mat)
    # --- Plot 1: Image with Circle Overlay ---

    material_name = ref_mat == :mat1 ? material_image.μ₁.name : material_image.μ₂.name
    ax1 = Axis(fig[1, 1], title = "$material_name Image", aspect = DataAspect())

    if isnothing(crange)
        # If no range is given, find the global min/max across BOTH images.
        crange = extrema(vec(image))
    end

    heatmap!(ax1, image, colormap = :grays, colorrange = crange)
    # Draw the circle boundary
    arc!(
        Point2f(circle.center),
        circle.radius,
        0,
        2π,
        color = :red,
        linewidth = 2,
        label = "Circle Edge",
    )
    !isnothing(background_mask) &&
        poly!(ax1, background_mask, color = (:green, 0.5), label = "Signal Mask")
    hidedecorations!(ax1) # Hides axis numbers and ticks

    # --- Plot 2: Edge Spread Function (ESF) ---
    p_esf = fit_esf(esf) # the sigmoid fit parameters

    ax2 = Axis(
        fig[1, 2],
        title = "Edge Spread Function (ESF)",
        xlabel = "Distance from radius (mm)",
        ylabel = "Average Intensity",
    )
    plot!(ax2, esf, p_esf = p_esf)

    # --- Plot 3: Line Spread Function (LSF) ---
    ax3 = Axis(
        fig[1, 3],
        title = "Line Spread Function (LSF)",
        xlabel = "Distance from radius (mm)",
        ylabel = "LSF",
    )
    #plot!(ax3, lsf; prominences=true, widths=true)
    plot!(ax3, lsf, p_esf = p_esf)

    return fig
end

"""
    plot_esf(esf_distances, esf)

Plots the Edge Spread Function (ESF).
"""
function plot(esf::ESF; p_esf = [])
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title = "Edge Spread Function (ESF)",
        xlabel = "Distance from Radius (mm)",
        ylabel = "Average Intensity",
    )
    plot!(ax, esf, p_esf = p_esf)
    return fig
end

"""
    plot_esf(esf_distances, esf)

Plots the Edge Spread Function (ESF).
"""
function plot!(ax::Axis, esf::ESF; p_esf = [])
    lines!(ax, esf.x, esf.y, linewidth = 2, color = :black, label = "ESF")

    if !isempty(p_esf)
        # Plot the fitted model
        lines!(
            ax,
            esf.x,
            sigmoid(esf.x, p_esf);
            linewidth = 2,
            linestyle = :dash,
            label = "ESF fit",
        )
    end
    axislegend(ax, position = :lt)
end


"""
    plot(lsf::LSF; fit_function::Union{Nothing, Function} = nothing)

Plot the Line Spread Function (LSF) data using a Makie figure.

# Arguments
- `lsf::LSF`: The LSF data object to be plotted.
- `p_esf`: Parameters of the fitted sigmoid ESF, used to compute the LSF fit if needed.
- `p_lsf`: Parameters of the gaussian fitted LSF function. No longer used
# Description
Creates a plot of the provided LSF data. If a `fit_function` is provided, it will be used to fit and display an additional curve on the plot.
Contrarily to plot(ax::Axis, lsf::LSF; prominences::Bool=true, widths::Bool=true) function, this one use the fit to compute the FWHM
# Returns
- `fig`: The generated Makie `Figure` object containing the plot.

"""
function plot(lsf::LSF; p_esf = [], p_lsf = [])

    # Create the plot
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1],
        title = "Line Spread Function",
        xlabel = "Distance from radius (pixel)",
        ylabel = "LSF",
    )

    # Use the new function to plot the data and analyses
    plot!(ax, lsf, p_esf = p_esf, p_lsf = p_lsf)

    # Display the figure
    fig

end

function plot!(ax::Axis, lsf::LSF; p_esf = [], p_lsf = [])
    # Plot the raw signal
    lines!(ax, lsf.x, lsf.y; linewidth = 2, label = "Numeric LSF", color = :black)

    if !isempty(p_esf)
        # Compute LSF fit from ESF parameters
        @assert length(p_esf) == 4 "p_esf must have 4 parameters from the esf fit with a sigmoid"
        fwhm = fwhm_prime_sigmoid(p_esf)
        fwhm_rounded = round(fwhm, digits = 2)
        lines!(
            ax,
            lsf.x,
            prime_sigmoid(lsf.x, p_esf),
            linewidth = 2,
            linestyle = :dash,
            #label = "Analytic LSF \n FWHM = \n       $fwhm_rounded mm",
            label = "Analytic LSF",
        )
    end

    if !isempty(p_lsf)
        # Plot the fitted model
        @assert length(p_lsf) == 3 "p_lsf must have 3 parameters from the lsf fit with a gaussian"
        fwhm = fwhm_gaussian(p_lsf)
        fwhm_rounded = round(fwhm, digits = 2)
        lines!(
            ax,
            lsf.x,
            gaussian(lsf.x, p_lsf);
            linewidth = 2,
            linestyle = :dash,
            label = "Analytic LSF \n (FWHM = $fwhm_rounded) mm",
        )
    end


    axislegend(ax, position = :rt) # position right-top
end



function plot_heatmap_similarity(
    img::AbstractMatrix,
    sim_mat::SparseMatrixCSC,
    pixel_idx::Tuple{Int64,Int64};
    use_second_half::Bool = false
)

    nrows, ncols = size(img)
    N = nrows * ncols
    nrows_sim, ncols_sim = size(sim_mat)
    pixel_idx = LinearIndices((nrows, ncols))[pixel_idx[1], pixel_idx[2]]

    @assert nrows_sim == ncols_sim "sim_mat must be square"
    @assert pixel_idx >= 1 && pixel_idx <= N "pixel_idx out of range"

    row_index = if nrows_sim == N
        pixel_idx
    elseif nrows_sim == 2N
        pixel_idx + (use_second_half ? N : 0)
    else
        throw(ArgumentError("sim_mat size not compatible"))
    end

    sims = vec(Array(sim_mat[row_index, :]))
    if length(sims) == 2N
        sims = use_second_half ? sims[N+1:end] : sims[1:N]
    end

    sim_map = reshape(sims, nrows, ncols)

    # normalize similarity to [0,1]
    sim_vals = sim_map .- minimum(sim_map)
    maxv = maximum(sim_vals)
    if maxv > 0
        sim_vals ./= maxv
    end

    # convert grayscale image to RGB directly (Makie has Gray/RGB types available)
    img = adjust_histogram(img, LinearStretching())
    img_rgb = RGB.(Gray.(img))

    # overlay red where similarity is high
    alpha = 0.5
    red = RGB(1, 0, 0)
    for i in axes(img, 1), j in axes(img, 2)
        a = sim_vals[i, j] * alpha
        if a > 0
            img_rgb[i, j] = (1 - a) * img_rgb[i, j] + a * red
        end
    end

    img_rgb
end


#Deprecated functions using Peaks.jl for peak analysis and FWHM calculation
# Now we use a gaussian fit to compute the FWHM
#= """
    plotpeaks(x, y, peaks; prominences=false, widths=false)

Plots peaks and optionally their prominences and widths on a CairoMakie.jl Axis.
"""
function plot(lsf::LSF; prominences::Bool=true, widths::Bool=true)

    # Create the plot
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title="Line Spread Function", xlabel="Distance from radius (pixel)", ylabel="LSF")

    # Use the new function to plot the data and analyses
    plot!(ax, lsf, prominences=prominences, widths=widths)

    # Add a legend
    axislegend(ax, position=:rt) # position right-top

    # Display the figure
    fig
end

"""
    plotpeaks!(ax::Axis, x, y, peaks; prominences=false, widths=false)

Plots in-place peaks and optionally their prominences and widths on a CairoMakie.jl Axis.
"""
function plot!(ax::Axis, lsf::LSF;
                    prominences::Bool=true, widths::Bool=true)

    peaks, _ = findmaximum(lsf.y)
    local maxima = true

    # Set plot attributes based on extrema type
    sgn = maxima ? -1 : +1
    ext_color = maxima ? :red : :green
    ext_label = maxima ? "maxima" : "minima" 

    xvals = lsf.x[peaks]
    yvals = lsf.y[peaks]

    # Plot the raw signal
    lines!(ax, lsf.x, lsf.y; linewidth=2, label="LSF", color=:black)

    # Calculate prominences if needed for either option
    local proms
    if prominences || widths
        _, proms = peakproms(peaks, lsf.y; strict=false)
    end

    # Plot prominences
    if prominences
        nans = fill(NaN, (length(peaks), 1))
        promlinesx = vec(vcat(reshape(repeat(xvals; inner=2), 2, length(peaks)), nans'))
        promlinesy = vec([yvals + sgn * proms yvals nans]')

        _, _, lower, upper = peakwidths(peaks, lsf.y, proms; strict=false, relheight=prevfloat(1.0))

        # Interpolate x-values for horizontal prominence lines
        lower_x = Peaks.interp(lsf.x, Peaks.drop_irrelevant_side.(lower, peaks, Ref(lsf.y), maxima))
        upper_x = Peaks.interp(lsf.x, Peaks.drop_irrelevant_side.(upper, peaks, Ref(lsf.y), maxima))
        promwidthlinesx = vec([lower_x upper_x nans]')
        promwidthlinesy = vec(vcat(reshape(repeat(yvals + sgn * proms; inner=2), 2, length(peaks)), nans'))

        # Plot vertical prominence lines 
        lines!(ax, promlinesx, promlinesy; label="Prominence", linewidth=1, color=:blue)

        # Plot horizontal prominence lines
        lines!(ax, promwidthlinesx, promwidthlinesy; linewidth=1, color=(:blue, 0.5), linestyle=:dash)
    end

    # Plot widths
    if widths
        nans = fill(NaN, (length(peaks), 1))
        # Calculate widths at half-prominence
        _, _, lower, upper = peakwidths(peaks, lsf.y, proms; strict=false)
        lower_x = Peaks.interp(lsf.x, lower)
        upper_x = Peaks.interp(lsf.x, upper)

        halfwidthlinesx = vec([lower_x upper_x nans]')
        halfwidthlinesy = vec(vcat(reshape(repeat(yvals + sgn * proms .* 0.5; inner=2), 2, length(peaks)), nans'))
        lsf_fwhm = round(fwhm(lsf.x, lsf.y), digits=2)

        # Plot width lines and points
        lines!(ax, halfwidthlinesx, halfwidthlinesy; linewidth=1, label="FWHM = $lsf_fwhm", color=:gray, linestyle=:dash)
        scatter!(ax, halfwidthlinesx, halfwidthlinesy; color=:gray, label=nothing)
    end

    # Plot extrema points
    scatter!(ax, xvals, yvals; label=ext_label, color=ext_color)
    axislegend(ax, position=:rt)
    return ax
end =#
