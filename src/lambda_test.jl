# This file defines the λTest struct and associated functions for testing and analyzing the effects of different regularization parameters (λ) on material decomposition results

"""
    λTest

A struct for testing and managing regularization parameters (λ) and their associated computations.

# Fields
- `reg_name::String`: The name of the regularization method or configuration being tested.
- `λ_vec::Vector{Float64}`: A vector of regularization parameter values (λ) to be evaluated.
- `material_images_vec::Vector{MI}`: A vector of material images corresponding to each λ value in `λ_vec`.
- `computation_times_vec::Vector{Float64}`: A vector of computation times for each regularization parameter test.
"""
struct λTest
    reg_name::String
    λ_vec::Vector{Float64}
    material_images_vec::Vector{MI}
    computation_times_vec::Vector{Float64}
end

# Constructor function
function λTest(prob::RegularizedDecompositionProblem, range_λ)
    reg_name = prob.regularization_name
    λ_vec = Vector{Float64}(undef, length(range_λ))
    material_images_vec = Vector{MI}(undef, length(range_λ))
    computation_times_vec = Vector{Float64}(undef, length(range_λ))

    for (i, λ) in enumerate(range_λ)
        print("Testing $reg_name with λ = $λ \n")

        prob.λ = λ
        material_images, stats =
            RegularizedDecomposition(prob, return_stats = Val(true))

        λ_vec[i] = λ
        material_images_vec[i] = material_images
        computation_times_vec[i] = stats.timer
    end
    λTest(reg_name, λ_vec, material_images_vec, computation_times_vec)
end

function plot!(
    ax::Axis,
    λ_test::λTest,
    circle,
    signal_mask,
    background_mask;
    x::Symbol = :fwhm,
    y::Symbol = :noise,
    reg_name::Union{Nothing,String} = nothing,
    ref_mat = :mat1,
    pixel_size = 1.0,
    params = nothing
)

    caracs =
        ImageCaracteristics(λ_test, circle, signal_mask, background_mask; ref_mat = ref_mat, pixel_size = pixel_size)
    x_val = caracs[!, x]
    y_val = caracs[!, y]
    λ_vec = caracs[!, :λ]

    if λ_test.reg_name == "" # used for slides to plot each point separate
        s = 10
        x_val = x_val[1:s]
        y_val = y_val[1:s]
        λ_vec = λ_vec[1:s]
    else
        s=10
    end

    # Use provided reg_name or λ_test.reg_name
    name = isnothing(reg_name) ? λ_test.reg_name : reg_name

    # Add text annotations for the lambda values at each point
    if isnothing(params)
        params = (
        pos = (:left, :top),
        offset = (6, 6),
        linestyle = :solid
    )
    end

    for (i, λ) in enumerate(λ_vec)
        lambda_str = @sprintf "λ=%.1e" λ
        text!(
            ax,
            x_val[i],
            y_val[i],
            text = lambda_str,
            align = params.pos,
            offset = params.offset,
            fontsize = 10,
        )
    end
    
    #scatter!(ax, x_val, y_val, markersize = 10)
    haskey(params, :linestyle) ? linestyle = params.linestyle : (linestyle = :solid)
    scatter!(ax, x_val, y_val, markersize = range(5, 5+s, length = length(x_val)))
    lines!(ax, x_val, y_val, label = name, linewidth = 2, linestyle = linestyle)
    # Do not force legend placement here so the caller (figure layout) can decide
    return nothing
end

function plot(
    λ_test_vec::Vector{λTest},
    circle::Circ,
    signal_mask,
    background_mask;
    x::Symbol = :fwhm,
    y::Symbol = :noise,
    reg_names::Union{Nothing,Vector{String}} = nothing,
    non_regularized_point::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    ref_mat = :mat1,
    pixel_size = 1.0,
    params = nothing
)

    n = length(λ_test_vec)
    n == 0 && error("λ_test_vec is empty")

    fig = Figure(backgroundcolor = :black, size = (800, 200 * n))

    # Column title at row 0
    Label(fig[0, 1], "Decomposition Characteristics", color = :white, halign = :center, tellwidth = false)

    for i = 1:n
        λ_test = λ_test_vec[i]
        reg_label = isnothing(reg_names) ? λ_test.reg_name : reg_names[i]

        # Row label on the left, rotated
        Label(fig[i, 0], reg_label, color = :white, rotation = pi / 2, tellheight = false, halign = :right)

        # Create axis for this regularization method
        ax = Axis(
            fig[i, 1],
            title = "",
            xlabel = string(x),
            ylabel = string(y),
        )

        # Delegate the plotting to the in-place plot! which will annotate points
        plot!(
            ax,
            λ_test,
            circle,
            signal_mask,
            background_mask;
            x = x,
            y = y,
            reg_name = isnothing(reg_names) ? nothing : reg_names[i],
            ref_mat = ref_mat,
            pixel_size = pixel_size,
            params = params[reg_label],
        )

        # If a non-regularized reference point is provided, only draw it on the
        # first row so it's visible in the grid context.
        if i == 1 && !isnothing(non_regularized_point)
            scatter!(
                ax,
                [non_regularized_point[1]],
                [non_regularized_point[2]],
                markersize = 16,
                marker = :star5,
                label = "No regularization",
            )
            text!(
                ax,
                non_regularized_point[1],
                non_regularized_point[2],
                text = "λ = 0.0",
                align = (:left, :bottom),
                offset = (5, 5),
            )
        end

        # Let each axis show its own legend
        axislegend(ax, position = :rt)
    end

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 5)
    return fig
end

function plot!(
    ax::Axis,
    λ_test_vec::Vector{λTest},
    circle::Circ,
    signal_mask,
    background_mask;
    x::Symbol = :fwhm,
    y::Symbol = :noise,
    reg_names::Union{Nothing,Vector{String}} = nothing,
    non_regularized_point::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    ref_mat = :mat1,
    pixel_size = 1.0,
    params = nothing
)

    for (i, λ_test) in enumerate(λ_test_vec)
        reg_name = isnothing(reg_names) ? nothing : reg_names[i]
        plot!(
            ax,
            λ_test,
            circle,
            signal_mask,
            background_mask;
            x = x,
            y = y,
            reg_name = reg_name,
            ref_mat = ref_mat,
            pixel_size = pixel_size,
            params = params[reg_name],
        )
    end

    # Plot non-regularized point if provided
    if !isnothing(non_regularized_point)
        scatter!(
            ax,
            [non_regularized_point[1]],
            [non_regularized_point[2]],
            markersize = 16,
            color = :orange,
            marker = :star5,
            label = "No regularization",
        )
        text!(
            ax,
            non_regularized_point[1],
            non_regularized_point[2],
            text = "λ = 0.0",
            align = (:left, :bottom),
            offset = (5, 5),
        )
    end
end


function plot_heatmap!(
    fig::Figure,
    i::Int,
    λ_test::λTest;
    crange,
    reg_name::String = "",
    zoom_region::Union{Nothing,AbstractVector} = nothing,
    ref_mat = :mat1,
)

    isempty(reg_name) && (reg_name = λ_test.reg_name)
    # Row titles on top (column 0)
    Label(
            fig[i, 0],
            reg_name,
            color = :white,
            tellheight = false,
            halign = :center,
            rotation = pi / 2
        )

    for (i, material_image) in enumerate(λ_test.material_images_vec)
        λ = λ_test.λ_vec[i]
        λ_str = @sprintf "λ=%.1e" λ

        # Column titles on top (row 0)
        Label(fig[0, i], λ_str, color = :white, halign = :center, tellwidth = false)

        mat = getfield(material_image, ref_mat)

        if !isnothing(zoom_region)
            zoom_x, zoom_y = zoom_region
            mat = @view mat[zoom_x, zoom_y]
        end

        ax1 = Axis(fig[1, i], aspect = DataAspect())
        heatmap!(ax1, mat, colormap = :grays, colorrange = crange)
        hidedecorations!(ax1)
        hidespines!(ax1)
    end

end

function plot_heatmap(
    λ_test::λTest;
    crange,
    reg_name::String = "",
    figsize = nothing,
    zoom_region::Union{Nothing,AbstractVector} = nothing,
    ref_mat = :mat1,
)
    # Create figure. If figsize is nothing, omit the size keyword to avoid passing `Nothing` to Figure.
    fig =
        isnothing(figsize) ? Figure(backgroundcolor = :black) :
        Figure(backgroundcolor = :black, size = figsize)
    plot_heatmap!(
        fig,
        1,
        λ_test;
        crange = crange,
        reg_name = reg_name,
        zoom_region = zoom_region,
        ref_mat = ref_mat,
    )
    return fig
end

"""
We assume that all λ_test have the same λ_vec for labeling
"""
function plot_heatmap(
    λ_test_vec::Vector{λTest};
    crange,
    reg_name::Vector{String} = [""],
    figsize = nothing,
    zoom_region::Union{Nothing,AbstractVector} = nothing,
    ref_mat = :mat1,
)

    @assert length(λ_test_vec) > 0 "λ_test_vec is empty"

        # Create figure. If figsize is nothing, omit the size keyword to avoid passing `Nothing` to Figure.
    fig =
        isnothing(figsize) ? Figure(backgroundcolor = :black) :
        Figure(backgroundcolor = :black, size = figsize)

    # We assume that all λ_test have the same λ_vec for labeling
    for (i, λ) in enumerate(λ_test_vec[1].λ_vec)
        λ_str = @sprintf "λ=%.1e" λ
        # Column titles on top (row 0)
        Label(fig[0, i], λ_str, color = :white, halign = :center, tellwidth = false)
    end

    for (i, λ_test) in enumerate(λ_test_vec)
        isempty(reg_name) ? (reg = λ_test.reg_name) : (reg = reg_name[i])
        # Row titles on top (column 0)
        Label(
                fig[i, 0],
                reg,
                color = :white,
                tellheight = false,
                halign = :center,
                rotation = pi / 2
            )

        for (j, material_image) in enumerate(λ_test.material_images_vec)

            mat = getfield(material_image, ref_mat)

            if !isnothing(zoom_region)
                zoom_x, zoom_y = zoom_region
                mat = @view mat[zoom_x, zoom_y]
            end

            ax1 = Axis(fig[i, j], aspect = DataAspect())
            heatmap!(ax1, mat, colormap = :grays, colorrange = crange)
            hidedecorations!(ax1)
            hidespines!(ax1)
            f2 = Figure(backgroundcolor = :black, size = (600, 600))
            ax2 = Axis(f2[1, 1], aspect = DataAspect())
            heatmap!(ax2, mat, colormap = :grays, colorrange = crange)
            hidedecorations!(ax2)
            hidespines!(ax2)
        end
    end
    return fig
end



"""
    plot_line_profile(image::AbstractMatrix, p1::Tuple, p2::Tuple; npoints::Int=200, ax=nothing)

Plots the intensity profile of `image` along the straight line from `p1` to `p2`.
- `p1`, `p2`: (row, col) coordinates.
- `npoints`: Number of points to sample along the line.
- `ax`: Optional Makie Axis to plot on.

Returns the sampled distances and intensity values.
"""
function plot_line_profile(image::AbstractMatrix, p1::Tuple, p2::Tuple; npoints::Int = 200)
    r1, c1 = p1
    r2, c2 = p2
    t = range(0, 1, length = npoints)
    rows = @. r1 + (r2 - r1) * t
    cols = @. c1 + (c2 - c1) * t

    # Interpolate image for subpixel sampling
    itp = linear_interpolation(
        (axes(image, 1), axes(image, 2)),
        Float64.(image),
        extrapolation_bc = Flat(),
    )
    intensities = [itp(r, c) for (r, c) in zip(rows, cols)]
    distances = sqrt.((rows .- r1) .^ 2 .+ (cols .- c1) .^ 2)

    fig = Figure(size = (800, 800))
    ax1 = Axis(
        fig[1, 1],
        xlabel = "Distance (pixels)",
        ylabel = "Intensity",
        title = "Line Profile",
    )
    lines!(ax1, distances, intensities)

    # Add inset axis for the image and line
    inset_width, inset_height = 0.3, 0.3
    ax_inset = Axis(
        fig[1, 1],
        width = Relative(inset_width),
        height = Relative(inset_height),
        halign = :right,
        valign = :top,
        aspect = DataAspect(),
    )
    # The following line may cause issues if image is not Float32/Float64 or has NaNs/Infs
    # Try converting image to Float64 and ensure finite values
    heatmap!(ax_inset, Float64.(image), colormap = :grays)
    lines!(ax_inset, [c1, c2], [r1, r2], color = :red, linewidth = 2)
    scatter!(ax_inset, [c1, c2], [r1, r2], color = :red)
    #hidedecorations!(ax_inset)
    #hidespines!(ax_inset)

    return fig
end


"""
    compute_lambda_for_noise(prob::RegularizedDecompositionProblem, background_mask, objective_noise; max_iter=20, tol=1e-3, λ_init=1.0)

Estimate the regularization parameter λ such that the noise (std) in the background region of the solution matches the desired `objective_noise`.

# Arguments
- `prob`: RegularizedDecompositionProblem instance (not mutated).
- `background_mask`: Mask (e.g., HyperRectangle or logical mask) defining the background region.
- `objective_noise`: Target standard deviation of noise in the background region.
- `λ_init`: Initial guess for λ
- `max_iter`: Maximum number of search iterations (default 20).
- `tol`: Tolerance for noise matching (default 1e-3).

# Returns
- Estimated λ value to achieve the target noise.
- The noise level achieved with the estimated λ.
"""
function compute_lambda_for_noise(
    prob::RegularizedDecompositionProblem,
    background_mask;
    objective_noise,
    λ_init,
    ref_mat = :mat1,
    max_iter = 20,
    reltol = 1e-3,
)
    # Helper to compute noise in background for a given λ
    function noise_for_lambda(λ)
        local_prob = deepcopy(prob)
        local_prob.λ = λ
        result = RegularizedDecomposition(local_prob)
        return image_noise(getfield(result, ref_mat), background_mask)
    end

    λ_low = λ_init / 100
    λ_high = 100 * λ_init
    best_λ = λ_init
    for iter = 1:max_iter
        λ_try = sqrt(λ_low * λ_high)
        noise = noise_for_lambda(λ_try)
        if abs(noise - objective_noise) < reltol * objective_noise
            return λ_try, noise_for_lambda(best_λ)
        elseif noise > objective_noise
            λ_low = λ_try
        else
            λ_high = λ_try
        end
        best_λ = λ_try
    end
    @warn "Did not converge to target noise within tolerance for $(prob.regularization_name). Returning best λ found. You can try increasing max_iter or just a better initial λ_init."
    return best_λ, noise_for_lambda(best_λ)
end


function plot_lambda_vs_time(
    λ_test_vec::Vector{λTest},
    offset::Vector{Float64},
    labels::Vector{String},
    target_λ= nothing,
)
    fig = Figure(size = (900, 400))
    ax = Axis(
        fig[1, 1],
        title = "Computation Time vs Regularization Parameter λ",
        xlabel = "Regularization Parameter λ",
        ylabel = "Computation Time (s)",
        yminorticksvisible = true,
        yminorgridvisible = true,
        yticks=LogTicks(-1:2),
        yminorticks = IntervalsBetween(9),
        yscale = log10,
    )
    for (i, λ_test) in enumerate(λ_test_vec)
        CairoMakie.scatterlines!(
            ax,
            λ_test.λ_vec,
            λ_test.computation_times_vec .+ offset[i],
            label = labels[i],
        )
        if offset[i] != 0.0
            CairoMakie.scatterlines!(
                ax,
                λ_test.λ_vec,
                λ_test.computation_times_vec,
                label = labels[i] * "\n(solving time only)",
                linestyle = :dash,
                alpha = 0.5,
            )
        end
    end

    if !isnothing(target_λ)
        target_λ_rounded = @sprintf "%.1e" target_λ
        vlines!(
            ax,
            [target_λ],
            linestyle = :dash,
            label = "Target λ = $target_λ_rounded",
        )
    end

    hlines!(
        ax,
        [0.200],
        linestyle = :dash,
        label = "Near real-time\n(200ms)",
    )


    # place legend in a separate column to the right (outside the main axis)
    fig[1, 2] = axislegend(ax, position = :ct)
    fig
end
