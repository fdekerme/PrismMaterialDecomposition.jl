# This file contains utility functions for computing and analyzing the Edge Spread Function (ESF) and Line Spread Function (LSF) from images.

"""
        ESF(image, center, radius; num_angles=360, profile_range=20.0, profile_points=400)

    Computes the Edge Spread Function (ESF) from a circular edge in an image.

    It works by extracting multiple radial line profiles across the circle's edge
    at different angles, aligning and averaging them to create a high-resolution ESF

    # Arguments
    - `image`: The 2D image matrix.
    - `center`: A tuple `(center_row, center_col)` specifying the circle's center coordinates.
    - `radius`: The radius of the circle in pixels.

    # Keyword Arguments
    - `num_angles`: The number of radial profiles to extract around the circle. Default is 360.
    - `profile_range`: The range in pixels on either side of the nominal radius to sample. Default is 20.0.
    - `profile_points`: The number of points to sample in each profile, allowing for sub-pixel resolution. Default is 400.
    """
function ESF(
    image::AbstractMatrix,
    circle::Circ;
    range_angles::Tuple{Number,Number} = (0.0, 2π),
    num_angles::Int = 360,
    profile_range::Float64 = 20.0,
    profile_points::Int = 400,
    pixel_size::Float64 = 1.12,
)
    center_row, center_col = circle.center
    rad = circle.radius

    # 1. Create an interpolation object for the image.
    itp = linear_interpolation(
        (axes(image, 1), axes(image, 2)),
        Float64.(image),
        extrapolation_bc = Flat(),
    )

    # 2. Define the spatial coordinates for the ESF profile.
    esf_x = range(-profile_range / 2, profile_range / 2, length = profile_points)

    # 3. Extract radial profiles at various angles.
    all_profiles = []
    angles = range(range_angles[1], range_angles[2], length = num_angles)

    for angle in angles
        profile = zeros(Float64, profile_points)
        for (i, d) in enumerate(esf_x)
            current_radius = rad + d
            row = center_row + current_radius * sin(angle)
            col = center_col + current_radius * cos(angle)
            profile[i] = itp(row, col)
        end
        push!(all_profiles, profile)
    end

    # 4. Compute the ESF by averaging all extracted profiles.
    if isempty(all_profiles)
        error("No profiles were extracted. Check input parameters.")
    end
    esf_y = mean(hcat(all_profiles...), dims = 2)[:, 1]
    esf_x = esf_x .* pixel_size  # Convert to physical units

    # Return all arrays needed for plotting.
    return ESF(esf_x, esf_y)
end

function LSF(
    image::AbstractMatrix,
    circle::Circ;
    range_angles::Tuple{Number,Number} = (0.0, 2π),
    num_angles::Int = 360,
    profile_range::Float64 = 20.0,
    profile_points::Int = 400,
    pixel_size::Float64 = 1.12,
)

    esf = ESF(image, circle, range_angles, num_angles, profile_range, profile_points, pixel_size)

    # 5. Compute the LSF by differentiating the ESF.
    # The LSF is the derivative of the ESF. We use a simple finite difference.
    # The resulting array will be one element shorter than the ESF.
    lsf_y = diff(esf.y) ./ diff(esf.x)

    # Create a corresponding distance array for the LSF, which is one element shorter.
    # We shift the points to be in the middle of the original ESF distance points.
    lsf_x = esf.x[1:end-1] .+ (step(esf.x) / 2)

    return LSF(lsf_x, lsf_y)
end

function LSF(esf::ESF)
    lsf_y = diff(esf.y) ./ diff(esf.x)
    # Create a corresponding distance array for the LSF, which is one element shorter.
    # We shift the points to be in the middle of the original ESF distance points.
    lsf_x = esf.x[1:end-1] .+ (step(esf.x) / 2)

    return LSF(lsf_x, lsf_y)
end


gaussian(x, a::Vector{Float64}) = @. a[1] * exp(-0.5 * ((x - a[2]) / a[3])^2) + a[4]
"""
    fit_lsf(lsf::LSF)

Fit a gaussian function to the provided LSF data.
No longer used; we directly fit the esf with a logistic and derive the lsf fwhm from it.
"""
function fit_lsf(lsf::LSF)
    # Define a Gaussian model with offset: y = A * exp(-0.5 * ((x - μ)/σ)^2) + C
    # Initial guesses: amplitude, mean, sigma, offset

    # Robust initial guesses that handle negative LSFs:
    x = lsf.x
    y = lsf.y

    # Estimate baseline from the median (robust to outliers)
    baseline = median(y)

    # Centered signal and locate the strongest (abs) peak, which may be negative
    y_centered = y .- baseline
    peak_idx = argmax(abs.(y_centered))
    A0 = y_centered[peak_idx]           # can be negative or positive
    μ0 = x[peak_idx]

    # Width guess: use span/8 or at least one step to avoid zero
    span = maximum(x) - minimum(x)
    σ0 = max(span / 8, step(x))

    C0 = baseline

    p0 = Float64[A0, μ0, abs(σ0), C0]
    
    fit = LsqFit.curve_fit(gaussian, lsf.x, lsf.y, p0)
    p = coef(fit)
    return p
end


fwhm_gaussian(p::Vector{Float64}) = 2 * sqrt(2 * log(2)) * abs(p[3])
"""
    fwhm(lsf::LSF)
Compute the Full Width at Half Maximum (FWHM) of the provided LSF by fitting a Gaussian.
No longer used; we directly fit the esf with a logistic and derive the lsf fwhm from it.
"""
function fwhm(lsf::LSF)
    p_lsf = fit_lsf(lsf)
    # FWHM of a Gaussian is 2 * sqrt(2 * log(2)) * σ
    return fwhm_gaussian(p_lsf)
end

sigmoid(x, p) = @. p[1] + p[2] / (1 + exp(- (x - p[3]) / p[4]))
prime_sigmoid(x, p) = @. (p[2] / p[4]) * exp(- (x - p[3]) / p[4]) / (1 + exp(- (x - p[3]) / p[4]))^2
"""
    fit_esf(esf::ESF)
Fit a sigmoid function to the provided ESF data.
"""
function fit_esf(esf::ESF)
    # Fit a 4-parameter logistic (sigmoid) to the ESF:
    #   y = A + B / (1 + exp(-(x - μ) / σ))
    x = esf.x
    y = esf.y

    if isempty(x) || isempty(y) || length(x) != length(y)
        @warn "ESF data empty or lengths mismatch. Returning NaN parameter vector."
        return fill(NaN, 4)
    end

    n = length(x)
    n_end = max(1, Int(clamp(round(n * 0.05), 1, n))) # use ~5% of points to estimate asymptotes

    # Robust estimates for asymptotes
    low_est = median(y[1:n_end])
    high_est = median(y[end - n_end + 1:end])

    A0 = low_est
    B0 = high_est - low_est

    # initial center: location of the half-way value
    mid_val = A0 + B0 / 2
    μ0 = x[argmin(abs.(y .- mid_val))]

    # scale guess
    σ0 = max((maximum(x) - minimum(x)) / 10, step(x))

    p0 = Float64[A0, B0, μ0, σ0]
    p1 = deepcopy(p0)

    try
        fit = LsqFit.curve_fit(sigmoid, x, y, p0)
        p1 = coef(fit)  # [A, B, μ, σ]
    catch e
        @warn "fit_esf: sigmoid fit failed, returning initial guess. Error: $e"
        return p1
    end

    # We then deed to refine p) to match the peak of the lsf (caracterised by the parameter p[4])
    lsf = LSF(esf)
    try
        prime_sigmoid_temp(x, p4) = prime_sigmoid(x, [p1[1], p1[2], p1[3], p4[1]])
        fit = LsqFit.curve_fit(prime_sigmoid_temp, lsf.x, lsf.y, [p1[4]])
        p1[4] = coef(fit)[1]  # update only the σ (p1[4]) parameter
        return p1
    catch e
        @warn "fit_esf: refining σ parameter from LSF failed, returning previous fit. Error: $e"
        return p1
    end
end


fwhm_prime_sigmoid(p_esf::Vector{Float64}) = abs(p_esf[4]) * log((3 + 2 * sqrt(2)) / (3 - 2 * sqrt(2)))
"""
    fwhm(esf::ESF)
Compute the Full Width at Half Maximum (FWHM) of the provided ESF by fitting a logistic function.
The derivative of a logistic is given by prime_sigmoid.
The FWHM of this derivative only depend on the σ (p[4]) parameter of the logistic.
For logistic derivative the FWHM is: FWHM = σ * log((3 + 2√2) / (3 - 2√2)) ≈ 3.52549 * σ
It can be demonstrated using the hyperbolic secant (sech) function
 https://en.wikipedia.org/wiki/Full_width_at_half_maximum
"""
function fwhm(esf::ESF)

    p_esf = fit_esf(esf)
    if any(isnan.(p_esf))
        @warn "fwhm(esf): fit_esf failed, returning NaN."
        return NaN
    end
    return fwhm_prime_sigmoid(p_esf)
end

