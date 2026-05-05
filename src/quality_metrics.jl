#This file contains functions to compute image quality metrics such as SNR, CNR, noise, and mean signal/background values.

function ImageCaracteristics(
    image::AbstractMatrix,
    circle::Circ,
    signal_mask,
    background_mask;
    λ = nothing,
    pixel_size = 1.0
)
    esf = ESF(image, circle, pixel_size = pixel_size)
    esf_fwhm = fwhm(esf) #fit esf with a logistic to get the lsf fwhm

    cnr = CNR(image, signal_mask, background_mask)
    snr = SNR(image, signal_mask, background_mask)
    noise = image_noise(image, background_mask)
    mean_signal = mean(extract_pixels(image, signal_mask))
    mean_background = mean(extract_pixels(image, background_mask))

    DataFrame(λ = λ, fwhm = esf_fwhm, cnr = cnr, snr = snr, noise = noise, mean_signal = mean_signal, mean_background = mean_background)
end

function ImageCaracteristics(
    material_images::MI,
    circle::Circ,
    signal_mask,
    background_mask;
    ref_mat = :mat1,
    λ = nothing,
    pixel_size = 1.0
)
    ref_image = getfield(material_images, ref_mat)
    return ImageCaracteristics(
        ref_image,
        circle,
        signal_mask,
        background_mask;
        λ = λ,
        pixel_size = pixel_size
    )
end

function ImageCaracteristics(
    λ_test::λTest,
    circle::Circ,
    signal_mask,
    background_mask;
    ref_mat = :mat1,
    pixel_size = 1.0,
)
    results = DataFrame(
        λ = Float64[],
        fwhm = Float64[],
        cnr = Float64[],
        snr = Float64[],
        noise = Float64[],
        mean_signal = Float64[],
        mean_background = Float64[],
    )

    for (i, material_images) in enumerate(λ_test.material_images_vec)
        λ = λ_test.λ_vec[i]
        df = ImageCaracteristics(
            material_images,
            circle,
            signal_mask,
            background_mask,
            ref_mat = ref_mat,
            λ = λ,
            pixel_size = pixel_size,
        )
        append!(results, df)
    end
    return results
end

function image_noise(image, background_mask)
    local background = extract_pixels(image, background_mask)
    return std(background)
end

function SNR(image, signal_mask, background_mask)
    local signal = extract_pixels(image, signal_mask)
    local background = extract_pixels(image, background_mask)
    snr = mean(signal) / std(background)
    return round(snr, digits = 2)
end

function CNR(image, signal_mask, background_mask)
    local signal = extract_pixels(image, signal_mask)
    local background = extract_pixels(image, background_mask)
    cnr = abs(mean(signal) - mean(background)) / std(background)
    return round(cnr, digits = 2)
end


