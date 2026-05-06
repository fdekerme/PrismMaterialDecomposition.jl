module PrismMaterialDecomposition

using Attenuations: Elements, Element, Materials, Material, μᵨ, val
using Attenuations: μ as μ_linear
using CairoMakie
using CircleFit
using CUDA
using CUDA.CUSPARSE
using DataFrames
using Distributions
using FileIO
using GeometryBasics: Rect
using HDF5
using Images: Gray, canny, Percentile, RGB
using GeometryBasics: HyperRectangle
using Images.ImageContrastAdjustment: AbstractHistogramAdjustmentAlgorithm, LinearStretching
using Interpolations
using JLD2
using Krylov
using LinearAlgebra
using LinearSolve
using LineSearch
using NonlinearSolve
using LsqFit
using MAT: matread
using Metal
using Printf: @sprintf
using SciMLOperators
using SparseArrays
using SpecialFunctions
using Statistics
using Unitful


import LinearAlgebra: cond, normalize
import Images: adjust_histogram
import Images.ImageTransformations: imrotate
import ImageFiltering: imfilter
import CairoMakie: plot, plot!

const Axis = CairoMakie.Axis
const scatter! = CairoMakie.scatter!

include("types.jl")
include("linear_solve.jl")
include("lambda_test.jl")
include("utils.jl")
include("plot_utils.jl")
include("data_loader.jl")
include("quality_metrics.jl")
include("blend_functions.jl")
include("matlab_parser.jl")
include("ESF_utils.jl")
include("A_contructor.jl")
include("V_constructor.jl")
include("regularization_utils.jl")
include("regularization_matrix.jl")
include("regularization_operators.jl")
include("regularization_operators_gpu.jl")
include("regularization_constructors.jl")

export DLI, μ, Blend, MI, Spectra, ESF, LSF, Circ
export log_norm, load_images, load_png_images, load_jld2_images, load_jld2, load_spectra
export change_of_basis, material_decomposition, VMI_blend
export compute_mean_energy, effective_mass_attenuations, effective_linear_attenuations
export normalize, matrix, rectangle, extract_pixels, compute_edge_map, fit_circle_on_zoomed
export Regularization,
    RegularizedDecompositionProblem,
    RegularizedDecompositionProblemCPU,
    RegularizedDecompositionProblemCUDA,
    RegularizedDecompositionProblemMetal,
    RegularizedDecomposition
export ∇R_quad_op,
    ∇R_quad_mat,
    ∇R_quad_ew_op,
    ∇R_quad_ew_mat,
    ∇R_similarity_op,
    ∇R_similarity_mat, #can be both similarity and cross-similarity, depending on the input matrix
    ∇R_cross_similarity_op_gpu_metal, #no mat version on Metal (sparse array not supported)
    ∇R_cross_similarity_op_gpu_cuda,
    ∇R_similarity_mat_gpu_cuda #can be both similarity and cross-similarity, depending on the input matrix
export generate_quadratic_regularization_matrix,
    generate_ew_quadratic_regularization_matrix,
    generate_similarity_matrix_W,
    generate_cross_similarity_matrix_W
export euclidean_distance, gaussian_spatial_distance, no_distance_penalty
export image_noise, SNR, CNR
export fwhm, fit_lsf, fit_esf, plot, plot!, plot_lambda_vs_time
export ImageCaracteristics, λTest, compute_lambda_for_noise
export enable_plot_theme!, plot_spectra, plot_mean_spectra, plot_attenuation, plot_energy_map, plot_material_map
export plot_histogram, plot_heatmap, plot_materials_grid, plot_esf_lsf, plot_heatmap_similarity
export mat_to_JLD2, mat_to_png, mat_spectra_to_JLD2
export compute_contrast_std, compute_contrast_stats


export NonLinearPrismSolver,
    NonLinearDecompositionProblem,
    NonLinearDecompositionProblemCPU,
    NonLinearDecompositionProblemMetal,
    NonLinearDecompositionProblemCUDA,
    NonLinearDecompositionProblemGPUArraysCUDA,
    NonLinearDecomposition,
    NonLinearDecompositionGN

#Re-export from Attenuations.jl
export val, Materials, Elements

end # module PrismMaterialDecomposition
