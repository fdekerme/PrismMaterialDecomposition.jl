### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ efc10042-c560-11f0-944c-f3a972cc8c8b
begin
	using Pkg
	Pkg.activate("../")
end

# ╔═╡ 9aee381e-8248-429c-837a-0a67ea5c76f0
begin
	using Prism
	using CairoMakie
	using PlutoUI
	using JLD2
	import PlutoUI: combine, Slider, Select
	using MAT
	enable_plot_theme!() #enable latexfonts CairoMakie plot theme
end;

# ╔═╡ 5fb4c21d-07ef-48d9-bb5f-fa42a62107f0
md"#### Decomposition backend (Metal GPU or CUDA GPU"

# ╔═╡ 88d03c8e-0e7a-4156-8ea4-6fe34d439f20
@bind reg Select(["Metal GPU", "CUDA GPU"])

# ╔═╡ ba9b0aa6-14ad-4d36-a7a6-4130bb3289c2
function decomposition_params(params::Vector)
	
	return combine() do Child
		
		inputs = [
			md""" $(p[1]): $(
				Child(p[1], Slider(p[2], show_value=true))
			)"""
			
			for p in params
		]
		
		md"""
		#### Parameters
		- Regularization weight λ
		$(inputs[1])
		- Log-substraction parameter α
		$(inputs[2])
		- Projection angle θ
		$(inputs[3])
		"""
	end
end

# ╔═╡ a562072e-7021-4c23-9c73-43070e6a7a81
@bind params decomposition_params([("λ", 0.0:0.1:5.0) , ("α", -1.5:0.001:2.0), ("θ", 1:350)])

# ╔═╡ 369b1d5a-3aa3-4fb3-b05a-9905167c085d
begin
	λ, α, x = params.λ, params.α, params.θ
	#p = "../data/Patient_21/output_20260205170231/881753813"
	#frame_name = "Frame02824.jld2"
	
	#p = "../data/Patient_21/output_20260205170203/881753812"
	#frame_name = "Frame02703.jld2"
	
	#p = "../data/Leeds TOR18 3"
	#frame_name = "frame_$x.jld2"

	#p = "../data/Thorax_Patient1"
	#frame_name = "frame_$x.jld2"

	#p = "../data/Aquisitions_3_20_2026/Pelvic_phantom_outputs/trig_noshift_beam1_output/886091351"
	#frame_name = "Frame28872.jld2"

	p = "../data/Aquisitions_3_20_2026/Thorax_phantom_outputs/trig_noshift_beam1_output/886091296"
	frame_name = "Frame07772.jld2"
	
	dli_images = load_images(p, frame_name, true)
	#dli_images = DLI(rotr90(dli_images.top), rotr90(dli_images.bottom))
	dli_images = DLI(reverse(dli_images.top, dims=2), reverse(dli_images.bottom, dims=2))
	#dli_images = DLI(dli_images.top[begin:end, 250:end], dli_images.bottom[begin:end, 250:end])

	A_inv = [-1.0 α; 1.0 0.0]
	A = inv(A_inv)
	μ₁ = μ( "log", A[1, 1], A[2, 1])
	μ₂ = μ("bone", A[1, 2], A[2, 2])
	
	material_images_no_reg = material_decomposition(dli_images, μ₁, μ₂)
	background_mask = rectangle(125, 150, 105,130)
	#h_top, h_bottom = image_noise(dli_images.top, background_mask), image_noise(dli_images.bottom, background_mask)
	h_top, h_bottom = 0.25, 0.25
	half_size = 10
	
	if reg == "Metal GPU"
		reg_cross_similarity_op_metal = Regularization("cross_similarity_op_metal", ∇R_cross_similarity_op_gpu_metal, (dli_images.top, dli_images.bottom, h_top, h_bottom, half_size))
		
		p_metal = RegularizedDecompositionProblemMetal(dli_images, μ₁, μ₂, λ, background_mask, reg_cross_similarity_op_metal)
		material_images  = RegularizedDecomposition(p_metal)
	elseif reg == "CUDA GPU"
		reg_cross_similarity_op_cuda = Regularization("cross_similarity_op_cuda", ∇R_cross_similarity_op_gpu_cuda, (dli_images.top, dli_images.bottom, h_top, h_bottom, half_size))
		
		p_cuda = RegularizedDecompositionProblemCUDA(dli_images, μ₁, μ₂, λ, background_mask, reg_cross_similarity_op_cuda)
		material_images  = RegularizedDecomposition(p_cuda)
	end


	f = Figure(size = (1200, 400))
	ax1 = Axis(f[1,1], aspect = DataAspect(), title = "Top layer")
	heatmap!(ax1, dli_images.top, colormap = :grays)
	
	ax2 = Axis(f[1, 2], aspect = DataAspect(), title = "No regularization")
	heatmap!(ax2, material_images_no_reg.mat1, colormap = :grays)
	crange = (minimum(material_images_no_reg.mat1), maximum(material_images_no_reg.mat1))
	
	ax3 = Axis(f[1,3], aspect = DataAspect(), title = "Lung tissues")
	heatmap!(ax3, material_images.mat1, colormap = :grays, colorrange = crange)

	f
end

# ╔═╡ 6fc9b6b4-d9e8-4319-ae54-a4b9d427c94c
md"Save: $(@bind save Switch())"

# ╔═╡ 9e6c7e06-92bb-4a2d-b398-0096766af579
if save
	matwrite("$(frame_name)_decomposed.mat", Dict(
		"mat1_data" => material_images.mat1,
		"mat2_data" => material_images.mat2,
		"top_data" => dli_images.top,
		"bottom_layer_data" => dli_images.bottom,
		"metatdata" => Dict(
			"A" => A,
			"mat1" => μ₁.name,
			"mat2" => μ₂.name,
			"lambda" => λ,
			"regularization" => reg_cross_similarity_op_metal.name
	)))
end

# ╔═╡ Cell order:
# ╟─efc10042-c560-11f0-944c-f3a972cc8c8b
# ╠═9aee381e-8248-429c-837a-0a67ea5c76f0
# ╟─5fb4c21d-07ef-48d9-bb5f-fa42a62107f0
# ╟─88d03c8e-0e7a-4156-8ea4-6fe34d439f20
# ╟─a562072e-7021-4c23-9c73-43070e6a7a81
# ╟─369b1d5a-3aa3-4fb3-b05a-9905167c085d
# ╟─ba9b0aa6-14ad-4d36-a7a6-4130bb3289c2
# ╟─6fc9b6b4-d9e8-4319-ae54-a4b9d427c94c
# ╟─9e6c7e06-92bb-4a2d-b398-0096766af579
