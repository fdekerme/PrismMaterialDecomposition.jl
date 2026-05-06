using Test
using PrismMaterialDecomposition

@testset "PrismMaterialDecomposition Tests" begin

    function generate_fake_DLI(rows::Int, cols::Int)
        r = rand(rows, cols)
        top_image = r .+ 0.5
        bottom_image = r .+ 0.6
        return DLI(top_image, bottom_image)
    end

    @testset "Material Decomposition (No Regularization)" begin
        dli_images = generate_fake_DLI(512, 512)

        material_1 = Materials.PMMA
        material_2 = Elements.Copper

        μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
        μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

        material_images = material_decomposition(dli_images, μ₁, μ₂)

        @test material_images isa MI
        @test size(material_images.mat1) == size(dli_images.top)
        @test size(material_images.mat2) == size(dli_images.top)
    end

    @testset "Regularization Setup" begin
        dli_images = generate_fake_DLI(512, 512)

        # Test edge detection
        combined_edges = compute_edge_map(dli_images, 0.965)
        @test combined_edges isa BitMatrix
        @test size(combined_edges) == size(dli_images.top)

        # Test regularization construction
        reg_quad = Regularization("quad", ∇R_quad_op, (2.0, 2.0, 10.0))
        @test reg_quad.name == "quad"

        reg_quad_ew = Regularization(
            "quad_ew",
            ∇R_quad_ew_op,
            (combined_edges, 0.2, 1.0),
        )
        @test reg_quad_ew.name == "quad_ew"
    end

    @testset "Regularized Decomposition Problem" begin
        dli_images = generate_fake_DLI(512, 512)

        material_1 = Materials.PMMA
        material_2 = Elements.Copper

        μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
        μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

        background_mask = rectangle(200, 250, 290, 340)
        λ = 10e-3

        reg_quad = Regularization("quad", ∇R_quad_op, (2.0, 2.0, 10.0))
        p_quad = RegularizedDecompositionProblemCPU(
            dli_images,
            μ₁,
            μ₂,
            λ,
            background_mask,
            reg_quad,
        )

        @test p_quad isa RegularizedDecompositionProblem
        @test p_quad.λ == λ
    end

    @testset "Regularized Decomposition with Regularization" begin
        dli_images = generate_fake_DLI(512, 512)

        material_1 = Materials.PMMA
        material_2 = Elements.Copper

        μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
        μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

        background_mask = rectangle(200, 250, 290, 340)
        λ = 10e-3

        reg_quad = Regularization("quad", ∇R_quad_op, (2.0, 2.0, 10.0))
        p_quad = RegularizedDecompositionProblemCPU(
            dli_images,
            μ₁,
            μ₂,
            λ,
            background_mask,
            reg_quad,
        )

        material_images_reg = RegularizedDecomposition(p_quad)

        @test material_images_reg isa MI
        @test size(material_images_reg.mat1) == size(dli_images.top)
    end

    # Test Metal regularization if Metal.jl and compatible hardware are available

    using Metal
    if Metal.device() !== nothing
        @testset "Linear Decomposition with Metal Regularization" begin
            dli_images = generate_fake_DLI(512, 512)

            material_1 = Materials.PMMA
            material_2 = Elements.Copper

            μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
            μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

            background_mask = rectangle(200, 250, 290, 340)
            λ = 10e-3

            params_sim_metal = (dli_images.top, dli_images.bottom, 0.1, 0.1, 5)

            reg_metal = Regularization(
                "similarity",
                ∇R_cross_similarity_op_gpu_metal,
                params_sim_metal,
            )
            p_metal = RegularizedDecompositionProblemMetal(
                dli_images,
                μ₁,
                μ₂,
                λ,
                background_mask,
                reg_metal,
            )

            material_images_reg = RegularizedDecomposition(p_metal)

            @test material_images_reg isa MI
            @test size(material_images_reg.mat1) == size(dli_images.top)
        end
    else
        @info "Metal.jl not available or no compatible device"
    end

    # Test CUDA regularization if CUDA.jl and compatible hardware are available
    using CUDA
    if CUDA.has_cuda()
            @testset "Regularized Decomposition with CUDA Regularization" begin
            dli_images = generate_fake_DLI(512, 512)

            material_1 = Materials.PMMA
            material_2 = Elements.Copper

            μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
            μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

            background_mask = rectangle(200, 250, 290, 340)
            λ = 10e-3

            params_sim_cuda = (dli_images.top, dli_images.bottom, 0.1, 0.1, 5)

            reg_cuda = Regularization(
                "similarity",
                ∇R_cross_similarity_op_gpu_cuda,
                params_sim_cuda,
            )

            p_cuda = RegularizedDecompositionProblemCUDA(
                dli_images,
                μ₁,
                μ₂,
                λ,
                background_mask,
                reg_cuda,
            )

            material_images_reg = RegularizedDecomposition(p_cuda)

            @test material_images_reg isa MI
            @test size(material_images_reg.mat1) == size(dli_images.top)
        end
    else
        @info "CUDA.jl not available or no compatible device"
    end

    @testset "ESF and LSF Computation" begin
        dli_images = generate_fake_DLI(512, 512)

        material_1 = Materials.PMMA
        material_2 = Elements.Copper

        μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
        μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

        material_images = material_decomposition(dli_images, μ₁, μ₂)
        circle = Circ((198.0, 177.0), 7.0)

        esf = ESF(material_images.mat1, circle)
        lsf = LSF(esf)
        fwhm_val = fwhm(lsf)

        @test esf isa ESF
        @test lsf isa LSF
        @test fwhm_val > 0
    end

    @testset "Decomposition Characteristics" begin
        dli_images = generate_fake_DLI(512, 512)

        material_1 = Materials.PMMA
        material_2 = Elements.Copper

        μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
        μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)
        material_images = material_decomposition(dli_images, μ₁, μ₂)

        circle = Circ((198.0, 177.0), 7.0)
        signal_mask = rectangle(195, 200, 175, 180)
        background_mask = rectangle(200, 250, 290, 340)

        caracs = ImageCaracteristics(
            material_images,
            circle,
            signal_mask,
            background_mask,
            ref_mat = :mat1,
        )

        @test caracs.noise[1] > 0
        @test caracs.fwhm[1] > 0
        @test caracs.cnr[1] > 0
    end

    @testset "λTest Functionality" begin
        dli_images = generate_fake_DLI(512, 512)

        material_1 = Materials.PMMA
        material_2 = Elements.Copper

        μ₁ = μ("Water, Liquid", 0.1926, 0.1781)
        μ₂ = μ("Bone, Cortical (ICRU-44)", 0.5173, 0.4058)

        background_mask = rectangle(200, 250, 290, 340)
        λ_range = 10 .^ range(-4, -1, length = 4)

        reg_quad = Regularization("quad", ∇R_quad_op, (2.0, 2.0, 10.0))
        p_quad = RegularizedDecompositionProblemCPU(
            dli_images,
            μ₁,
            μ₂,
            λ_range[1],
            background_mask,
            reg_quad,
        )

        λ_vec = λTest(p_quad, λ_range)

        @test length(λ_vec.material_images_vec) == length(λ_range)
        @test all(mat -> mat isa MI, λ_vec.material_images_vec)
    end

end
