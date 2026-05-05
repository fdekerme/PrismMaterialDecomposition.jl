"""
        RegularizedDecompositionProblem{Tλ,Tmat,Tvec}

Abstract container for all data and operators needed to solve a regularized
decomposition of dual-energy images using RegularizedDecomposition function.
This struct is not intended to be constructed directly, but via one of the
provided constructors: 
- RegularizedDecompositionProblemCPU
- RegularizedDecompositionProblemCUDA
- RegularizedDecompositionProblemMetal

Parameters / fields
- `dli_images::DLI` : Dual-energy image container (expects `.top` and `.bottom`).
- `μ₁::μ`, `μ₂::μ` : Material attenuation descriptors with `.low` and `.high` fields.
- `λ::Tλ` : Regularization weight.
- `background_mask::HyperRectangle{2,Int}` : Mask for background statistics / V⁻¹ construction.
- `nrows::Int`, `ncols::Int`, `N::Int` : Image dimensions and flattened size.
- `M::Tmat` : System matrix (2x2) with attenuation coefficients.
- `L::Tvec` : Stacked data vector (top then bottom images).
- `initial_guess::Tvec`
- `prototype::Tvec` : Solver initial guess and prototype used by regularizers.
- `regularization_name::String` : Human readable regularizer name.
- `A`, `Aᵀ`, `V⁻¹`, `∇R` : Operators used by the linear system. May be matrix or function operators
    and can be CPU, CUDA or Metal backed depending on the constructor used.

Notes
- The struct is mutable to allow updating fields (e.g., `λ` or `initial_guess`) without
    reallocating the entire container.
"""
mutable struct RegularizedDecompositionProblem{Tλ,Tmat,Tvec}
    dli_images::DLI
    μ₁::μ
    μ₂::μ
    λ::Tλ
    background_mask::HyperRectangle{2,Int}

    nrows::Int
    ncols::Int
    N::Int

    M::Tmat
    L::Tvec
    initial_guess::Tvec
    prototype::Tvec
    regularization_name::String

    A::Union{FunctionOperator,MatrixOperator}
    Aᵀ::Union{FunctionOperator,MatrixOperator}
    V⁻¹::Union{FunctionOperator,MatrixOperator}
    ∇R::Union{SciMLOperators.ComposedOperator,FunctionOperator,MatrixOperator}
end


"""
    RegularizedDecompositionProblemCPU(dli_images, μ₁, μ₂, λ, background_mask, reg)

Build a `RegularizedDecompositionProblem` using CPU-backed types (`Matrix`, `Vector`).

Arguments
- `dli_images::DLI` : dual-energy images with `.top` and `.bottom` fields.
- `μ₁::μ`, `μ₂::μ` : material attenuation descriptors with `.low` and `.high` fields.
- `λ::Float64` : regularization weight.
- `background_mask::HyperRectangle{2,Int}` : mask used to construct `V⁻¹`.
- `reg::Regularization` : regularizer object whose `.constructor` builds `∇R`.

Returns
- `RegularizedDecompositionProblem{Float64,Matrix{Float64},Vector{Float64}}`.
"""
function RegularizedDecompositionProblemCPU(
    dli_images::DLI,
    μ₁::μ,
    μ₂::μ,
    λ::Float64,
    background_mask::HyperRectangle{2,Int},
    reg::Regularization,
)
    nrows, ncols = size(dli_images.top)
    N = nrows * ncols

    M = [μ₁.low μ₂.low; μ₁.high μ₂.high]
    L = vcat(vec(dli_images.top), vec(dli_images.bottom))

    initial_guess_result = material_decomposition(dli_images, μ₁, μ₂)
    initial_guess_vec = vcat(vec(initial_guess_result.mat1), vec(initial_guess_result.mat2))

    prototype = zeros(Float64, 2N)
    A = A_mat_cpu(N, M)
    Aᵀ = At_mat_cpu(N, M)
    V⁻¹ = Vinv_mat_cpu(dli_images, background_mask)
    ∇R = reg.constructor(
        (N, nrows, ncols, LinearIndices((nrows, ncols)), reg.params),
        prototype,
    )

    return RegularizedDecompositionProblem{Float64,Matrix{Float64},Vector{Float64}}(
        dli_images,
        μ₁,
        μ₂,
        λ,
        background_mask,
        nrows,
        ncols,
        N,
        M,
        L,
        initial_guess_vec,
        prototype,
        reg.name,
        A,
        Aᵀ,
        V⁻¹,
        ∇R,
    )
end


"""
    RegularizedDecompositionProblemCUDA(dli_images, μ₁, μ₂, λ, background_mask, reg)

Build a `RegularizedDecompositionProblem` using CUDA-backed types (`CuMatrix`, `CuVector`).

Arguments
- `dli_images::DLI` : dual-energy images with `.top` and `.bottom` fields.
- `μ₁::μ`, `μ₂::μ` : material attenuation descriptors with `.low` and `.high` fields.
- `λ::Float64` : regularization weight.
- `background_mask::HyperRectangle{2,Int}` : mask used to construct `V⁻¹`.
- `reg::Regularization` : regularizer object whose `.constructor` builds `∇R`.

Returns
- `RegularizedDecompositionProblem{Float32,CuMatrix{Float32},CuVector{Float32}}`.
"""
function RegularizedDecompositionProblemCUDA(
    dli_images::DLI,
    μ₁::μ,
    μ₂::μ,
    λ::Float64,
    background_mask::HyperRectangle{2,Int},
    reg::Regularization,
)
    @assert CUDA.has_cuda() "CUDA.jl is not available or no compatible device found."
    nrows, ncols = size(dli_images.top)
    N = nrows * ncols

    nrows32, ncols32, N32 = Int32(nrows), Int32(ncols), Int32(N)

    M = CuMatrix(Float32.([μ₁.low μ₂.low; μ₁.high μ₂.high]))
    L = CuVector(Float32.(vcat(vec(dli_images.top), vec(dli_images.bottom))))

    initial_guess_result = material_decomposition(dli_images, μ₁, μ₂)
    initial_guess_vec = CuVector(
        Float32.(vcat(vec(initial_guess_result.mat1), vec(initial_guess_result.mat2))),
    )

    prototype = CuVector(zeros(Float32, 2N32))
    A = A_op_gpu_cuda(N32, M)
    Aᵀ = At_op_gpu_cuda(N32, M)
    V⁻¹ = Vinv_op_gpu_cuda(dli_images, background_mask)
    ∇R = reg.constructor((N32, nrows32, ncols32, nothing, reg.params), prototype)

    return RegularizedDecompositionProblem{Float32,CuMatrix{Float32},CuVector{Float32}}(
        dli_images,
        μ₁,
        μ₂,
        Float32(λ),
        background_mask,
        nrows32,
        ncols32,
        N32,
        M,
        L,
        initial_guess_vec,
        prototype,
        reg.name,
        A,
        Aᵀ,
        V⁻¹,
        ∇R,
    )
end


"""
    RegularizedDecompositionProblemMetal(dli_images, μ₁, μ₂, λ, background_mask, reg)

Build a `RegularizedDecompositionProblem` using Metal-backed types (`MtlMatrix`, `MtlVector`).

Arguments
- `dli_images::DLI` : dual-energy images with `.top` and `.bottom` fields.
- `μ₁::μ`, `μ₂::μ` : material attenuation descriptors with `.low` and `.high` fields.
- `λ::Float64` : regularization weight.
- `background_mask::HyperRectangle{2,Int}` : mask used to construct `V⁻¹`.
- `reg::Regularization` : regularizer object whose `.constructor` builds `∇R`.

Returns
- `RegularizedDecompositionProblem{Float32,MtlMatrix{Float32},MtlVector{Float32}}`.
"""
function RegularizedDecompositionProblemMetal(
    dli_images::DLI,
    μ₁::μ,
    μ₂::μ,
    λ::Float64,
    background_mask::HyperRectangle{2,Int},
    reg::Regularization,
)
    @assert Metal.device() !== nothing "Metal.jl is not available or no compatible device found."
    nrows, ncols = size(dli_images.top)
    N = nrows * ncols

    nrows32, ncols32, N32 = Int32(nrows), Int32(ncols), Int32(N)

    M = MtlMatrix(Float32.([μ₁.low μ₂.low; μ₁.high μ₂.high]))
    L = MtlVector(Float32.(vcat(vec(dli_images.top), vec(dli_images.bottom))))

    initial_guess_result = material_decomposition(dli_images, μ₁, μ₂)
    initial_guess_vec = MtlVector(
        Float32.(vcat(vec(initial_guess_result.mat1), vec(initial_guess_result.mat2))),
    )

    prototype = MtlVector(zeros(Float32, 2N32))
    A = A_op_gpu_metal(N32, M)
    Aᵀ = At_op_gpu_metal(N32, M)
    V⁻¹ = Vinv_op_gpu_metal(dli_images, background_mask)
    ∇R = reg.constructor((N32, nrows32, ncols32, nothing, reg.params), prototype)

    return RegularizedDecompositionProblem{Float32,MtlMatrix{Float32},MtlVector{Float32}}(
        dli_images,
        μ₁,
        μ₂,
        Float32(λ),
        background_mask,
        nrows32,
        ncols32,
        N32,
        M,
        L,
        initial_guess_vec,
        prototype,
        reg.name,
        A,
        Aᵀ,
        V⁻¹,
        ∇R,
    )
end

"""
    RegularizedDecomposition(prob; solver=KrylovJL_CG())

Solve the following regularized linear material decomposition problem:

Aᵀ * V⁻¹ * A + λ * ∇R = Aᵀ * V⁻¹ * L
for the given `RegularizedDecompositionProblem` instance.

Arguments
- `prob::RegularizedDecompositionProblem` : problem container with operators and data.
- `solver` : optional Krylov solver instance (default `KrylovJL_CG()` conjugate gradient).
- `reltol::Float64=1e-6` : relative tolerance for the linear solver.
- `verbose::Bool=true` : print solver information at each iteration.
- `return_stats::Val{RS}=Val(false)` : if `Val(true)`, return solver statistics along with the result. You need to write explicitly return_stats=Val(true) and not just return_stats=true.

Behavior
- Assembles the system matrix `Aᵀ * V⁻¹ * A + λ * ∇R` and the right term `Aᵀ * V⁻¹ * L`.
- Wraps the system into a `LinearProblem{true}` and calls `init(...); solve!`.
- Returns a material images `MI` struct with `mat1` and `mat2` reshaped to `(nrows, ncols)`.
"""
function RegularizedDecomposition(
    prob::RegularizedDecompositionProblem;
    solver = KrylovJL_CG(),
    reltol = 1e-6,
    verbose::Bool = false,
    return_stats::Val{RS} = Val(false), #use this syntax to keep the function type stable
) where {RS}
    # Assemble linear operator Aᵀ * V⁻¹ * A + λ ∇R
    A = prob.A
    Aᵀ = prob.Aᵀ
    V⁻¹ = prob.V⁻¹
    ∇R = prob.∇R  #it is actually ∇R/2 in the equation. The ∇R are alrready devied by 2. 
    λ = prob.λ

    # Left-hand side: A = Mᵀ * V⁻¹ * M + λ ∇R
    system_matrix = cache_operator(Aᵀ * V⁻¹ * A + λ * ∇R, prob.L)

    # Right-hand side: L2 = Mᵀ * V⁻¹ * L
    L2 = Aᵀ * V⁻¹ * prob.L

    # Linear problem ----------------------------------------------------------
    linear_problem = LinearProblem{true}(system_matrix, L2; u0 = prob.initial_guess) # true = isinplace
    linear_solve = init(
        linear_problem,
        solver,
        reltol = reltol,
        callback = callback,
        verbose = verbose,
    )
    solve!(linear_solve)

    @assert linear_solve.cacheval.stats.solved "Linear solver did not converge."

    # Extract results ---------------------------------------------------------
    u = linear_solve.u
    mat1 = reshape(u[1:prob.N], (prob.nrows, prob.ncols)) |> Matrix{Float64}
    mat2 = reshape(u[prob.N+1:end], (prob.nrows, prob.ncols)) |> Matrix{Float64}

    material_images = MI(prob.μ₁, prob.μ₂, mat1, mat2)

    if RS
        return (material_images, linear_solve.cacheval.stats)
    else
        return material_images
    end
end
