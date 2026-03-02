
"""
Create N samples of eigenvalues (σ, ω) from a Gaussian mixture.

Arguments
- μf::Vector{Float64}: modal frequencies (Hz).
- μζ::Vector{Float64}: damping ratios.
- Σs::Vector{Matrix{Float64}}: covariance matrices for each mixture component (2×2).
- weights::Vector{Float64}: mixture weights (should sum ≈ 1.0).
- N::Int64: number of samples to generate.
- constrain::Bool: if true, only accept samples inside σrange and ωrange.
- σrange::Tuple{Float64,Float64}=(-4e3,0): allowed range for σ (real part).
- ωrange::Tuple{Float64,Float64}=(0.0,8e2): allowed range for ω (angular frequency).

Returns
- Vector{NTuple{2,Float64}}: vector of (σ, ω) pairs.
"""
function create_eigenvalue_distribution(
        μf::Vector{Float64}, μζ::Vector{Float64}, Σs::Vector{Matrix{Float64}}, weights::Vector{Float64}, N::Int64, constrain::Bool; 
        σrange=(-4e3,0), ωrange=(0.0,8e2)
    )

    @assert sum(weights) ≈ 1.0 "Weights must sum to 1.0"
    # creating means for frequencies and damping ratios
    μω = 2π .* μf
    μσ = vcat(-μζ' .* μω...)
    μω = vcat(μω..., μω...)
    μs = [[μσ, μω] for (μσ, μω) in zip(μσ, μω)]

    # creating the multivariate normal distributions
    comps = [MvNormal(μ, Symmetric(Σ)) for (μ,Σ) in zip(μs, Σs)]
    mix   = MixtureModel(comps, weights)
    
    pts = Vector{NTuple{2,Float64}}()
    while length(pts) < N
        σ, ω = rand(mix)
        if constrain
            if σrange[1] ≤ σ ≤ σrange[2] && ωrange[1] ≤ ω ≤ ωrange[2]
                push!(pts, (σ, ω))
            end
        else
            push!(pts, (σ, ω))
        end
    end
    return pts
end

"""
biased_cluster_sample(λs::Vector{Tuple{Float64, Float64}}, n::Float64; k::Int64=24, γ::Float64=0.7, rng=Random.GLOBAL_RNG)

Sample indices from a set of 2D points using k-means clusters with bias to remain in the same cluster.

Arguments
- λs::Vector{Tuple{Float64, Float64}}: vector of 2D points (pairs of Float64).
- n::Float64: number of indices to sample.
- k::Int64=24: number of clusters for k-means.
- γ::Float64=0.7: probability to stay in the current cluster (higher => more similar consecutive picks).
- rng: random number generator (defaults to Random.GLOBAL_RNG).

Returns
- Vector{Int}: sampled indices of length n.
"""
function biased_cluster_sample(λs::Vector{Tuple{Float64, Float64}}, n::Float64; k::Int64=24, γ::Float64=0.7, rng=Random.GLOBAL_RNG)
    # higher γ, more similar picks
    X = hcat([λ[1] for λ in λs], [λ[2] for λ in λs])'
    km = kmeans(X, k; maxiter=200)
    labels = assignments(km)                   
    buckets = [Int[] for _ in 1:k]

    # fill each cluster bucket with indices of the points
    for (i, c) in enumerate(labels)
        push!(buckets[c], i)
    end

    sizes = map(length, buckets)
    # categorical over clusters (by size)
    cum = cumsum(sizes) ./ sum(sizes)
    function pick_cluster()
        u = rand(rng)
        searchsortedfirst(cum, u)
    end

    idxs = Int[]
    c = pick_cluster()  
    while length(idxs) < n
        
        if rand(rng) > γ
            # pick a new cluster
            c = pick_cluster()
        end

        # draw one from current cluster
        if !isempty(buckets[c])
            push!(idxs, rand(rng, buckets[c]))
        else
            c = pick_cluster()
        end
    end
    idxs
end

"""
Create a complex-conjugate poles from a real-imaginary pair.

Arguments
- ev::Tuple{Float64, Float64}: (σ, ω) where σ is the real part and ω is the imaginary frequency.

Returns
- Vector{Complex{Float64}}: two poles [σ + im*ω, σ - im*ω].
"""
function create_compconj_poles(ev::Tuple{Float64, Float64})
    σ, ω = ev
    return [σ + im * ω, σ - im * ω]
end

"""
Create a pair of complex-conjugate poles.

Arguments
- ω::Float64: Imaginary part (frequency).
- σ::Float64: Real part (damping).

Returns
- Vector{Complex{Float64}}: Two-element array [σ + im*ω, σ - im*ω].
"""
function create_compconj_poles(ω::Float64, σ::Float64)
    return [σ + im * ω, σ - im * ω]
end

"""
sample_eigenvalues(λs::Vector{Tuple{Float64, Float64}}, func_order::Int64; allow_unstable::Bool=false)

Sample eigenvalues as real poles or complex-conjugate pole pairs using biased cluster sampling.
For odd func_order an extra real pole (zero-pole) is added based on the mean real part of selected poles.

Arguments:
- λs :: Vector{Tuple{Float64, Float64}} : candidate pole parameters for biased sampling.
- func_order :: Int64 : desired number of poles / function order.
- allow_unstable :: Bool = false : if false, enforce strictly negative real parts for real poles.

Returns:
- Vector{Complex{Float64}} : sampled eigenvalues (real poles have zero imaginary part).
"""
function sample_eigenvalues(λs::Vector{Tuple{Float64, Float64}}, func_order::Int64; allow_unstable::Bool=false)
    if func_order == 1
        if allow_unstable
            evs = [rand(Truncated(Normal(-1, 10.), -5e5, 5.)) + im * 0.0]
        else
            evs = [rand(Truncated(Normal(-1, 10.), -5e5, -0.0001)) + im * 0.0]
        end
    elseif  func_order % 2 != 0
        ev_idxs = biased_cluster_sample(λs, (func_order - 1) / 2)
        evs = [create_compconj_poles(λs[i]) for i in ev_idxs]
        # compute the zero-pole based on average previous poles        
        mean_real = mean(real.(vcat(evs...)))
        if allow_unstable
            zero_pole = [rand(Truncated(Normal(mean_real, 10.), -5e5, 5.)) + im * 0.0]
            push!(evs, zero_pole)
        else
            zero_pole = [rand(Truncated(Normal(mean_real, 10.), -5e5, -0.0001)) + im * 0.0]
            push!(evs, zero_pole)
        end
    else
        ev_idxs = biased_cluster_sample(λs, func_order / 2)
        evs = [create_compconj_poles(λs[i]) for i in ev_idxs]
    end
    return vcat(evs...)
end

# creates a transfer function with the poles defined above
"""
create_transfer_function(poles::Vector{ComplexF64}; dc_gain::Float64=1.0)

Create a continuous-time transfer function from the given complex poles and an optional DC gain.

Arguments
- `poles::Vector{ComplexF64}`: Vector of complex poles (denominator roots).
- `dc_gain::Float64=1.0`: Scalar DC gain to apply to the numerator (default 1.0).

Returns
- Transfer function object representing the system (numerator and normalized denominator).
"""
function create_transfer_function(poles::Vector{ComplexF64}; dc_gain::Float64=1.0)
    s = Polynomial([0, 1])
    p = prod(s .- poles)
    den = reverse(real.(coeffs(p)))
    den = den ./ den[1]  
    num = [dc_gain]
    return tf(num, den)
end

# TODO: tweak distribution parameters
"""
transform_data(y::AbstractArray{Float64}) -> AbstractArray{Float64}

Apply a small random offset and a random frequency deviation to the input array.
The input is transformed to the unit range, scaled by the sampled deviation, and shifted by the offset.

Arguments
- y::AbstractArray{Float64}: input array of Float64 values.

Returns
- AbstractArray{Float64}: transformed array with the same shape as `y`.
"""
function transform_data(y::AbstractArray{Float64})
    inital_offset = rand(Normal(0, 0.005))
    freq_deviation = rand(Truncated(Normal(0.005, 0.05), -0.2, 0.2))
    y = (1 - inital_offset) .+ freq_deviation .* StatsBase.transform(
        StatsBase.fit(UnitRangeTransform, y),
        y,
    )
    return y
end

# if a response is expected to be HF only, we can adjust the time range accordingly
# there is no point in simulating very steep transients
"""
Generate a time vector based on the dominant eigenvalue's decay time.

Arguments
- evs::Vector{ComplexF64}: eigenvalues to determine dominant decay rate
- T::Int: number of time points to generate
- default_t_end::Int=10: fallback maximum end time

Returns
- t::AbstractRange{Float64}: range from 0 to computed end time with T points
"""
function generate_step_range(evs::Vector{ComplexF64}, T::Int; default_t_end::Int=10)
    dominant_ev = evs[argmax(real.(evs))]
    τ = -1 / real(dominant_ev)
    τ_multiplier = rand(Uniform(20, 30))
    t_end = min(τ_multiplier * τ, default_t_end)
    t = range(0, t_end, length=T)
    return t
end

"""
simulate_delayed_step_response(λs::Vector{Tuple{Float64, Float64}}, tfunc_order::Int, T::Int; allow_unstable::Bool=false)

Simulate a delayed step response using sampled eigenvalues and a constructed transfer function.

Arguments
- λs::Vector{Tuple{Float64, Float64}}: candidate eigenvalue pairs (real, imag) for sampling.
- tfunc_order::Int: desired order of the transfer function.
- T::Int: time horizon / length used to generate the time vector.
- allow_unstable::Bool=false: whether to permit unstable eigenvalues during sampling.

Returns
- transformed_response: the simulated output after applying transform_data (type depends on transform_data).
"""
function simulate_delayed_step_response(λs::Vector{Tuple{Float64, Float64}}, tfunc_order::Int, T::Int; allow_unstable::Bool=false)
    evs = sample_eigenvalues(λs, tfunc_order; allow_unstable=allow_unstable)
    tfunc = create_transfer_function(evs)
        
    t = generate_step_range(evs, T)
    u = zeros(length(t))
    dt = t[2] - t[1]
    t_delay = rand(Uniform(minimum(t) + 10 * step(t), maximum(t) - 1000 * step(t)))
    imp = rand(Truncated(Normal(dt, 0.1*dt), 0.5*dt, 1.5*dt)) 
    u[findfirst(x -> x ≥ t_delay, t):lastindex(u)] .= imp / dt
    
    y, _, _ = lsim(tfunc, u', t)
    
    return transform_data(y)
end

"""
simulate_delayed_impulse_response(λs::Vector{Tuple{Float64,Float64}}, tfunc_order::Int, T::Int; allow_unstable::Bool=false)

Simulate a delayed impulse response from a sampled transfer function and return the transformed output.

Arguments
- λs::Vector{Tuple{Float64, Float64}}: eigenvalue ranges (real, imag) used for sampling.
- tfunc_order::Int: order of the transfer function to create.
- T::Int: total simulation time / horizon for the generated time vector.
- allow_unstable::Bool=false: keyword; if true, permit sampling unstable eigenvalues.

Returns
- Vector{Float64}: transformed output time series (result of transform_data applied to the simulated response).
"""
function simulate_delayed_impulse_response(λs::Vector{Tuple{Float64, Float64}}, tfunc_order::Int, T::Int; allow_unstable::Bool=false)
    evs = sample_eigenvalues(λs, tfunc_order; allow_unstable=allow_unstable)
    tfunc = create_transfer_function(evs)
    
    t = generate_step_range(evs, T)
    u = zeros(length(t))
    dt = t[2] - t[1]
    t_delay = rand(Uniform(minimum(t) + 10 * step(t), maximum(t) - 1000 * step(t)))
    imp = rand(Truncated(Normal(dt, 0.1*dt), 0.5*dt, 1.5*dt))  
    u[findfirst(x -> x ≥ t_delay, t)] =  - imp / dt
    
    y, _, _ = lsim(tfunc, u', t)
    
    return transform_data(y)
end

# sample transfer function order from a Poisson distribution
"""
Sample N order sizes from a Poisson distribution truncated to [min_order, max_order].

Arguments
- N::Int64: number of samples to generate
- min_order::Int64=1: minimum acceptable order value (inclusive)
- max_order::Int64=9: maximum acceptable order value (inclusive)
- mean_order::Float64=5.0: mean parameter of the Poisson distribution

Returns
- Vector{Int}: a vector of N sampled order sizes within [min_order, max_order]
"""
function sample_order_poisson(N::Int64, min_order::Int64=1, max_order::Int64=9, mean_order::Float64=5.)
    po = Poisson(mean_order)
    orders = Int[]
    while length(orders) < N
        k = rand(po)
        if min_order <= k <= max_order
            push!(orders, k)
        end
    end
    return orders
end

"""
Generate N delayed step responses.

Arguments
- dargs::Any: configuration object providing parameters used internally (must include T and other fields referenced by the generator).
- N::Int: number of responses to generate.

Returns
- Array{Float32,3}: tensor of shape (dargs.T, 1, N) where each slice [:, 1, i] is a delayed step response.
"""
function generate_delayed_step_responses(dargs, N::Int)
    pts = create_eigenvalue_distribution(dargs.μf, dargs.μζ, dargs.Σs, dargs.weights, dargs.N_eigenvals, true)
    fo = sample_order_poisson(N, dargs.func_min_order, dargs.func_max_order, dargs.func_mean_order)

    delayed_step_responses = zeros(Float32, dargs.T, 1, N)
    @showprogress desc="Generating $(N) delayed step responses" for (i, o) in enumerate(fo)
        delayed_step_responses[:, 1, i] = simulate_delayed_step_response(pts, o, dargs.T, allow_unstable=dargs.allow_unstable)
    end
    return delayed_step_responses
end

"""
generate_delayed_impulse_responses(dargs, N::Int)

Generate N delayed impulse responses by sampling eigenvalue distributions and function orders.

Arguments
- dargs::Any: parameter container (struct or NamedTuple) with required fields:
    μf, μζ, Σs, weights, N_eigenvals, func_min_order, func_max_order, func_mean_order, T, allow_unstable
- N::Int: number of impulse responses to generate

Returns
- Array{Float32,3} of size (dargs.T, 1, N): delayed impulse responses
"""
function generate_delayed_impulse_responses(dargs, N::Int)
    pts = create_eigenvalue_distribution(dargs.μf, dargs.μζ, dargs.Σs, dargs.weights, dargs.N_eigenvals, true)
    fo = sample_order_poisson(N, dargs.func_min_order, dargs.func_max_order, dargs.func_mean_order)

    delayed_impulse_responses = zeros(Float32, dargs.T, 1, N)
    @showprogress desc="Generating $(N) delayed impulse responses" for (i, o) in enumerate(fo)
        delayed_impulse_responses[:, 1, i] = simulate_delayed_impulse_response(pts, o, dargs.T, allow_unstable=dargs.allow_unstable)
    end
    return delayed_impulse_responses
end

"""
generate_training_validation_data(dargs; save_data::Bool=false, training_split::Int=80)

Generate training and validation datasets of delayed impulse and step responses.

Arguments
- dargs::Any — object containing at least N_samples::Number and any parameters required by the internal generator functions.
- save_data::Bool=false — whether to save the generated datasets to disk.
- training_split::Int=80 — percentage of samples allocated to the training set.

Returns
- Dict with keys :training_data and :validation_data. Each value is a Dict containing
    "delayed_impulse_responses" and "delayed_step_responses".
"""
function generate_training_validation_data(dargs; save_data::Bool=false, training_split::Int=80)
    N = Int(dargs.N_samples * (training_split / 100))
    step_res = generate_delayed_step_responses(dargs, N)
    imp_res = generate_delayed_impulse_responses(dargs, N)
    if save_data
        jldsave(
            "../data/training_data.jld2",
            delayed_impulse_responses=imp_res,
            delayed_step_responses=step_res,
        )
    end

    N_val = Int(dargs.N_samples - N)
    step_res_val = generate_delayed_step_responses(dargs, N_val)
    imp_res_val = generate_delayed_impulse_responses(dargs, N_val)
    if save_data
        jldsave(
            "../data/validation_data.jld2",
            delayed_impulse_responses=imp_res_val,
            delayed_step_responses=step_res_val,
        )
    end

    return Dict(
        :training_data => Dict(
            "delayed_impulse_responses" => imp_res,
            "delayed_step_responses" => step_res,
        ),
        :validation_data => Dict(
            "delayed_impulse_responses" => imp_res_val,
            "delayed_step_responses" => step_res_val,
        )
    )
end

