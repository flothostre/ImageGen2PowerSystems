# shared model parameters for VQVAE and forecaster
Base.@kwdef mutable struct SharedArgs
    embedding_dim::Int = 128
    num_embeddings::Int = 512
    compression_factor::Int = 16

    synthetic_data_only::Bool = false

    filename::String = ""
end

Base.@kwdef mutable struct VQVAEArgs
    shared::SharedArgs = SharedArgs()

    epochs::Int = 48
    batch_size::Int = 256

    learning_rate::Float32 = 2e-4
    learning_rate_decay::Float32 = 0.01
    commitment_cost::Float32 = 0.2
    grad_clip::Float32 = 1
    data_variance::Float32 = 1.0
    decay::Float32 = 0.99 # if not zero --> use EMA

    pre_vq_layer_norm::Bool = true
    input_length::Int = 2048  

    kernel_size_ds::Int = 9
    kernel_size_res::Int = 5

    apply_enc_lpf::Bool = true
    enc_lpf_taps::Int = 63
    enc_kaiser_beta::Float64 = 8.0
    enc_cutoff_factor::Float64 = 0.9

    apply_denc_lpf::Bool = true
    denc_lpf_taps::Int = 63
    denc_kaiser_beta::Float64 = 8.0
    denc_cutoff_factor::Float64 = 0.9

    dilated_residuals::Bool = false
    dires_contribution_factor::Float64 = 0.10

    in_channels::Int = 1
    num_hiddens::Int = 128
    num_residual_layers::Int = 2
    num_residual_hiddens::Int = 32
    
    save_checkpoints::Bool = true
    compute_perplexity::Bool = false
end

Base.@kwdef mutable struct MRSTFTArgs
    use_mrstft::Bool = true
    contribution_factor::Float32 = 0.5 # scaling for λ terms
    λ_sc::Float32 = 0.5
    λ_mag::Float32 = 0.5
    fft_sizes::Vector{Int} = [128,256,512]
    hop_factor::Int = 4
end

Base.@kwdef mutable struct DataGenerationArgs
    shared::SharedArgs = SharedArgs()

    # pole placement args
    N_eigenvals::Int = 50000
    # expected mean centers of frequencies (in Hz)
    μf::Vector{Float64}  = [0.2, 0.5, 0.8, 30.0, 200., 1000., 3000.]
    # expected mean centers of damping ratios (under- and overdamped)
    μζ::Vector{Float64} = [0.7, 1.2]

    # covariance matrices
    # damping variance
    var_damping::Float64 = 2e1
    # frequency variance
    var_freq::Float64 = 2e1
    data_tilt::Float64 = 1.
    Σs::Vector{Matrix{Float64}} = [
        # underdamped
        # 0.2 Hz oscillation
        [1e1*var_damping  data_tilt;
        data_tilt  1e0*var_freq],
        # 0.5 Hz oscillation
        [1e2*var_damping  data_tilt;
        data_tilt  1e1*var_freq],
        # 0.8 Hz oscillation
        [1e3*var_damping  data_tilt;
        data_tilt  1e1*var_freq],
        # 30 Hz oscillation
        [3e4*var_damping  data_tilt;
        data_tilt  1e3*var_freq],
        # 200 Hz oscillation
        [3e5*var_damping  data_tilt;
        data_tilt  1e4*var_freq],
        # 2000 Hz oscillation
        [1e6*var_damping  data_tilt;
        data_tilt  1e5*var_freq],
        # 5000 Hz oscillation
        [1e7*var_damping  data_tilt;
        data_tilt  1e6*var_freq],
        
        # overdamped
        # 0.2 Hz oscillation
        [1e1*var_damping  data_tilt;
        data_tilt  1e0*var_freq],
        # 0.5 Hz oscillation
        [1e2*var_damping  data_tilt;
        data_tilt  1e1*var_freq],
        # 0.8 Hz oscillation
        [1e3*var_damping  data_tilt;
        data_tilt  1e1*var_freq],
        # 30 Hz oscillation
        [3e4*var_damping  data_tilt;
        data_tilt  1e3*var_freq],
        # 200 Hz oscillation
        [3e5*var_damping  data_tilt;
        data_tilt  1e4*var_freq],
        # 2000 Hz oscillation
        [1e6*var_damping  data_tilt;
        data_tilt  1e5*var_freq],
        # 5000 Hz oscillation
        [1e7*var_damping  data_tilt;
        data_tilt  1e6*var_freq],
    ]
    # weight vector of the frequency modes
    weights::Vector{Float64} = [
        .03575, .03575, # 0.2 Hz
        .0784, .0784, # 0.5 Hz
        .03575, .03575, # 0.8 Hz
        .0714, .0714, # 30 Hz
        .0929, .0929, # 200 Hz
        .0929, .0929, # 1000 Hz
        .0929, .0929, # 3000 Hz
        ] 

    # simulation_timesteps
    T::Int = 2048
    default_t_end::Int = 10

    allow_unstable::Bool = false

    func_min_order::Int = 1
    func_max_order::Int = 9
    func_mean_order::Float64 = 5.

    N_samples::Int = 250000  # for step and impulse response each

end
