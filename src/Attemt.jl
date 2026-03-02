module Attemt

using Flux
using NNlib
using CUDA
using cuDNN
using Zygote
using ChainRulesCore
using Optimisers
using MLUtils
using OneHotArrays
using ProgressMeter
using Statistics
using Random
using JLD2
using CSV
using DataFrames
using ControlSystems
using DSP
using Distributions
using StatsBase
using MultivariateStats
using Statistics
using Interpolations
using LinearAlgebra
using Polynomials
using Clustering
using Random
using AbstractFFTs


# for package testing
using Plots
using BenchmarkTools

function greet()
    println("Hello from the attemt module!")
end

dev = Flux.get_device()

# unfortunately we have to use gloabal variables for that...
const IS_TRAINING = Ref{Bool}(false)
is_training() = IS_TRAINING[]
set_training!(x::Bool) = (IS_TRAINING[] = x)

include("./params.jl")
include("./utils.jl")

include("./VQVAE_utils.jl")

include("./models/model_1D/mr_stft.jl")
include("./models/model_1D/residuals.jl")
include("./models/model_1D/dilated_residuals.jl")

include("./models/model_1D/vanilla_encoders.jl")
include("./models/model_1D/vanilla_decoders.jl")
include("./models/model_1D/lowpass_layers.jl")
include("./models/model_1D/lp_encoders.jl")
include("./models/model_1D/dilp_encoders.jl")
include("./models/model_1D/lp_decoders.jl")

include("./models/model_1D/VQ.jl")
include("./models/model_1D/VQVAE.jl")

include("./data_provider.jl")
include("./data_generation/data_generation.jl")
include("./data_generation/synthetic_data_generation.jl")
include("./VQVAE_training.jl")



export greet
export SharedArgs, VQVAEArgs, DataGenerationArgs
export dev
export IS_TRAINING, is_training, set_training!
export Flux, Optimisers, OptimiserChain

export MRSTFTArgs, MRSTFT
export VQVAEModel, train_VQVAE, run_VQVAE_training, fancy_train_VQVAE, run_fancy_VQVAE_training
export generate_training_validation_data, load_training_data, load_validation_data
export get_minibatch, tokenize_input, tokenize_fast, decode_output, tokenize_dataset, 
export manual_normalization, manual_denormalization


# data generation
export create_eigenvalue_distribution, biased_cluster_sample, create_compconj_poles, sample_eigenvalues, create_transfer_function, transform_data


end
