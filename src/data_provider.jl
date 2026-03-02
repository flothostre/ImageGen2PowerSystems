"""
Load training data from file or generate if missing.

Arguments
- dargs::Any: parameters for data generation.
- filename::String="./data/training_data.jld2": path to the JLD2 file.

Returns
- Dict{String,Any}: keys "delayed_impulse_responses" and "delayed_step_responses".
"""
function load_training_data(dargs; filename::String="./data/training_data.jld2")
    data = if isfile(filename)
        @info "Loaded training data from $(filename)"
        JLD2.load(filename)
    else
        @warn "File $(filename) not found. Generating new data."
        generate_training_validation_data(dargs, save_data=true)[:training_data]
    end

    return Dict(
        "delayed_impulse_responses" => data["delayed_impulse_responses"],
        "delayed_step_responses" => data["delayed_step_responses"]
    )
end

"""
Load validation data from a JLD2 file or generate it if the file is missing.

Arguments
- dargs::Any: Arguments forwarded to the data generation routine when the file is not found.
- filename::String="./data/validation_data.jld2": Path to the JLD2 file to load.

Returns
- Dict{String,Any}: Dictionary with keys "delayed_impulse_responses" and "delayed_step_responses".
"""
function load_validation_data(dargs; filename::String="./data/validation_data.jld2")
    data = if isfile(filename)
        @info "Loaded validation data from $(filename)"
        JLD2.load(filename)
    else
        @warn "File $(filename) not found. Generating new data."
        generate_training_validation_data(dargs, save_data=true)[:validation_data]
    end

    return Dict(
        "delayed_impulse_responses" => data["delayed_impulse_responses"],
        "delayed_step_responses" => data["delayed_step_responses"]
    )
end

"""
    get_minibatch(data::Dict; B::Int=128)

Creates a minibatch data loader from the provided data dictionary.

# Arguments
- `data::Dict`: A dictionary containing the following keys, each mapping to an array:
    - `"function_evaluations"`
    - `"impulse_responses"`
    - `"step_responses"`
    - `"delayed_impulse_responses"`
    - `"delayed_step_responses"`
- `B::Int=128`: (Optional) The batch size for the data loader.

# Returns
- `data_loader`: An `MLUtils.DataLoader` object that yields normalized minibatches of the concatenated data along the third dimension, with shuffling and parallel loading enabled.
"""
function get_minibatch(data::Dict; B::Int=128)
    data_cat = cat(
        [data[k] for k in keys(data)]...;
        dims=3
        )
    data_loader = MLUtils.DataLoader(
        data_cat,
        batchsize=B,
        shuffle=true,
        parallel=true
        )
    return data_loader  
end

"""
get_minibatch(data_cat::AbstractArray{Float32,3}; B::Int=128)

Create an MLUtils.DataLoader that yields minibatches from a 3‑D Float32 array.

Arguments
- data_cat::AbstractArray{Float32,3}: Input array with samples along the 3rd dimension (size (d1, d2, N)).
- B::Int=128: Batch size.

Returns
- MLUtils.DataLoader: A data loader producing minibatches of up to B samples, with shuffling and parallel loading enabled.
"""
function get_minibatch(data_cat::AbstractArray{Float32, 3}; B::Int=128)
    data_loader = MLUtils.DataLoader(
        data_cat,
        batchsize=B,
        shuffle=true,
        parallel=true
        )
    return data_loader  
end

"""
    get_specific_minibatch(data::Dict, specific_set::String; B::Int=128)

Creates a minibatch data loader for a specified dataset within the provided dictionary.

# Arguments
- `data::Dict`: A dictionary containing datasets, where each key is a set name and each value is the corresponding data.
- `specific_set::String`: The key identifying which dataset to use from `data`.
- `B::Int=128`: (Optional) The batch size for the data loader. Defaults to 128.

# Returns
- An `MLUtils.DataLoader` object that yields normalised minibatches from the specified dataset.
"""
function get_specific_minibatch(data::Dict, specific_set::String; B::Int=128)
    data_loader = MLUtils.DataLoader(
        data[specific_set],
        batchsize=B,
        shuffle=true,
        parallel=true
    )
    return data_loader  
end

"""
    tokenize_dataset(data::Dict, vqvae_model::VQVAEModel, args::VQVAEArgs, dev; batchsize::Int=256)

Tokenizes a dataset using a provided VQ-VAE model.

# Arguments
- `data::Dict`: A dictionary containing the dataset, where each value is expected to be an array.
- `vqvae_model::VQVAEModel`: The VQ-VAE model used for tokenization.
- `args::VQVAEArgs`: The arguments for the VQ-VAE model.
- `dev`: The device (e.g., `cpu` or `gpu`) to which data batches are moved for processing.
- `batchsize::Int=256`: (Optional) The number of samples per batch during tokenization.

# Returns
- A matrix of tokenized representations, concatenated along the second dimension.
"""
function tokenize_dataset(data::Dict, vqvae_model::VQVAEModel, args::VQVAEArgs, dev; batchsize::Int=256)
    X_cat = cat(
        [data[k] for k in keys(data)]...;
        dims=3
    ) 
    _, _, N_total = size(X_cat)
    encoded_list = []

    for i in 1:batchsize:N_total
        idx_end = min(i + batchsize - 1, N_total)
        batch = X_cat[:, :, i:idx_end] |> dev  
        encoded_batch = tokenize_fast(vqvae_model, batch, args)
        push!(encoded_list, encoded_batch |> cpu)
    end

    return hcat(encoded_list...)
end

"""
tokenize_dataset(X_cat::AbstractArray{Float32,3}, vqvae_model::VQVAEModel, args::VQVAEArgs, dev; batchsize::Int=256)

Tokenize a 3D batch of inputs using a VQ-VAE model in minibatches.

Arguments
- X_cat::AbstractArray{Float32,3}: input tensor with shape (channels, length, N_samples)
- vqvae_model::VQVAEModel: VQ-VAE model used to encode inputs
- args::VQVAEArgs: model/configuration arguments for tokenization
- dev: device to move batches to (e.g. GPU/CPU)
- batchsize::Int=256: optional batch size for processing

Returns
- AbstractArray{<:Integer,2}: concatenated encoded token indices for all samples (tokens × N_samples)
"""
function tokenize_dataset(X_cat::AbstractArray{Float32,3}, vqvae_model::VQVAEModel, args::VQVAEArgs, dev; batchsize::Int=256)
    _, _, N_total = size(X_cat)
    encoded_list = []

    for i in 1:batchsize:N_total
        idx_end = min(i + batchsize - 1, N_total)
        batch = X_cat[:, :, i:idx_end] |> dev  
        encoded_batch = tokenize_fast(vqvae_model, batch, args)
        push!(encoded_list, encoded_batch |> cpu)
    end

    return hcat(encoded_list...)
end


"""
    get_minibatch(X::AbstractArray, Y::AbstractArray; B::Int=128)

Creates a minibatch data loader from input arrays `X` and `Y` using `MLUtils.DataLoader` .

# Arguments
- `X::AbstractArray`: Input features array of tokens.
- `Y::AbstractArray`: Target labels array, tokens expected in `onehotbatch` format.
- `B::Int=128`: (Optional) Batch size for the data loader.

# Returns
- An `MLUtils.DataLoader` object that yields minibatches of `(X, Y)` pairs, shuffled and loaded in parallel.
"""
function get_minibatch(X::AbstractArray, Y::AbstractArray; B::Int=128)
    data_loader = MLUtils.DataLoader(
        (X, Y);
        batchsize=B,
        shuffle=true,
        parallel=true
    )
    return data_loader
end

# "normalizes" to around 0 and keeps interval in [-2, 2] (fixed values)
"""
Normalize a 3D Float32 array by subtracting a mean and dividing by a std.

Arguments
- x::AbstractArray{Float32,3}: input 3D array
- mean::Float32 = Float32(1.0): mean value to subtract
- std::Float32 = Float32(0.05): standard deviation to divide by

Returns
- AbstractArray{Float32,3}: normalized array with same shape and element type
"""
function manual_normalization(x::AbstractArray{Float32,3}; mean::Float32=Float32(1.), std::Float32=Float32(.05))
    return (x .- mean) ./ std
end

"""Normalize concatenated 3D Float32 arrays from a Dict.

Arguments:
- data::Dict{String, Array{Float32,3}}: mapping of names → 3D Float32 arrays.
- mean::Float32=1.0: value to subtract.
- std::Float32=0.05: value to divide by.

Returns:
- Array{Float32,3}: concatenated along dim 3 and normalized.
"""
function manual_normalization(data::Dict{String, Array{Float32, 3}}; mean::Float32=Float32(1.), std::Float32=Float32(.05))
    data_cat = cat(
        [data[k] for k in keys(data)]...;
        dims=3
        )
    return (data_cat .- mean) ./ std
end

"""
manual_denormalization(x::AbstractArray{Float32,3}; mean::Float32=1.0f0, std::Float32=0.05f0)

Denormalize a 3‑D Float32 array by scaling with `std` and adding `mean`.

Arguments
- x::AbstractArray{Float32,3}: Normalized input tensor.
- mean::Float32=1.0f0: Mean to add after scaling.
- std::Float32=0.05f0: Standard deviation used to scale `x`.

Returns
- AbstractArray{Float32,3}: Denormalized array with the same shape as `x`.
"""
function manual_denormalization(x::AbstractArray{Float32,3}; mean::Float32=Float32(1.), std::Float32=Float32(.05))
    return (x .* std) .+ mean
end

"""
manual_denormalization(x::AbstractArray{Float32,3}; mean::Float32=1.0f0, std::Float32=0.05f0)

Denormalize a 3‑D Float32 array by scaling with `std` and adding `mean`.

Arguments
- x::AbstractArray{Float32,3}: Normalized input tensor.
- mean::Float32=1.0f0: Mean to add after scaling.
- std::Float32=0.05f0: Standard deviation used to scale `x`.

Returns
- AbstractArray{Float32,3}: Denormalized array with the same shape as `x`.
"""
function manual_denormalization(x::AbstractArray{Float32,1}; mean::Float32=Float32(1.), std::Float32=Float32(.05))
    return (x .* std) .+ mean
end

"""
manual_denormalization(x::AbstractArray{Float32,3}; mean::Float32=1.0f0, std::Float32=0.05f0)

Denormalize a 3‑D Float32 array by scaling with `std` and adding `mean`.

Arguments
- x::AbstractArray{Float32,3}: Normalized input tensor.
- mean::Float32=1.0f0: Mean to add after scaling.
- std::Float32=0.05f0: Standard deviation used to scale `x`.

Returns
- AbstractArray{Float32,3}: Denormalized array with the same shape as `x`.
"""
function manual_denormalization(x::AbstractMatrix{Float32}; mean::Float32=Float32(1.), std::Float32=Float32(.05))
    return (x .* std) .+ mean
end

"""
manual_denormalization(data::Dict{String, Array{Float32, 3}}; mean::Float32=Float32(1.), std::Float32=Float32(.05))

Denormalize a 3‑D Float32 array by scaling with `std` and adding `mean`.

Arguments
- data::Dict{String, Array{Float32,3}}: mapping of names → 3D Float32 arrays.
- mean::Float32=1.0f0: Mean to add after scaling.
- std::Float32=0.05f0: Standard deviation used to scale `x`.

Returns
- AbstractArray{Float32,3}: Denormalized array with the same shape as `x`.
"""
function manual_denormalization(data::Dict{String, Array{Float32, 3}}; mean::Float32=Float32(1.), std::Float32=Float32(.05))
    data_cat = cat(
        [data[k] for k in keys(data)]...;
        dims=3
        )
    return (data_cat .* std) .+ mean
end
