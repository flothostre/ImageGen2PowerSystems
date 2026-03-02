abstract type VQ end

struct VectorQuantizer <: VQ
    num_embeddings::Int
    embedding_dim::Int
    commitment_cost::Float32
    embedding::Embedding  # Codebook of size (embedding_dim x num_embeddings)
end

"""
    VectorQuantizer(num_embeddings::Int, embedding_dim::Int, commitment_cost::Float32)

Creates a vector quantizer module for use in a Vector Quantized Variational Autoencoder (VQ-VAE).

# Arguments
- `num_embeddings::Int`: The number of discrete embeddings in the codebook.
- `embedding_dim::Int`: The latent dimensionality of each embedding vector.
- `commitment_cost::Float32`: The weighting factor for the commitment loss

# Returns
A vector quantizer object that can be used to quantize continuous latent representations into discrete embeddings.
"""
function VectorQuantizer(num_embeddings::Int, embedding_dim::Int, commitment_cost::Float32)
    embedding = Embedding(
        embedding_dim => num_embeddings; # TODO: (maybe) change that so it matches transformer emb dims
        init=Flux.glorot_uniform(gain=1.0) # uniform dist ~ [-.1, .1]
        ) 
    return VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, embedding)
end

"""
    (::VectorQuantizer)(inputs::AbstractArray{Float32, 3}, args::VQVAEArgs)

Applies the vector quantization operation to the given 3-dimensional array of `Float32` values.
Implemented as callable object for the forward pass

# Arguments
- `inputs::AbstractArray{Float32, 3}`: A 3-dimensional array of type `Float32` representing the input data to be quantized.
- `args::VQVAEArgs`: An instance of `VQVAEArgs` containing configuration parameters for the VQ-VAE model.

# Returns
The quantized representation of the input data.
"""
function (vq::VectorQuantizer)(inputs::AbstractArray{Float32, 3}, args::VQVAEArgs)
    # julia: data comes in TCB format, we need to do TCB -> BTC (T: time dimension)
    inputs = permutedims(inputs, (3, 1, 2)) 
    input_shape = size(inputs)

    # Flatten input to (batch_size * height * width, embedding_dim)
    flat_input = reshape(inputs, :, vq.embedding_dim)  

    # Calculate matrix of euklidian distances between input and embeddings
    distances = (sum(flat_input.^2, dims=2) .+ sum(vq.embedding.weight.^2, dims=2)' .- 2 .* (flat_input * vq.embedding.weight'))

    # Find nearest embedding index, returns vector of minimal distance for each row packed into OneHotArrays.onehotbatch
    encodings = nothing  
    if CUDA.has_cuda_gpu()
        int_encoding_indices = gpu_argmin_rows(distances)  
        ChainRulesCore.@ignore_derivatives encodings = permutedims(
            OneHotArrays.onehotbatch(int_encoding_indices, 1:vq.num_embeddings),
            (2,1)
        )  
    else
        encoding_indices = argmin(distances, dims=2)  
        int_encoding_indices = vec(getindex.(encoding_indices, 2))  
        ChainRulesCore.@ignore_derivatives encodings = permutedims(
            OneHotArrays.onehotbatch(int_encoding_indices, 1:vq.num_embeddings),
            (2,1)
        )
    end    

    # Map one-hot encodings to embedding weight matrix and unflatten (= quantization)
    quantized = reshape(encodings * vq.embedding.weight, input_shape) 

    # Codebook loss, moves embedding vectors to encoder outputs --> gradient only flows to VQ embedding table
    codebook_loss = Flux.mse(quantized, stopgrad(inputs))
    # Commitment loss, makes sure encoder stays close to the codebook --> gradient only flows to encoder
    commitment_loss = Flux.mse(stopgrad(quantized), inputs)
    vq_loss = args.commitment_cost * commitment_loss + codebook_loss

    # Straight-through estimator for backpropagation (for gradient flow)
    quantized = straight_through(quantized, inputs)

    # Perplexity, a measure of how well the quantizer is performing
    if args.compute_perplexity
        avg_probs = sum(encodings, dims=2) ./ size(encodings, 2)  
        perplexity = exp(-sum(avg_probs .* log.(avg_probs .+ 1e-10)))
    else
        perplexity = 0.  
    end

    # Convert quantized from BTC -> TCB
    quantized = permutedims(quantized, (2, 3, 1))  

    return quantized, perplexity, encodings, vq_loss
end


struct VectorQuantizerEMA <: VQ
    num_embeddings::Int
    embedding_dim::Int
    commitment_cost::Float32
    embedding::Embedding
    decay::Float32
    ϵ::Float32
    ema_cluster_size::AbstractArray{Float32}  # this variable should be stored in to "output", but not trained by the optimizer
    ema_w::AbstractArray{Float32}
end

"""
    VectorQuantizerEMA(num_embeddings::Int, embedding_dim::Int, commitment_cost::Float32, decay::Float32; ϵ::Float32=1e-5f0)

Constructs a Vector Quantizer with Exponential Moving Average (EMA) updates for embeddings.

# Arguments
- `num_embeddings::Int`: Number of discrete embedding vectors.
- `embedding_dim::Int`: Dimensionality of each embedding vector.
- `commitment_cost::Float32`: Weight for the commitment loss term.
- `decay::Float32`: Decay rate for the EMA updates.
- `ϵ::Float32=1e-5f0`: Small constant for numerical stability (optional, default is `1e-5f0`).

# Returns
A `VectorQuantizerEMA` object initialized with the specified parameters.
"""
function VectorQuantizerEMA(num_embeddings::Int, embedding_dim::Int, commitment_cost::Float32, decay::Float32; ϵ::Float32=Float32(1e-5))
    embedding = Embedding(
        embedding_dim => num_embeddings;
        init=Flux.glorot_uniform(gain=2.0) # uniform dist ~ [-.2, .2]
        )
    ema_cluster_size = zeros(Float32, num_embeddings)
    ema_w = randn(Float32, num_embeddings, embedding_dim)
    return VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, embedding, decay, ϵ, ema_cluster_size, ema_w)
end

"""
    (vq::VectorQuantizerEMA)(inputs::AbstractArray{Float32, 4}; training::Bool=true)

Applies the Exponential Moving Average (EMA) Vector Quantization to the input tensor.

# Arguments
- `inputs::AbstractArray{Float32, 4}`: Input tensor of shape (W, H, C, B), where W = width, H = height, C = channels, B = batch size.
- `training::Bool=true`: Flag indicating whether the model is in training mode. If `true`, EMA updates are performed.

# Returns
- `quantized`: Quantized tensor of the same shape as input, with values replaced by nearest embeddings.
- `perplexity`: Scalar value representing the perplexity of the quantizer, a measure of codebook utilization.
- `encodings`: One-hot encoded matrix indicating the selected embedding for each input vector.
- `vq_loss`: Commitment loss for training the quantizer.
"""
function (vq::VectorQuantizerEMA)(inputs::AbstractArray{Float32, 3}, args::VQVAEArgs)
    # julia: data comes in TCB format, we need to do TCB -> BTC (T: time dimension)
    inputs = permutedims(inputs, (3, 1, 2)) 
    input_shape = size(inputs)

    # Flatten input to (batch_size * height * width, embedding_dim)
    flat_input = reshape(inputs, :, vq.embedding_dim)

    # Calculate matrix of euklidian distances between input and embeddings
    distances = sum(flat_input.^2, dims=2) .+ sum(vq.embedding.weight.^2, dims=2)' .- 2 .* (flat_input * vq.embedding.weight')

    # Find nearest embedding index, returns vector of minimal distance for each row packed into OneHotArrays.onehotbatch
    encodings = nothing  
    if CUDA.has_cuda_gpu()
        int_encoding_indices = gpu_argmin_rows(distances)  
        ChainRulesCore.@ignore_derivatives encodings = permutedims(
            OneHotArrays.onehotbatch(int_encoding_indices, 1:vq.num_embeddings),
            (2,1)
        )  
    else
        encoding_indices = argmin(distances, dims=2)  
        int_encoding_indices = vec(getindex.(encoding_indices, 2))  
        ChainRulesCore.@ignore_derivatives encodings = permutedims(
            OneHotArrays.onehotbatch(int_encoding_indices, 1:vq.num_embeddings),
            (2,1)
        )
    end 

    # Map one-hot encodings to embedding weight matrix and unflatten (= quantization)
    quantized = reshape(encodings * vq.embedding.weight, input_shape) 

    # EMA update (only if training, also no backprop as updates are not grad-based)
    if is_training()  
        Zygote.ignore() do
            vq = update!(vq, encodings, flat_input)
        end
    end

    # Loss
    commitment_loss = Flux.mse(stopgrad(quantized), inputs)
    vq_loss = args.commitment_cost * commitment_loss

    # Straight-through estimator for backpropagation (for gradient flow)
    quantized = straight_through(quantized, inputs)

    # Perplexity, a measure of how well the quantizer is performing
    if args.compute_perplexity
        avg_probs = sum(encodings, dims=2) ./ size(encodings, 2)  
        perplexity = exp(-sum(avg_probs .* log.(avg_probs .+ 1e-10)))
    else
        perplexity = 0.  
    end

    # Convert quantized from BTC -> TCB
    quantized = permutedims(quantized, (2, 3, 1))

    return quantized, perplexity, encodings, vq_loss
end

"""
Update EMA statistics and embedding weights for a VectorQuantizerEMA.

Arguments
- vq::VectorQuantizerEMA: Object holding EMA buffers (ema_cluster_size, ema_w), embeddings, decay and ϵ.
- encodings::AbstractArray{Bool,2}: Binary assignment matrix (batch_size × num_embeddings).
- flat_input::AbstractArray{Float32,2}: Flattened encoder outputs (batch_size × embedding_dim).

Returns
- Nothing. Updates vq in-place (ema_cluster_size, ema_w, and embedding.weight).
"""
function ema_update!(vq::VectorQuantizerEMA, encodings::AbstractArray{Bool, 2}, flat_input::AbstractArray{Float32, 2})
    # Update cluster size
    a = vq.ema_cluster_size .* vq.decay
    b = (1 - vq.decay) .* sum(encodings; dims=1)'
    vq.ema_cluster_size .= a + b
    n = sum(vq.ema_cluster_size)
    vq.ema_cluster_size .= ((vq.ema_cluster_size .+ vq.ϵ) ./ (n + vq.num_embeddings * vq.ϵ)) .* n

    # Update ema_w
    dw = encodings' * flat_input
    vq.ema_w .= vq.ema_w .* vq.decay .+ (1 - vq.decay) .* dw

    # Normalize and update embedding weights
    vq.embedding.weight .= (vq.ema_w ./ vq.ema_cluster_size)
end
