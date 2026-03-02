struct VQVAEModel
    encoder::Encoder
    pre_vq_conv::Conv
    lnorm::LayerNorm
    vq::VQ
    decoder::Decoder
end

"""
VQVAEModel(args::VQVAEArgs)

Construct a VQ-VAE model configured from `args`. Selects encoder/decoder variants
based on compression factor, dilated residuals and low-pass filter flags, and
builds the pre-quantization conv, layer norm and chosen vector quantizer.

Arguments
- args::VQVAEArgs : configuration struct containing relevant fields, e.g.
    - shared.compression_factor::Int
    - shared.embedding_dim::Int
    - shared.num_embeddings::Int
    - in_channels::Int
    - num_hiddens::Int
    - num_residual_layers::Int
    - num_residual_hiddens::Int
    - kernel_size_ds::Int
    - kernel_size_res::Int
    - dilated_residuals::Bool
    - apply_enc_lpf::Bool
    - apply_denc_lpf::Bool
    - enc_lpf_taps, enc_kaiser_beta, enc_cutoff_factor
    - denc_lpf_taps, denc_kaiser_beta, denc_cutoff_factor
    - dires_contribution_factor, commitment_cost, decay

Returns
- VQVAEModel : a model instance containing (encoder, pre_vq_conv, layernorm, vq, decoder).
"""
function VQVAEModel(args::VQVAEArgs) 
    if args.dilated_residuals && args.apply_enc_lpf
        encoder = if args.shared.compression_factor == 4
            DILPEncoder4(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.dires_contribution_factor,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        elseif args.shared.compression_factor == 8
            DILPEncoder8(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.dires_contribution_factor,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        elseif args.shared.compression_factor == 16
            DILPEncoder16(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.dires_contribution_factor,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        elseif args.shared.compression_factor == 32
            DILPEncoder32(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.dires_contribution_factor,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        end
    elseif !args.dilated_residuals && args.apply_enc_lpf
        encoder = if args.shared.compression_factor == 4
            LPEncoder4(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        elseif args.shared.compression_factor == 8
            LPEncoder8(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        elseif args.shared.compression_factor == 16
            LPEncoder16(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        elseif args.shared.compression_factor == 32
            LPEncoder32(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.enc_lpf_taps,
                beta=args.enc_kaiser_beta,
                cutoff_factor=args.enc_cutoff_factor
            )
        end
    else
        encoder = if args.shared.compression_factor == 4
            Encoder4(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
                )
        elseif args.shared.compression_factor == 8
            Encoder8(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
            )
        elseif args.shared.compression_factor == 16
            Encoder16(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
            )
        elseif args.shared.compression_factor == 32
            Encoder32(
                args.in_channels,
                args.num_hiddens,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
            )
        end
    end

    if args.apply_denc_lpf
        decoder = if args.shared.compression_factor == 4
            LPDecoder4(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.denc_lpf_taps,
                beta=args.denc_kaiser_beta,
                cutoff_factor=args.denc_cutoff_factor
                )
        elseif args.shared.compression_factor == 8
            LPDecoder8(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.denc_lpf_taps,
                beta=args.denc_kaiser_beta,
                cutoff_factor=args.denc_cutoff_factor
            )
        elseif args.shared.compression_factor == 16
            LPDecoder16(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.denc_lpf_taps,
                beta=args.denc_kaiser_beta,
                cutoff_factor=args.denc_cutoff_factor
            )
        elseif args.shared.compression_factor == 32
            LPDecoder32(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res;
                taps=args.denc_lpf_taps,
                beta=args.denc_kaiser_beta,
                cutoff_factor=args.denc_cutoff_factor
            )
        end
    else
        decoder = if args.shared.compression_factor == 4
            Decoder4(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
                )
        elseif args.shared.compression_factor == 8
            Decoder8(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
            )
        elseif args.shared.compression_factor == 16
            Decoder16(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
            )
        elseif args.shared.compression_factor == 32
            Decoder32(
                args.shared.embedding_dim,
                args.num_hiddens,
                args.in_channels,
                args.num_residual_layers,
                args.num_residual_hiddens,
                args.kernel_size_ds,
                args.kernel_size_res
            )
        end
        
    end

    pre_vq_conv = Conv(
        (1,),
        args.num_hiddens => args.shared.embedding_dim;
        stride=1,
        )
    lnorm = Flux.LayerNorm(args.shared.embedding_dim)
    vq = if args.decay > 0.0
        VectorQuantizerEMA(
            args.shared.num_embeddings,
            args.shared.embedding_dim,
            args.commitment_cost, 
            args.decay
        )
    else
        VectorQuantizer(
            args.shared.num_embeddings,
            args.shared.embedding_dim,
            args.commitment_cost
        )
    end
    return VQVAEModel(encoder, pre_vq_conv, lnorm, vq, decoder)
end


"""
Forward pass of VQVAEModel.

Arguments
- model::VQVAEModel: VQ-VAE model containing encoder, pre_vq_conv, vq, decoder, and optional layer norm.
- x::AbstractArray{Float32,3}: Input tensor (e.g., channels × length × batch).
- args::VQVAEArgs: Configuration and flags for the VQ module (e.g., pre_vq_layer_norm).

Returns
- x_recon::AbstractArray{Float32,3}: Reconstructed output from the decoder.
- perplexity::Real: Perplexity reported by the vector quantizer.
- encodings::AbstractArray: Discrete encodings produced by the quantizer.
- quantized::AbstractArray{Float32,3}: Quantized latent representations passed to the decoder.
- vq_in::AbstractArray{Float32,3}: Latent features before quantization (after encoder and 1×1 conv).
- vq_loss::Real: Quantization loss term.
"""
function (model::VQVAEModel)(x::AbstractArray{Float32, 3}, args::VQVAEArgs)
    enc_out = model.encoder(x)  # Pass input through the encoder
    vq_in = model.pre_vq_conv(enc_out)  # Apply 1x1 convolution
    if args.pre_vq_layer_norm
        vq_in = permutedims(vq_in, (2, 1, 3))  
        vq_in = model.lnorm(vq_in)
        vq_in = permutedims(vq_in, (2, 1, 3))
    end
    quantized, perplexity, encodings, vq_loss = model.vq(vq_in, args)  # Vector quantization
    x_recon = model.decoder(quantized)  # Decode the quantized representation
    return x_recon, perplexity, encodings, quantized, vq_in, vq_loss
end

# warmup pass, without VQ layer
"""
Call method for VQVAEModel used during warmup to encode input, apply pre-VQ convolution
(and optional layer normalization), and decode the resulting latent (AE only).

Arguments
- x::AbstractArray{Float32,3}: 3‑D input tensor.
- args::VQVAEArgs: argument struct; uses args.pre_vq_layer_norm to decide whether to apply layer normalization.
- warmup::Bool: when true runs the warmup path; otherwise a warning is emitted.

Returns
- x_recon::AbstractArray{Float32,3}: reconstructed output from the decoder.
- vq_in::AbstractArray{Float32,3}: pre-quantization latent (after pre_vq_conv and optional layer norm).
"""
function (model::VQVAEModel)(x::AbstractArray{Float32, 3}, args::VQVAEArgs, warmup::Bool)
    if warmup
        enc_out = model.encoder(x)  # Pass input through the encoder
        vq_in = model.pre_vq_conv(enc_out)  # Apply 1x1 convolution
        if args.pre_vq_layer_norm
            vq_in = permutedims(vq_in, (2, 1, 3))  
            vq_in = model.lnorm(vq_in)
            vq_in = permutedims(vq_in, (2, 1, 3))
        end
        x_recon = model.decoder(vq_in)  # Decode the quantized representation
        return x_recon, vq_in
    else
       @warn "Wrong model call for warmup!"
    end
end


"""
Compute the training loss for a VQ-VAE (recon loss and commitment loss).

Arguments
- model::VQVAEModel: the VQ-VAE model.
- x::AbstractArray{Float32,3}: input batch tensor.
- args::VQVAEArgs: arguments for the model (must include data_variance and any VQ loss params).

Returns
- loss::Number: scalar loss = (reconstruction MSE / args.data_variance) + vq_loss.
"""
function loss(model::VQVAEModel, x::AbstractArray{Float32, 3}, args::VQVAEArgs)
    x_recon, _, _, _, _, vq_loss = model(x, args) 

    # Reconstruction loss, optimizes encoder and decoder (approximated log-likelihood)
    recon_loss = Flux.mse(x_recon, x)
    loss = (recon_loss / args.data_variance) + vq_loss
    return loss
end

"""
Compute the combined training loss for a VQVAE model, to be used with the MR-STFT.

Arguments
- model :: VQVAEModel — the VQ-VAE model
- mr    :: MRSTFT     — multi-resolution STFT loss evaluator
- x     :: AbstractArray{Float32,3} — input batch (e.g. [channels, length, batch])
- args  :: VQVAEArgs  — training arguments (contains data_variance used for scaling)

Returns
- loss :: Real — scalar loss = (MSE(recon, x) / args.data_variance) + vq_loss + mr_stft
"""
function loss(model::VQVAEModel, mr::MRSTFT, x::AbstractArray{Float32, 3}, args::VQVAEArgs)
    x_recon, _, _, _, _, vq_loss = model(x, args) 

    # Reconstruction loss, optimizes encoder and decoder (approximated log-likelihood)
    recon_loss = Flux.mse(x_recon, x)
    # MR-STFT reconstruction loss
    mr_stft = mr(x, x_recon).total 
    loss = (recon_loss / args.data_variance) + vq_loss + mr_stft
    return loss
end

"""
Compute the warmup loss for a VQVAE, recon only, also calls the AE-only Forward pass.

Arguments
- model::VQVAEModel: the VQVAE model instance
- x::AbstractArray{Float32,3}: input batch
- args::VQVAEArgs: arguments/configuration containing `data_variance`

Returns
- loss::Float32: reconstruction loss divided by `args.data_variance`
"""
function loss_warmup(model::VQVAEModel, x::AbstractArray{Float32, 3}, args::VQVAEArgs)
    x_recon, _ = model(x, args, true) 
    recon_loss = Flux.mse(x_recon, x)
    loss = (recon_loss / args.data_variance) 
    return loss
end

"""
quantize_input(model::VQVAEModel, td::AbstractArray{Float32,3}, args::VQVAEArgs, dargs::DataGenerationArgs)

Quantize an input batch of time-series using the provided VQVAE model, processing in minibatches.

Arguments
- model::VQVAEModel: VQ-VAE model; called as model(mb, args, true) and expected to return (_, quantized).
- td::AbstractArray{Float32,3}: Input tensor of shape (time, channels, batch).
- args::VQVAEArgs: Contains model/runtime settings (e.g. batch_size, shared.compression_factor, shared.embedding_dim).
- dargs::DataGenerationArgs: Contains data settings (e.g. T for time length).

Returns
- Array{Float32,3}: Quantized embeddings of shape (Int(dargs.T / args.shared.compression_factor), Int(args.shared.embedding_dim), N_total) as Float32.
"""
function quantize_input(model::VQVAEModel, td::AbstractArray{Float32, 3}, args::VQVAEArgs, dargs::DataGenerationArgs)
    _, _, N_total = size(td)
    q = zeros(
        Float32,
        Int(dargs.T / args.shared.compression_factor),
        Int(args.shared.embedding_dim),
        N_total
    )

    for i in 1:args.batch_size:N_total
        idx_end = min(i + args.batch_size - 1, N_total)
        mb = td[:, :, i:idx_end] |> dev
        _, quantized_mb = model(mb, args, true) # first dim=time, second=channels, third=batch
        q[:, :, i:idx_end] .= quantized_mb |> cpu
    end
    return q    
end

"""
    tokenize_input(model::VQVAEModel, x::AbstractArray{Float32, 3}, args::VQVAEArgs)

Processes the input tensor `x` through the encoder and pre-vector-quantization (pre-VQ) convolutional layers of the given `model`, then applies vector quantization.

# Arguments
- `model::VQVAEModel`: The VQ-VAE model containing the encoder, pre-VQ convolution, and vector quantizer.
- `x::AbstractArray{Float32, 3}`: The input data tensor, typically of shape (time, chanels, batch dimension).
- `args::VQVAEArgs`: Configuration arguments for the model and vector quantizer.

# Returns
- `quantized`: The quantized output from the vector quantizer.
- `encodings`: The encoding indices representing the quantized vectors.
"""
function tokenize_input(model::VQVAEModel, x::AbstractArray{Float32, 3}, args::VQVAEArgs)
    vq_in = model.pre_vq_conv(
        model.encoder(x)
    )
    quantized, _, encodings, _ = model.vq(vq_in, args)
    return quantized, encodings
end

"""
    tokenize_fast(model::VQVAEModel, x::AbstractArray{Float32, 1}, args::VQVAEArgs)

Efficiently tokenizes a 1D input array `x` using the provided `model` (of type `VQVAEModel`) by mapping the input to the nearest vector quantization (VQ) embedding indices.

# Arguments
- `model::VQVAEModel`: The VQ-VAE model containing encoder and VQ embedding layers.
- `x::AbstractArray{Float32, 1}`: The 1D input array to be tokenized.
- `args::VQVAEArgs`: Configuration arguments for the model and vector quantizer.

# Returns
- `int_encoding_indices`: A vector of integer indices corresponding to the nearest embedding for each input vector.
"""
function tokenize_fast(model::VQVAEModel, x::AbstractArray{Float32, 1}, args::VQVAEArgs)
    x = reshape(x, length(x), 1, 1)
    vq_in = model.pre_vq_conv(
        model.encoder(x)
    )
    if args.pre_vq_layer_norm
        vq_in = permutedims(vq_in, (2, 1, 3))  
        vq_in = model.lnorm(vq_in)
        vq_in = permutedims(vq_in, (2, 1, 3))
    end
    inputs = permutedims(vq_in, (3, 1, 2))
    flat_input = reshape(inputs, :, model.vq.embedding_dim)  

    # Calculate matrix of euklidian distances between input and embeddings
    distances = (sum(flat_input.^2, dims=2) .+ sum(model.vq.embedding.weight.^2, dims=2)' .- 2 .* (flat_input * model.vq.embedding.weight'))

    # Find nearest embedding index, returns vector of minimal distance for each row packed into OneHotArrays.onehotbatch
    int_encoding_indices = nothing  
    if CUDA.has_cuda_gpu()
        int_encoding_indices = gpu_argmin_rows(distances)   
    else
        encoding_indices = argmin(distances, dims=2)  
        int_encoding_indices = vec(getindex.(encoding_indices, 2))  
    end 
    return int_encoding_indices
end

"""
    tokenize_fast(model::VQVAEModel, x::AbstractArray{Float32, 3}, args::VQVAEArgs) -> Matrix{Int}

Tokenizes the input tensor `x` using the provided `model` of type `VQVAEModel`. This function processes the input through the encoder and pre-vector-quantization (pre-VQ) convolutional layers, then computes the nearest embedding indices for each input vector using Euclidean distance.

# Arguments
- `model::VQVAEModel`: The VQ-VAE model containing encoder, pre-VQ convolution, and vector quantizer.
- `x::AbstractArray{Float32, 3}`: Input tensor of shape (time, channels, batch).
- `args::VQVAEArgs`: Configuration arguments for the model and vector quantizer.

# Returns
- `Matrix{Int}`: A matrix of embedding indices with shape (T, N), where `T` is the sequence length and `N` is the batch size.
"""
function tokenize_fast(model::VQVAEModel, x::AbstractArray{Float32, 3}, args::VQVAEArgs)
    vq_in = model.pre_vq_conv(
        model.encoder(x)
    )
    # here we need to hack our manual layer norm
    if args.pre_vq_layer_norm
        vq_in = permutedims(vq_in, (2, 1, 3))  
        vq_in = model.lnorm(vq_in)
        vq_in = permutedims(vq_in, (2, 1, 3))
    end
    inputs = permutedims(vq_in, (3, 1, 2))
    N, T, emb = size(inputs)
    flat_input = reshape(inputs, :, model.vq.embedding_dim)  

    # Calculate matrix of euklidian distances between input and embeddings
    distances = (sum(flat_input.^2, dims=2) .+ sum(model.vq.embedding.weight.^2, dims=2)' .- 2 .* (flat_input * model.vq.embedding.weight'))

    # Find nearest embedding index, returns vector of minimal distance for each row packed into OneHotArrays.onehotbatch
    int_encoding_indices = nothing  
    if CUDA.has_cuda_gpu()
        int_encoding_indices = gpu_argmin_rows(distances)   
    else
        encoding_indices = argmin(distances, dims=2)  
        int_encoding_indices = vec(getindex.(encoding_indices, 2))  
    end 
    return permutedims(reshape(int_encoding_indices, N, T), (2,1))
end

"""
    decode_output(model::VQVAEModel, int_encoding_indices::AbstractArray{Int32, 1})

Decodes a sequence of integer encoding indices using a trained `VQVAEModel`.

# Arguments
- `model::VQVAEModel`: The trained VQ-VAE model containing the vector quantizer and decoder.
- `int_encoding_indices::AbstractArray{Int32, 1}`: A 1D array of integer indices representing quantized latent codes.

# Returns
- The reconstructed output from the decoder, corresponding to the provided encoding indices.
"""
function decode_output(model::VQVAEModel, int_encoding_indices::AbstractArray{Int32, 1})  
    encodings = permutedims(OneHotArrays.onehotbatch(int_encoding_indices, 1:model.vq.num_embeddings),(2,1))
    x_quantized = reshape(encodings * model.vq.embedding.weight, (size(encodings)[1], model.vq.embedding_dim, 1))  
    return model.decoder(x_quantized)
end