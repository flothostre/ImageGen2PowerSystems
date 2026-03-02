# this includes the training algorithms for the VQVAE model



"""
train_VQVAE(model::VQVAEModel, optimizer::Flux.Optimiser, args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)

Train a VQVAE model using the provided optimizer and configuration.

Arguments
- model::VQVAEModel: the VQ‑VAE model to train.
- optimizer::Flux.Optimiser: Flux optimizer used to update model parameters.
- args::VQVAEArgs: training settings (epochs, batch_size, checkpointing, filename, etc.).
- mrargs::MRSTFTArgs: multi‑resolution STFT loss and scheduling options.
- dargs::DataGenerationArgs: parameters for loading/generating training data.

Returns
- (model, optimizer, losses): the trained model, the optimizer state, and a collection of per‑epoch minibatch losses.
"""
function train_VQVAE(model::VQVAEModel, optimizer, args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)
    args = args  |> dev
    model = model |> dev
    optimizer = optimizer |> dev

    @info "Training experiment $(args.shared.filename) on $dev"
    @info "Number of total tunable model parameters: $(sum(length, Flux.params(model)))"

    Flux.trainmode!(model)
    set_training!(true)

    # to distinguish between synthetic and augmented training data
    training_data = if args.shared.synthetic_data_only
        @warn "DEPRECATED: Using synthetic training data only (normalization not supported anymore)"
        load_synthetic_training_data()
    else
        @info "Using generated transient responses..."
        load_training_data(dargs)
    end

    # applying the manual normalization
    training_data = manual_normalization(training_data)

    losses = []
    if mrargs.use_mrstft
        mrstft_schedule = exp_growth_scheduler(
            (args.epochs ÷ 2),
            Float32(mrargs.contribution_factor);
            n_epochs_final_value=(args.epochs ÷ 2)
        )
    end
    
    for i in 1:args.epochs
        training_loader = get_minibatch(training_data; B=args.batch_size) 

        if mrargs.use_mrstft
            mrargs.contribution_factor = Float32(mrstft_schedule[i])
            mr = MRSTFT(mrargs) |> dev
        end

        @info "Training epoch $(i) / $(args.epochs)"
        loss_mb = ones(Float32, length(training_loader)) |> dev
        @showprogress for (j, data) in enumerate(training_loader)
            data = data |> dev

            # Compute loss and gradients
            l, ∇ = Flux.withgradient(model) do m
                if mrargs.use_mrstft
                    loss(m, mr, data, args)
                else
                    loss(m, data, args)
                end
            end

            # Update model parameters
            Flux.update!(optimizer, model, ∇[1])
            
            loss_mb[j:j] = l
        end

        loss_mb = loss_mb |> cpu
        @info "Epoch $(i) mean loss: $(mean(loss_mb))"
        push!(losses, loss_mb)

        if args.save_checkpoints && i % 100 == 0
            @info "Saving model checkpoint after epoch $(i)"
            jldsave(
                "trained_models/VQVAE_models/" * args.shared.filename * "_VQVAE_epoch_$(i).jld2",
                model_state=Flux.state(model |> cpu),
                opt_state=optimizer,
                loss=losses,
                args=args,
                dargs=dargs
            )
        end
    end

    testmode!(model)
    set_training!(false)

    jldsave(
        "trained_models/VQVAE_models/" * args.shared.filename * "_VQVAE.jld2",
        model_state=Flux.state(model |> cpu),
        opt_state=optimizer,
        loss=losses,
        args = args,
        dargs=dargs
    )

    return model, optimizer, losses
end

"""
fancy_train_VQVAE(model::VQVAEModel, optimizer, args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)

Train a VQVAE with an autoencoder warm‑up, k‑means initialization of the VQ embeddings,
and full VQVAE training. Optionally includes MR‑STFT loss and schedules for MR contribution
and commitment cost. Periodically saves checkpoints and the final trained model.

Arguments
- model::VQVAEModel: the VQVAE model to train
- optimizer: optimizer instance (contains VQ optimizer state)
- args::VQVAEArgs: general training arguments (epochs, batch_size, commitment_cost, save flags, shared metadata, etc.)
- mrargs::MRSTFTArgs: MR‑STFT options (use_mrstft, contribution_factor, etc.)
- dargs::DataGenerationArgs: data generation/loading parameters

Returns
- (model, optimizer, losses): trained model, optimizer state, and collected losses (per mini‑batch)
"""
function fancy_train_VQVAE(model::VQVAEModel, optimizer, args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)
    args = args  |> dev
    model = model |> dev
    optimizer = optimizer |> dev 
    
    @info "Training experiment $(args.shared.filename) on $dev"
    @info "Number of tunable model parameters: $(sum(length, Flux.params(model)))"

    
    Flux.trainmode!(model)
    Optimisers.freeze!(optimizer.vq)
    set_training!(false)  
    @info "Warming up the Autoencoder..."
    @info "Number of AE tunable model parameters: $(sum(length, Flux.params(model)))"
    
    # to distinguish between synthetic and augmented training data
    training_data = if args.shared.synthetic_data_only
        @warn "DEPRECATED: Using synthetic training data only (normalization not supported anymore)"
        load_synthetic_training_data()
    else
        @info "Using generated transient responses..."
        load_training_data(dargs)
    end

    # applying the manual normalization
    training_data = manual_normalization(training_data)

    losses = []
    warmup_epochs = min(Int(args.epochs ÷ 5), 4)
    training_epochs = Int(args.epochs - warmup_epochs)

    # training loop 1: warming up the Autoencoder
    for i in 1:warmup_epochs
        training_loader = get_minibatch(training_data; B=args.batch_size) 

        @info "(Warm-up) epoch $(i) / $(warmup_epochs)"
        loss_mb = ones(Float32, length(training_loader)) |> dev
        @showprogress for (j, data) in enumerate(training_loader)
            data = data |> dev

            # Compute loss and gradients
            l, ∇ = Flux.withgradient(model) do m
                loss_warmup(m, data, args)
            end
            
            # Update model parameters
            Flux.update!(optimizer, model, ∇[1])
            
            loss_mb[j:j] = l
        end

        loss_mb = loss_mb |> cpu
        @info "Epoch $(i) mean loss: $(mean(loss_mb))"
        push!(losses, loss_mb)
    end

    # initializing the VQ layer with k-means
    @info "Initializing the VQ layer with k-means"
    _, _, N_total = size(training_data)
    rand_indices = rand(1:N_total, min(2000, N_total))
    td_rand = training_data[:, :, rand_indices]
    quantized = quantize_input(model, td_rand, args, dargs)
    Z = permutedims(reshape(quantized, :, size(quantized, 2)), (2,1)) |> cpu  # flattening time and batch dim
    km = kmeans(Z, args.shared.num_embeddings; maxiter=100, display=:none)
    emb = permutedims(km.centers, (2,1)) |> dev
    model.vq.embedding.weight .= emb
    @info "VQ layer has been initialized"

    # second training loop: full VQVAE training
    Optimisers.thaw!(optimizer.vq)
    set_training!(true)
    @info "Regular training the full VQVAE..."
    @info "Number of total tunable model parameters: $(sum(length, Flux.params(model)))"

    # TODO: one major issue: if training epochs is uneven, i get a bounds error --> FIX
    # MR-STFT contribution factor schedule
    if mrargs.use_mrstft
        mrstft_schedule = exp_growth_scheduler(
            (training_epochs ÷ 2),
            Float32(mrargs.contribution_factor);
            n_epochs_final_value=(training_epochs ÷ 2)
        )
    end
    
    # increase of commitment cost
    comm_cost_schedule = lin_growth_scheduler(
        (training_epochs ÷ 2), 
        Float32(0.1),
        Float32(args.commitment_cost);
        n_epochs_final_value=(training_epochs ÷ 2)
    )

    for i in 1:training_epochs
        training_loader = get_minibatch(training_data; B=args.batch_size) 

        if mrargs.use_mrstft
            mrargs.contribution_factor = Float32(mrstft_schedule[i])
            mr = MRSTFT(mrargs) |> dev
        end

        @info "Training epoch $(i) / $(training_epochs)"
        loss_mb = ones(Float32, length(training_loader)) |> dev
        @showprogress for (j, data) in enumerate(training_loader)
            data = data |> dev

            args.commitment_cost = Float32(comm_cost_schedule[i])

            # Compute loss and gradients
            l, ∇ = Flux.withgradient(model) do m
                if mrargs.use_mrstft
                    loss(m, mr, data, args)
                else
                    loss(m, data, args)
                end
            end

            # Update model parameters
            Flux.update!(optimizer, model, ∇[1])
            
            loss_mb[j:j] = l
        end

        loss_mb = loss_mb |> cpu
        @info "Epoch $(i) mean loss: $(mean(loss_mb))"
        push!(losses, loss_mb)

        if args.save_checkpoints && i % 100 == 0
            @info "Saving model checkpoint after epoch $(i)"
            jldsave(
                "trained_models/VQVAE_models/" * args.shared.filename * "_VQVAE_epoch_$(i).jld2",
                model_state=Flux.state(model |> cpu),
                opt_state=optimizer,
                loss=losses,
                args=args,
                dargs=dargs
            )
        end
    end

    testmode!(model)
    set_training!(false)

    jldsave(
        "trained_models/VQVAE_models/" * args.shared.filename * "_VQVAE.jld2",
        model_state=Flux.state(model |> cpu),
        opt_state=optimizer,
        loss=losses,
        args = args,
        dargs=dargs
    )

    return model, optimizer, losses
end

"""
run_VQVAE_training(args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)

Train a VQ-VAE model with provided training, MR-STFT and data-generation settings.

Arguments
- args::VQVAEArgs: Model and training hyperparameters (learning rate, grad clip, etc.).
- mrargs::MRSTFTArgs: Multi-resolution STFT arguments used for loss/computation.
- dargs::DataGenerationArgs: Data generation / loading parameters.

Returns
- model::VQVAEModel: Trained model moved to CPU.
- optimizer::Any: Optimizer state associated with the trained model (moved to CPU).
- loss::Real: Final training loss.
"""
function run_VQVAE_training(args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)
    model = VQVAEModel(args)

    opt = OptimiserChain(
        Optimisers.ClipGrad(args.grad_clip), 
        Optimisers.AdamW(
            args.learning_rate,
            (0.9, 0.99), 
            args.learning_rate_decay
        )
    )
    optimizer = Flux.setup(opt, model)

    # train the model
    model, optimizer, loss = train_VQVAE(model, optimizer, args, mrargs, dargs)

    # return to CPU to save memory
    model = model |> cpu
    optimizer = optimizer |> cpu

    return model, optimizer, loss
end

"""
run_fancy_VQVAE_training(args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)

Train a VQVAE model using provided model, MR-STFT, and data generation arguments.

Arguments
- args::VQVAEArgs: model definition and training hyperparameters.
- mrargs::MRSTFTArgs: MR-STFT / signal processing parameters.
- dargs::DataGenerationArgs: data generation and loading configuration.

Returns
- (model::VQVAEModel, optimizer::Any, loss::Real): the trained model moved to CPU, the optimizer state (on CPU), and the final training loss.
"""
function run_fancy_VQVAE_training(args::VQVAEArgs, mrargs::MRSTFTArgs, dargs::DataGenerationArgs)
    model = VQVAEModel(args)

    opt = OptimiserChain(
        Optimisers.ClipGrad(args.grad_clip), 
        Optimisers.AdamW(
            args.learning_rate,
            (0.9, 0.99), 
            args.learning_rate_decay
        )
    )
    optimizer = Flux.setup(opt, model)

    # train the model
    model, optimizer, loss = fancy_train_VQVAE(model, optimizer, args, mrargs, dargs)

    # return to CPU to save memory
    model = model |> cpu
    optimizer = optimizer |> cpu

    return model, optimizer, loss
end