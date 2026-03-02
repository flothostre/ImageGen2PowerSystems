# this is the file with the models that will be used for the OSMSES paper focusing on VQVAE only

# -----------------------------------------------------------------------------
# PSCC comments
# these are the final tests, i.e., the best model configurations that ended up in the PSCC paper
# exp2 performed best and had the most diverse latent space, so we used that for the forecaster and VQVAE...

# major changes to previous experiments:
# scheduled commitment cost, other parameters scheduled as well
# longer training
# fancy training loops

# ----------------------------------------------------------------------------------
using Attemt
using Dates


# part 1: influence of commitment cost

shargs = SharedArgs(compression_factor=16, filename="vqvae_exp1")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=true, apply_denc_lpf=true, commitment_cost=0.15)
mrargs = MRSTFTArgs(use_mrstft=true)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"


shargs = SharedArgs(compression_factor=16, filename="vqvae_exp2")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=true, apply_denc_lpf=true, commitment_cost=0.25)
mrargs = MRSTFTArgs(use_mrstft=true)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"


shargs = SharedArgs(compression_factor=16, filename="vqvae_exp3")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=true, apply_denc_lpf=true, commitment_cost=0.35)
mrargs = MRSTFTArgs(use_mrstft=true)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"



# part 2: influence of compression factor 

shargs = SharedArgs(compression_factor=8, filename="vqvae_exp4")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=false, apply_denc_lpf=false)
mrargs = MRSTFTArgs(use_mrstft=false)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"


shargs = SharedArgs(compression_factor=16, filename="vqvae_exp5")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=false, apply_denc_lpf=false)
mrargs = MRSTFTArgs(use_mrstft=false)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"


shargs = SharedArgs(compression_factor=32, filename="vqvae_exp6")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=false, apply_denc_lpf=false)
mrargs = MRSTFTArgs(use_mrstft=false)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"


# part 3: influence of LP filter

shargs = SharedArgs(compression_factor=16, filename="vqvae_exp7")
vargs = VQVAEArgs(shared=shargs, apply_enc_lpf=true, apply_denc_lpf=true)
mrargs = MRSTFTArgs(use_mrstft=false)
dargs = DataGenerationArgs()

t0 = now()
try
    run_fancy_VQVAE_training(vargs, mrargs, dargs);
catch e
    @warn "Experiment failed: $(shargs.filename). Skipping. Error: $e"
end
@info "Experiment $shargs took: $(now() - t0)"

# part 4: influence of MRSTFT (implicitly included in exp2), no training required...


exit() 


using JLD2
using Flux
using Attemt
using CUDA
using Plots
using CSV
using DataFrames
using Interpolations
using MLUtils
using MultivariateStats
using Statistics
using StatsBase
using BenchmarkTools

dev = Flux.get_device()
dargs = DataGenerationArgs()
movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]
dargs = DataGenerationArgs()

vargs_t1, vm_t1, vl_t1 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp1_VQVAE.jld2")
vargs_t2, vm_t2, vl_t2 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp2_VQVAE.jld2")
vargs_t3, vm_t3, vl_t3 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp3_VQVAE.jld2")
vargs_t4, vm_t4, vl_t4 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp4_VQVAE.jld2")
vargs_t5, vm_t5, vl_t5 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp5_VQVAE.jld2")
vargs_t6, vm_t6, vl_t6 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp6_VQVAE.jld2")
vargs_t7, vm_t7, vl_t7 = load_VQVAE_model("trained_models/VQVAE_models/vqvae_exp7_VQVAE.jld2")

vl_t1 = vcat(vl_t1...)
vl_t2 = vcat(vl_t2...)
vl_t3 = vcat(vl_t3...)
vl_t4 = vcat(vl_t4...)
vl_t5 = vcat(vl_t5...)
vl_t6 = vcat(vl_t6...)
vl_t7 = vcat(vl_t7...)

exp_names = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]

plot(movingaverage(vl_t1, 200), label=exp_names[1], xlabel="update steps", ylabel="loss", title="VQVAE model comparison", legend=:topright)
plot!(movingaverage(vl_t2, 200), label=exp_names[2])
plot!(movingaverage(vl_t3, 200), label=exp_names[3])
plot!(movingaverage(vl_t4, 200), label=exp_names[4])
plot!(movingaverage(vl_t5, 200), label=exp_names[5])
plot!(movingaverage(vl_t6, 200), label=exp_names[6])
plot!(movingaverage(vl_t7, 200), label=exp_names[7])

# recon comparison
for _ in 1:10
    original, reconstructions, _, _, losses = compute_VQVAE_reconstructions(
        [vm_t1, vm_t2, vm_t3, vm_t4, vm_t5, vm_t6, vm_t7],
        [vargs_t1, vargs_t2, vargs_t3, vargs_t4, vargs_t5, vargs_t6, vargs_t7],
        dargs;
        batch_size=256,
        validation_set=true
    )
    plot_VQVAE_original_reconstruction(
        original,
        reconstructions,
        losses,
        exp_names,
        256;
        save_png=true,
        exp_name="vqvae_exp_val"
    )
end