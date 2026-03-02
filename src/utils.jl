"""
exp_growth_scheduler(n_epochs::Int, final_value::Float32; n_epochs_final_value::Int=0)

Compute an exponential growth schedule over n_epochs, optionally followed by a plateau.

Arguments
- n_epochs::Int: number of epochs in the growth phase.
- final_value::Float32: target value reached at the end of the growth phase.
- n_epochs_final_value::Int=0: number of additional epochs to hold the final_value (default 0).

Returns
- Vector{Float32}: scheduled values (growth phase, then optional plateau).
"""
function exp_growth_scheduler(n_epochs::Int, final_value::Float32; n_epochs_final_value::Int=0)
    α = log(final_value + 1) / n_epochs
    growth = [final_value * (1 - exp(-α * epoch)) / (1 - exp(-α * n_epochs)) for epoch in 1:n_epochs]
    if n_epochs_final_value == 0
        return growth
    end
    plateau = fill(final_value, n_epochs_final_value)
    return vcat(growth, plateau)
end

"""
lin_growth_scheduler(n_epochs::Int, final_value::Float32; n_epochs_final_value::Int=0)

Linear growth scheduler.

Arguments
- n_epochs::Int: number of epochs for the linear segment.
- final_value::Float32: target value reached at the end of the linear segment.
- n_epochs_final_value::Int=0: optional number of epochs to hold the final value.

Returns
- Vector{Float32}: the linear schedule, possibly followed by a plateau.
"""
function lin_growth_scheduler(n_epochs::Int, final_value::Float32; n_epochs_final_value::Int=0)
    growth = [final_value * (epoch - 1) / (n_epochs - 1) for epoch in 1:n_epochs]
    if n_epochs_final_value == 0
        return growth
    end
    plateau = fill(final_value, n_epochs_final_value)
    return vcat(growth, plateau)
end

"""
Linear growth scheduler.

Arguments
- n_epochs::Int: Number of epochs for the linear ramp.
- start_value::Float32: Value at epoch 1.
- end_value::Float32: Value at epoch n_epochs.
- n_epochs_final_value::Int=0: Optional number of epochs to append with the constant end_value (default 0).

Returns
- Vector{Float32}: Scheduled values of length n_epochs (or n_epochs + n_epochs_final_value if appended).
"""
function lin_growth_scheduler(n_epochs::Int, start_value::Float32, end_value::Float32; n_epochs_final_value::Int=0)
    growth = [start_value + (end_value - start_value) * (epoch - 1) / (n_epochs - 1) for epoch in 1:n_epochs]
    if n_epochs_final_value == 0
        return growth
    end
    plateau = fill(end_value, n_epochs_final_value)
    return vcat(growth, plateau)
end

"""
Pick a random sampling horizon from a vector of integer values.

# Arguments
- values::AbstractVector{Int}: Vector of integer candidate horizons.

# Returns
- Int32: A randomly selected value converted to Int32.
"""
function pick_rand_sampling_horizon(values::AbstractVector{Int})
    return Int32(values[rand(1:length(values))])
end

"""
Select a sampling horizon from `values`, giving higher probability to later indices.

Arguments
- `values::AbstractVector{Int}`: Candidate integer horizons.
- `epoch::Int`: Current epoch number used to scale the selection range.
- `max_epochs::Int`: Maximum number of epochs for scaling.

Returns
- `Int32`: A selected value from `values`.
"""
function pick_sampling_horizon(values::AbstractVector{Int}, epoch::Int, max_epochs::Int)
    n = length(values)
    base_idx = clamp(ceil(Int, epoch / max_epochs * n), 1, n)
    idx_range = 1:base_idx
    weights = collect(idx_range) .^ 2
    probs = weights / sum(weights)  
    idx = rand(Categorical(probs))
    return Int32(values[idx])
end