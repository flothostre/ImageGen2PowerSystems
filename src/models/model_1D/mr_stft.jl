"""
hann(n::Integer, T=Float32)

Generate a Hann (raised-cosine) window of length `n`.

Arguments
- n::Integer: number of samples in the window.
- T: element type or conversion function for output (default `Float32`).

Returns
- Vector{T}: length-`n` Hann window with values in [0, 1].
"""
function hann(n::Integer, T=Float32)
    return T.(0.5 .- 0.5 .* cos.(2f0 * π .* (0:n-1) ./ n))
end

"""
Return a small epsilon (1e-7) converted to the element type of `x`.

Arguments
- x::AbstractArray: input whose element type is used.

Returns
- A value `1e-7` converted to `eltype(x)`.
"""
function eps_like(x)
    return eltype(x)(1e-7)
end

"""
Compute number of STFT frames for a signal of length T.

Arguments
- T::Int: input signal length in samples
- n_fft::Int: FFT window length
- hop::Int: hop (stride) between consecutive frames

Returns
- Int: number of frames (with padding so length >= n_fft)
"""
function compute_nframes(T::Int, n_fft::Int, hop::Int)
    Tpad = max(T, n_fft)
    return cld(Tpad - n_fft, hop) + 1
end


"""
GPU kernel that copies framed windows from a 2D input signal `y` into a 3D output tensor `F`
according to short-time framing parameters.

# Arguments
- `F::CuDeviceArray{T,3}`: Output tensor (n_fft × nfrm × batch) written in-place on the GPU.
- `y::CuDeviceArray{T,2}`: Input signal (time × batch) on the GPU.
- `n_fft::Int`: Frame length (number of samples per frame).
- `hop::Int`: Hop size (samples between consecutive frames).
- `nfrm::Int`: Number of frames per batch.
- `total::Int`: Total number of elements/threads to process (typically <= prod(size(F))).

# Returns
- `Nothing`: Updates `F` in-place.
"""
function framing_kernel(F, y, n_fft, hop, nfrm, total)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    
    if i <= total
        t = Int(mod(i - 1, n_fft)) + 1
        f = Int(mod((i - 1) ÷ n_fft, nfrm)) + 1
        b = Int((i - 1) ÷ (n_fft * nfrm)) + 1
        t0 = (f - 1) * hop + 1
        F[t, f, b] = y[t0 + t - 1, b]
    end
    return
end

"""
frame_gpu(x::CUDA.CuArray, n_fft::Int, hop::Int)

Frame a batched 1D signal on the GPU into overlapping windows.

Arguments
- x::CUDA.CuArray{T} where T : input array of shape (T, B) where T is time length and B is batch size
- n_fft::Int : frame length (number of samples per frame)
- hop::Int : hop (stride) between successive frames

Returns
- CUDA.CuArray{T} of shape (n_fft, nfrm, B) containing the framed (zero-padded) signal on the GPU,
    where nfrm is the computed number of frames.
"""
function frame_gpu(x::CUDA.CuArray, n_fft::Int, hop::Int)
    T, B = size(x)
    nfrm = compute_nframes(T, n_fft, hop)
    Tpad = (nfrm - 1) * hop + n_fft
    y = CUDA.zeros(eltype(x), Tpad, B)
    CUDA.@sync CUDA.copyto!(view(y, 1:T, :), x)  
    F = CUDA.CuArray{eltype(x)}(undef, n_fft, nfrm, B)
    
    threads = 256
    total = n_fft * nfrm * B
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks framing_kernel(F, y, n_fft, hop, nfrm, total)
    return F
end

"""
overlap_add_kernel!(gx, dF, n_fft, hop, nfrm, T, total)

Perform GPU overlap-add: each thread maps a sample from dF into gx using atomic adds.

Arguments
- gx::CUDA.CuDeviceArray{T,2}: output time-domain buffer, modified in-place (rows=time, cols=batch)
- dF::CUDA.CuDeviceArray{T,3}: input framed slices with shape (n_fft, nfrm, batch)
- n_fft::Int: FFT frame length
- hop::Int: hop (stride) between frames
- nfrm::Int: number of frames per batch
- T::Int: total available time samples (number of rows in gx)
- total::Int: total number of elements/threads to process

Returns
- nothing (in-place update of gx)
"""
function overlap_add_kernel!(gx, dF, n_fft, hop, nfrm, T, total)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if i <= total
        t = Int(mod(i - 1, n_fft)) + 1
        f = Int(mod((i - 1) ÷ n_fft, nfrm)) + 1
        b = Int((i - 1) ÷ (n_fft * nfrm)) + 1
        s = (f - 1) * hop + t
        if s <= T
            @inbounds CUDA.@atomic gx[s, b] += dF[t, f, b]
        end
    end
    return
end

Zygote.@adjoint function frame_gpu(x::CUDA.CuArray{T,2}, n_fft::Int, hop::Int) where {T}
    F = frame_gpu(x, n_fft, hop)  # primal
    function back(ΔF)
        Tlen, B = size(x)
        nfrm = compute_nframes(Tlen, n_fft, hop)
        gx = similar(x); CUDA.fill!(gx, zero(T))
        
        threads = 256
        total   = n_fft * nfrm * B
        blocks  = cld(total, threads)
        @cuda threads=threads blocks=blocks overlap_add_kernel!(gx, ΔF, n_fft, hop, nfrm, Tlen, total)
        return (gx, nothing, nothing)  # grads: (∂x, ∂n_fft, ∂hop)
    end
    return F, back
end

"""
Apply a 1D window vector along the first dimension of a 3D array.

Arguments
- F::AbstractArray{T,3}: input 3D array with size(F,1) == length(w)
- w::AbstractVector{T}: window vector to broadcast along the first dimension

Returns
- AbstractArray{T,3}: result of elementwise multiplying F by w broadcast over the first dimension
"""
function apply_window(F, w)
    @assert size(F, 1) == length(w)
    w3 = reshape(w, :, 1, 1)       
    return F .* w3                 
end


struct MRSTFT
    fft_sizes::Vector{Int}
    hops::Vector{Int}
    winfun::Function
    λ_sc::Float32      
    λ_mag::Float32     
end

"""
Construct an MRSTFT instance from an MRSTFTArgs container.

Arguments
- margs::MRSTFTArgs: input arguments containing fft_sizes::AbstractVector, hop_factor::Number (scalar or vector), contribution_factor::Number, λ_sc::Number, λ_mag::Number.

Returns
- MRSTFT: instance with fft_sizes::Vector{Int}, hops::Vector{Int}, winfun::Function (hann), and λ_sc, λ_mag as Float32.
"""
function MRSTFT(margs::MRSTFTArgs)
    hops = Int.(round.(margs.fft_sizes ./ margs.hop_factor))
    @assert length(margs.fft_sizes) == length(hops)
    winfun = n -> hann(n, Float32)

    λ_sc = Float32(margs.contribution_factor) * Float32(margs.λ_sc)
    λ_mag = Float32(margs.contribution_factor) * Float32(margs.λ_mag)

    return MRSTFT(
        collect(Int.(margs.fft_sizes)),
        collect(Int.(hops)),
        winfun,
        Float32(λ_sc),
        Float32(λ_mag)
    )
end

"""
Compute multiresolution STFT loss between a target and a prediction.
Callable func of MR-STFT object.

Arguments
- mr::MRSTFT: configuration containing fft_sizes, hops, winfun and weights (λ_sc, λ_mag).
- target::AbstractArray{Float32,3}: 3‑D Float32 array of reference audio frames.
- pred::AbstractArray{Float32,3}: 3‑D Float32 array of predicted audio frames (same size as `target`).

Returns
- NamedTuple{(:total,:sc,:logmag)}: 
    - total::Float32 — weighted sum of spectral convergence and log‑magnitude losses,
    - sc::Float32 — average spectral convergence across resolutions,
    - logmag::Float32 — average log‑magnitude loss across resolutions.
"""
function (mr::MRSTFT)(target::AbstractArray{Float32, 3}, pred::AbstractArray{Float32, 3})
    @assert size(target) == size(pred)
    target = reshape(target, :, size(target, 3))
    pred = reshape(pred, :, size(pred, 3))
    target = target |> dev
    pred = pred |> dev

    R = length(mr.fft_sizes)
    sc_sum = 0.0f0
    lm_sum = 0.0f0

    EPS_MAG = 1e-8
    EPS_LOG = 1e-5
    EPS_DEN = 1e-6
    CLIPMAX = 1e6

    for i in 1:R
        Fx = frame_gpu(target, mr.fft_sizes[i], mr.hops[i])
        Fy = frame_gpu(pred, mr.fft_sizes[i], mr.hops[i])

        w = mr.winfun(mr.fft_sizes[i]) |> dev

        apply_window(Fx, w)
        apply_window(Fy, w)

        X = AbstractFFTs.fft(Fx, 1)
        Y = AbstractFFTs.fft(Fy, 1)

        # alternative
        magX = sqrt.(real(X).^2 .+ imag(X).^2 .+ EPS_MAG)   
        magX = clamp.(magX, 0f0, CLIPMAX)

        magY = sqrt.(real(Y).^2 .+ imag(Y).^2 .+ EPS_MAG)
        magY = clamp.(magY, 0f0, CLIPMAX)

        # Spectral convergence
        Δ = magY .- magX
        num = sqrt(sum(Δ .^ 2))
        denom = max(sqrt(sum(magX .^ 2)) + eps_like(magX), EPS_DEN)
        sc = num / denom

        # log-magnitude
        logX = log.(clamp.(magX .+ eps_like(magY), EPS_LOG, CLIPMAX))
        logY = log.(clamp.(magY .+ eps_like(magY), EPS_LOG, CLIPMAX))
        lm   = mean(abs.(logY .- logX))

        sc_sum += Float32(sc)
        lm_sum += Float32(lm)
    end

    sc_avg = sc_sum / Float32(R)
    lm_avg = lm_sum / Float32(R)
    total = mr.λ_sc * sc_avg + mr.λ_mag * lm_avg

    return (total=total, sc=sc_avg, logmag=lm_avg)
end
