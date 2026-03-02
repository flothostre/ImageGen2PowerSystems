# first: definition of individual layer

struct LPlayer
    layer::Conv
end
Flux.@layer LPlayer
Flux.trainable(::LPlayer) = ()  

"""
sinc_lowpass(taps::Int, cutoff::Float64; beta::Float64=8.0)

Compute an odd-length low-pass FIR filter kernel using a windowed sinc (Kaiser) design.

Arguments
- taps::Int: odd number of filter taps (kernel length).
- cutoff::Float64: normalized cutoff frequency in (0, 1) (1.0 = Nyquist).
- beta::Float64=8.0: Kaiser window beta parameter (optional).

Returns
- Vector{Float32}: normalized filter coefficients (sums to 1).
"""
function sinc_lowpass(taps::Int, cutoff::Float64; beta::Float64=8.0)
    @assert isodd(taps) "use an odd number of taps for symmetric linear phase"
    @assert 0 < cutoff < 1
    n = 0:taps-1
    m = n .- (taps-1)/2 
    h = 2*cutoff .* sinc.(2*cutoff .* m)
    w = kaiser(taps, beta)
    h .*= w
    h ./= sum(h) 

    return Float32.(h)
end

"""
LPlayer(in_ch::Int; taps::Int=55, cutoff::Float64=0.45, beta::Float64=8.0)

Create a depthwise 1D low-pass layer initialized with a sinc-based filter; LP kernel parameters are not trainable.

Arguments
- in_ch::Int: number of input channels.
- taps::Int=55: length of the FIR filter kernel.
- cutoff::Float64=0.45: normalized cutoff frequency (0 < cutoff ≤ 0.5).
- beta::Float64=8.0: Kaiser window beta parameter.

Returns
- A Flux depthwise 1D convolution layer (no bias) with weights set to the low-pass kernel.
"""
function LPlayer(in_ch::Int; taps::Int=55, cutoff::Float64=0.45, beta::Float64=8.0)
    h = sinc_lowpass(taps, cutoff; beta=beta)
    lpf = Flux.Conv(
        (taps,),
        in_ch=>in_ch;
        stride=1,
        pad=Flux.SamePad(),
        groups=in_ch,
        bias=false
    )
    for c in 1:in_ch
        lpf.weight[:, 1, c] .= h
    end

    return LPlayer(lpf)
end