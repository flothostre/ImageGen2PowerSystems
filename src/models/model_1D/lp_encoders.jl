struct LPConv
    lpf::LPlayer
    conv::Conv
end
Flux.@layer LPConv
Flux.trainable(a::LPConv) = (a.conv,) # only Conv to be learned, LPF frozen

"""
LPConv(k::Int, in_ch::Int, out_ch::Int; stride::Int=2, pad::Int=1, taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct a low‑pass filtered 1D convolution layer by creating an LPlayer and a Flux.Conv, and returning LPConv(lpf, conv).
Convolutinal parameters are not trainable.

Arguments
- k::Int: kernel size
- in_ch::Int: number of input channels
- out_ch::Int: number of output channels
- stride::Int=2: convolution stride
- pad::Int=1: convolution padding
- taps::Int=63: number of filter taps for the LPlayer
- beta::Float64=8.0: filter sharpness parameter for the LPlayer
- cutoff_factor::Float64=0.9: factor used to compute the cutoff frequency (scaled by stride)

Returns
- LPConv: composite layer combining the low‑pass filter and the convolution
"""
function LPConv(k::Int, in_ch::Int, out_ch::Int; stride::Int=2, pad::Int=1, taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    cutoff = min(cutoff_factor / stride, 0.35)
    lpf  = LPlayer(
        in_ch;
        taps=taps,
        cutoff=cutoff,
        beta=beta
    )
    conv = Flux.Conv(
        (k,),
        in_ch=>out_ch;
        stride=stride,
        pad=pad,
    )
    return LPConv(lpf, conv)
end

"""
Apply the layer of the low-pass filter then a convolution.

# Arguments
- x::AbstractArray{Float32, 3}: 3‑D input tensor (Float32).

# Returns
- AbstractArray{Float32, 3}: Output tensor after low-pass filtering and convolution.
"""
function (a::LPConv)(x::AbstractArray{Float32, 3}) 
    return a.conv(a.lpf.layer(x))
end


struct LPEncoder4 <: Encoder
    conv1::LPConv
    conv2::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
LPEncoder4(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an LP-based encoder consisting of two low-pass downsampling LPConv layers, a residual convolution, and a residual stack.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden/output channels used in the encoder.
- num_residual_layers::Int: number of residual layers in the ResidualStack.
- num_residual_hiddens::Int: number of hidden channels inside each residual block.
- kernel_size_ds::Int: kernel size for the downsampling (LPConv) layers.
- kernel_size_res::Int: kernel size for the residual convolution layers.

Keyword arguments
- taps::Int=63: number of filter taps for LPConv.
- beta::Float64=8.0: beta parameter for LP filter design.
- cutoff_factor::Float64=0.9: cutoff frequency scaling factor for LP filters.

Returns
- LPEncoder4: an initialized encoder model (instance of LPEncoder4).
"""
function LPEncoder4(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int,
    kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = LPConv(
        kernel_size_ds, 
        in_channels,
        num_hiddens ÷ 2;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv2 = LPConv(
        kernel_size_ds,
        num_hiddens ÷ 2,
        num_hiddens; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor

    )
    conv3 = Conv(
        (kernel_size_res,), 
        num_hiddens => num_hiddens; 
        stride=1, 
        pad=compute_padding(kernel_size_res, 1),
        )
    residual_stack = ResidualStack(
        num_hiddens, 
        num_hiddens, 
        num_residual_layers, 
        num_residual_hiddens,
        kernel_size_res
    )
    return LPEncoder4(conv1, conv2, conv3, residual_stack)
end

"""
Call method for LPEncoder4.

Arguments
- x::AbstractArray{Float32,3}: Input 3D Float32 array (e.g., channels × length × batch).

Returns
- AbstractArray{Float32,3}: Output feature map after sequential convolutions and the residual stack.
"""
function (lpencoder::LPEncoder4)(x::AbstractArray{Float32, 3})
    x = lpencoder.conv1(x)
    x = relu.(x)

    x = lpencoder.conv2(x)
    x = relu.(x)

    x = lpencoder.conv3(x)
    return lpencoder.residual_stack(x)
end


struct LPEncoder8 <: Encoder
    conv1::LPConv
    conv2::LPConv
    conva::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
Construct an LP-based encoder composed of low-pass convolutional layers and a residual stack.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden feature channels.
- num_residual_layers::Int: number of residual layers.
- num_residual_hiddens::Int: hidden channels inside each residual block.
- kernel_size_ds::Int: kernel size for downsampling LPConv layers.
- kernel_size_res::Int: kernel size for the residual Conv layer.
- taps::Int=63: number of FIR taps for LP filters.
- beta::Float64=8.0: beta parameter for filter design.
- cutoff_factor::Float64=0.9: cutoff frequency factor relative to Nyquist.

Returns
- LPEncoder8: an LPEncoder8 instance containing the configured LPConv/Conv layers and ResidualStack.
"""
function LPEncoder8(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int,
    kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = LPConv(
        kernel_size_ds,
        in_channels,
        num_hiddens ÷ 2;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv2 = LPConv(
        kernel_size_ds,
        num_hiddens ÷ 2,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conva = LPConv(
        kernel_size_ds,
        num_hiddens,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 512, 256),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv3 = Conv(
        (kernel_size_res,),
        num_hiddens => num_hiddens;
        stride=1,
        pad=compute_padding(kernel_size_res, 1)
    )
    residual_stack = ResidualStack(
        num_hiddens,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        kernel_size_res
    )
    return LPEncoder8(conv1, conv2, conva, conv3, residual_stack)
end

"""
lpencoder(x)

Apply LPEncoder8 forward pass.

Arguments
- x::AbstractArray{Float32,3}: 3D Float32 input tensor.

Returns
- AbstractArray{Float32,3}: Output tensor after convolutions and residual stack.
"""
function (lpencoder::LPEncoder8)(x::AbstractArray{Float32, 3})
    x = lpencoder.conv1(x)
    x = relu.(x)

    x = lpencoder.conv2(x)
    x = relu.(x)

    x = lpencoder.conva(x)
    x = relu.(x)

    x = lpencoder.conv3(x)
    return lpencoder.residual_stack(x)
end


struct LPEncoder16 <: Encoder
    conv1::LPConv
    conv2::LPConv
    conva::LPConv
    convb::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
LPEncoder16(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Constructs a low‑pass encoder that performs 16× downsampling using LPConv blocks followed by a residual stack.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels/features.
- num_residual_layers::Int: Number of residual layers in the ResidualStack.
- num_residual_hiddens::Int: Hidden channels inside each residual layer.
- kernel_size_ds::Int: Kernel size for downsampling LPConv layers.
- kernel_size_res::Int: Kernel size for the residual Conv layer.
- taps::Int=63: Number of FIR filter taps used by LPConv.
- beta::Float64=8.0: Kaiser window beta parameter for LPConv filters.
- cutoff_factor::Float64=0.9: Cutoff frequency scaling factor for LPConv filters.

Returns
- LPEncoder16: An encoder instance composed of configured LPConv layers and a ResidualStack.
"""
function LPEncoder16(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int,
    kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = LPConv(
        kernel_size_ds,
        in_channels,
        num_hiddens ÷ 2;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv2 = LPConv(
        kernel_size_ds,
        num_hiddens ÷ 2,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conva = LPConv(
        kernel_size_ds,
        num_hiddens,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 512, 256),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    convb = LPConv(
        kernel_size_ds,
        num_hiddens,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 256, 128),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv3 = Conv(
        (kernel_size_res,),
        num_hiddens => num_hiddens;
        stride=1,
        pad=compute_padding(kernel_size_res, 1)
    )
    residual_stack = ResidualStack(
        num_hiddens,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        kernel_size_res
    )
    return LPEncoder16(conv1, conv2, conva, convb, conv3, residual_stack)
end

"""
LPEncoder16 call operator.

# Arguments
- `x::AbstractArray{Float32, 3}`: 3‑D input tensor (e.g. channels × length × batch).

# Returns
- `AbstractArray{Float32, 3}`: Encoded output tensor of Float32; exact shape depends on the encoder's layer configuration.
"""
function (lpencoder16::LPEncoder16)(x::AbstractArray{Float32, 3})
    x = lpencoder16.conv1(x)
    x = relu.(x)

    x = lpencoder16.conv2(x)
    x = relu.(x)

    x = lpencoder16.conva(x)
    x = relu.(x)

    x = lpencoder16.convb(x)
    x = relu.(x)

    x = lpencoder16.conv3(x)
    return lpencoder16.residual_stack(x)
end


struct LPEncoder32 <: Encoder
    conv1::LPConv
    conv2::LPConv
    conva::LPConv
    convb::LPConv
    convc::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
LPEncoder32(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
            kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an LPEncoder32 composed of low-pass LPConv downsampling layers, a residual Conv layer and a ResidualStack.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels/features.
- num_residual_layers::Int: Number of residual layers in the residual stack.
- num_residual_hiddens::Int: Number of hidden channels inside residual blocks.
- kernel_size_ds::Int: Kernel size for downsampling LPConv layers.
- kernel_size_res::Int: Kernel size for the residual Conv layer.
- taps::Int=63: Number of FIR taps for the LP filters (keyword).
- beta::Float64=8.0: Kaiser window beta parameter for filter design (keyword).
- cutoff_factor::Float64=0.9: Cutoff frequency factor relative to Nyquist (keyword).

Returns
- LPEncoder32: An LPEncoder32 object built from the configured layers.
"""
function LPEncoder32(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int,
    kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = LPConv(
        kernel_size_ds,
        in_channels,
        num_hiddens ÷ 2;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv2 = LPConv(
        kernel_size_ds,
        num_hiddens ÷ 2,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conva = LPConv(
        kernel_size_ds,
        num_hiddens,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 512, 256),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    convb = LPConv(
        kernel_size_ds,
        num_hiddens,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 256, 128),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    convc = LPConv(
        kernel_size_ds,
        num_hiddens,
        num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 128, 64),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv3 = Conv(
        (kernel_size_res,),
        num_hiddens => num_hiddens;
        stride=1,
        pad=compute_padding(kernel_size_res, 1)
    )
    residual_stack = ResidualStack(
        num_hiddens,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        kernel_size_res
    )
    return LPEncoder32(conv1, conv2, conva, convb, convc, conv3, residual_stack)
end

"""
LPEncoder32(x::AbstractArray{Float32,3})

Forward pass for an LPEncoder32.

Arguments
- x::AbstractArray{Float32,3}: Input tensor of Float32 (e.g., channels × length × batch).

Returns
- AbstractArray{Float32,3}: Output tensor after the encoder's convolutions and residual stack.
"""
function (lpencoder32::LPEncoder32)(x::AbstractArray{Float32, 3})
    x = lpencoder32.conv1(x)
    x = relu.(x)

    x = lpencoder32.conv2(x)
    x = relu.(x)

    x = lpencoder32.conva(x)
    x = relu.(x)

    x = lpencoder32.convb(x)
    x = relu.(x)

    x = lpencoder32.convc(x)
    x = relu.(x)

    x = lpencoder32.conv3(x)
    return lpencoder32.residual_stack(x)
    
end