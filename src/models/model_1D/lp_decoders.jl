struct LPTConv
    tconv::ConvTranspose
    lpf::LPlayer
end

Flux.@layer LPTConv
Flux.trainable(u::LPTConv) = (u.tconv,) 

"""
LPTConv(k::Int, in_ch::Int, out_ch::Int; stride::Int=2, pad::Int=1, taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Create a transposed convolution layer paired with a low-pass filter.
LP parameters are not trainable parameters.

Arguments
- k::Int: kernel size (1D).
- in_ch::Int: input channel count.
- out_ch::Int: output channel count.
- stride::Int=2: convolution stride.
- pad::Int=1: convolution padding.
- taps::Int=63: number of FIR filter taps.
- beta::Float64=8.0: window shaping parameter for the filter.
- cutoff_factor::Float64=0.9: factor used to compute filter cutoff (scaled by 1/stride).

Returns
- LPTConv: a composite containing the ConvTranspose layer and the LPlayer low-pass filter.
"""
function LPTConv(k::Int, in_ch::Int, out_ch::Int; stride::Int=2, pad::Int=1, taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    tconv = Flux.ConvTranspose(
        (k,),
        in_ch=>out_ch;
        stride=stride,
        pad=pad, 
        outpad=1
    )
    cutoff = min(cutoff_factor / stride, 0.35)
    lpf = LPlayer(
        out_ch;
        taps=taps,
        cutoff=cutoff,
        beta=beta
    )

    return LPTConv(tconv, lpf)
end

"""
Apply the LPTConv model.

Arguments
- u::LPTConv: the LPTConv layer/functor
- x::AbstractArray{Float32, 3}: input tensor

Returns
- AbstractArray{Float32, 3}: filtered output tensor
"""
function (u::LPTConv)(x::AbstractArray{Float32, 3}) 
    return u.lpf.layer(u.tconv(x))
end


struct LPDecoder4 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_trans1::LPTConv
    conv_trans2::LPTConv
end

"""
LPDecoder4(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct a 1D LP decoder consisting of an initial convolution, a residual stack, and two LPTConv upsampling layers.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden channels for conv and residual stack.
- out_channels::Int: number of output channels.
- num_residual_layers::Int: number of residual layers.
- num_residual_hiddens::Int: hidden channels inside residual layers.
- kernel_size_ds::Int: kernel size for the LPTConv upsampling layers.
- kernel_size_res::Int: kernel size for the initial conv and residual blocks.
- taps::Int=63: (keyword) number of FIR taps for LPTConv filters.
- beta::Float64=8.0: (keyword) beta parameter for filter design.
- cutoff_factor::Float64=0.9: (keyword) cutoff factor for filter design.

Returns
- LPDecoder4: an assembled decoder model instance.
"""
function LPDecoder4(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int, 
    kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = Conv(
        (kernel_size_res,), 
        in_channels => num_hiddens; 
        stride=1, 
        pad=compute_paddingT(kernel_size_res, 1)
        )
    residual_stack = ResidualStack(
        num_hiddens, 
        num_hiddens, 
        num_residual_layers, 
        num_residual_hiddens,
        kernel_size_res
    )
    conv_trans1 = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens ÷ 2; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans2 = LPTConv(
        kernel_size_ds, 
        num_hiddens ÷ 2, 
        out_channels; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    return LPDecoder4(conv1, residual_stack, conv_trans1, conv_trans2)
end

"""
LPDecoder4 callable: apply the LP decoder to a 3‑D Float32 tensor.

Arguments
- x::AbstractArray{Float32,3}: input 3‑D Float32 tensor.

Returns
- AbstractArray{Float32,3}: decoded 3‑D Float32 tensor output.
"""
function (decoder::LPDecoder4)(x::AbstractArray{Float32, 3})
    x = decoder.conv1(x)
    x = decoder.residual_stack(x)  

    x = decoder.conv_trans1(x)
    x = relu.(x)

    return decoder.conv_trans2(x)
end


struct LPDecoder8 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_transa::LPTConv
    conv_trans1::LPTConv
    conv_trans2::LPTConv
end

"""
LPDecoder8(in_channels::Int, num_hiddens::Int, out_channels::Int,
           num_residual_layers::Int, num_residual_hiddens::Int,
           kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63,
           beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an LP-based decoder consisting of an initial convolution, a residual stack,
and three LPTConv upsampling blocks.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden channels throughout the decoder.
- out_channels::Int: number of output channels.
- num_residual_layers::Int: number of layers in the residual stack.
- num_residual_hiddens::Int: hidden channels inside each residual layer.
- kernel_size_ds::Int: kernel size for LPTConv (upsampling) layers.
- kernel_size_res::Int: kernel size for initial convolution and residual blocks.
- taps::Int=63: number of filter taps for LPTConv.
- beta::Float64=8.0: beta parameter for LPTConv filter design.
- cutoff_factor::Float64=0.9: cutoff factor for LPTConv filter design.

Returns
- LPDecoder8: an instance of the LPDecoder8 model.
"""
function LPDecoder8(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int, 
    kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = Conv(
        (kernel_size_res,), 
        in_channels => num_hiddens; 
        stride=1, 
        pad=compute_paddingT(kernel_size_res, 1),
        )
    residual_stack = ResidualStack(
        num_hiddens, 
        num_hiddens, 
        num_residual_layers, 
        num_residual_hiddens,
        kernel_size_res
    )
    conv_transa = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 256, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans1 = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens ÷ 2; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans2 = LPTConv(
        kernel_size_ds, 
        num_hiddens ÷ 2, 
        out_channels; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    return LPDecoder8(conv1, residual_stack, conv_transa, conv_trans1, conv_trans2)
end

"""
    (decoder::LPDecoder8)(x::AbstractArray{Float32,3})

Model pass of LPDecoder8.

Arguments
- decoder::LPDecoder8: the decoder instance.
- x::AbstractArray{Float32,3}: input Float32 tensor.

Returns
- AbstractArray{Float32,3}: decoded Float32 output tensor.
"""
function (decoder::LPDecoder8)(x::AbstractArray{Float32, 3})
    x = decoder.conv1(x)
    x = decoder.residual_stack(x)

    x = decoder.conv_transa(x)
    x = relu.(x)

    x = decoder.conv_trans1(x)
    x = relu.(x)

    return decoder.conv_trans2(x)
end


struct LPDecoder16 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_transa::LPTConv
    conv_transb::LPTConv
    conv_trans1::LPTConv
    conv_trans2::LPTConv
end

"""
LPDecoder16(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Constructs a 1D LPDecoder model composed of an initial conv, a residual stack and a sequence of learnable polyphase transposed convolutions.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels/features.
- out_channels::Int: Number of output channels.
- num_residual_layers::Int: Number of layers in the residual stack.
- num_residual_hiddens::Int: Hidden channels inside residual blocks.
- kernel_size_ds::Int: Kernel size for the down/up-sampling (transposed) convolutions.
- kernel_size_res::Int: Kernel size for the residual/convolutional layers.
- taps::Int=63: Number of taps for the LPTConv filters.
- beta::Float64=8.0: Beta parameter controlling the LPTConv windowing.
- cutoff_factor::Float64=0.9: Cutoff factor for LPTConv filter design.

Returns
- LPDecoder16: An instance of the LPDecoder16 model.
"""
function LPDecoder16(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int, 
    kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = Conv(
        (kernel_size_res,), 
        in_channels => num_hiddens; 
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
    conv_transa = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 128, 256),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_transb = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 256, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans1 = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens ÷ 2; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans2 = LPTConv(
        kernel_size_ds, 
        num_hiddens ÷ 2, 
        out_channels; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    return LPDecoder16(conv1, residual_stack, conv_transa, conv_transb, conv_trans1, conv_trans2)
end

"""
LPDecoder16 callable: decodes a 3‑D Float32 tensor through the decoder network.

Arguments
- x::AbstractArray{Float32, 3}: input 3‑dimensional Float32 array.

Returns
- AbstractArray{Float32, 3}: decoded 3‑dimensional Float32 output.
"""
function (decoder::LPDecoder16)(x::AbstractArray{Float32, 3})
    x = decoder.conv1(x)
    x = decoder.residual_stack(x)

    x = decoder.conv_transa(x)
    x = relu.(x)

    x = decoder.conv_transb(x)
    x = relu.(x)

    x = decoder.conv_trans1(x)
    x = relu.(x)

    return decoder.conv_trans2(x)
end


struct LPDecoder32 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_transa::LPTConv
    conv_transb::LPTConv
    conv_transc::LPTConv
    conv_trans1::LPTConv
    conv_trans2::LPTConv
end

"""
LPDecoder32(in_channels::Int, num_hiddens::Int, out_channels::Int,
            num_residual_layers::Int, num_residual_hiddens::Int,
            kernel_size_ds::Int, kernel_size_res::Int;
            taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an LPDecoder32 model consisting of an initial convolution, a residual stack,
and a sequence of LPTConv upsampling blocks.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels used throughout the model.
- out_channels::Int: Number of output channels.
- num_residual_layers::Int: Number of layers in the residual stack.
- num_residual_hiddens::Int: Number of hidden channels inside residual layers.
- kernel_size_ds::Int: Kernel size for the down/up-sampling LPTConv layers.
- kernel_size_res::Int: Kernel size for the initial conv and residual convolutions.
- taps::Int=63: (keyword) Number of filter taps for the LPTConv filters.
- beta::Float64=8.0: (keyword) Beta parameter for the LPTConv windowing.
- cutoff_factor::Float64=0.9: (keyword) Cutoff factor for LPTConv filters.

Returns
- LPDecoder32: A constructed LPDecoder32 model instance.
"""
function LPDecoder32(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int, 
    kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    conv1 = Conv(
        (kernel_size_res,), 
        in_channels => num_hiddens; 
        stride=1, 
        pad=compute_paddingT(kernel_size_res, 1),
    )
    residual_stack = ResidualStack(
        num_hiddens, 
        num_hiddens, 
        num_residual_layers, 
        num_residual_hiddens,
        kernel_size_res
    )
    conv_transa = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 64, 128),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_transb = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 128, 256),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_transc = LPTConv(
        kernel_size_ds, 
        num_hiddens, 
        num_hiddens ÷ 2; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 256, 512),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans1 = LPTConv(
        kernel_size_ds, 
        num_hiddens ÷ 2, 
        num_hiddens ÷ 2; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    conv_trans2 = LPTConv(
        kernel_size_ds, 
        num_hiddens ÷ 2, 
        out_channels; 
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        taps=taps,
        beta=beta,
        cutoff_factor=cutoff_factor
    )
    return LPDecoder32(conv1, residual_stack, conv_transa, conv_transb, conv_transc, conv_trans1, conv_trans2)
end

"""
LPDecoder32 forward pass.

Applies the decoder's convolutional and residual blocks to the input tensor.

# Arguments
- `decoder::LPDecoder32` : decoder instance
- `x::AbstractArray{Float32,3}` : 3‑D input array (channels, length, batch)

# Returns
- `AbstractArray{Float32,3}` : decoded output tensor (Float32)
"""
function (decoder::LPDecoder32)(x::AbstractArray{Float32, 3})
    x = decoder.conv1(x)
    x = decoder.residual_stack(x)

    x = decoder.conv_transa(x)
    x = relu.(x)

    x = decoder.conv_transb(x)
    x = relu.(x)

    x = decoder.conv_transc(x)
    x = relu.(x)

    x = decoder.conv_trans1(x)
    x = relu.(x)

    return decoder.conv_trans2(x)
end