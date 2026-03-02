abstract type Decoder end

struct Decoder4 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_trans1::ConvTranspose
    conv_trans2::ConvTranspose
end

"""
Construct a 1D decoder consisting of an initial convolution, a residual stack,
and two transposed convolutions for upsampling.

Arguments
- in_channels::Int: number of input channels
- num_hiddens::Int: number of hidden channels in intermediate layers
- out_channels::Int: number of output channels
- num_residual_layers::Int: number of residual layers in the residual stack
- num_residual_hiddens::Int: number of hidden channels inside each residual block
- kernel_size_ds::Int: kernel size for the down/up-sampling (transpose conv) layers
- kernel_size_res::Int: kernel size for residual and initial conv layers

Returns
- Decoder4: a Decoder4 instance containing (conv1, residual_stack, conv_trans1, conv_trans2)
"""
function Decoder4(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    kernel_size_ds::Int, kernel_size_res::Int)
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
    conv_trans1 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        outpad=1
    )
    conv_trans2 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => out_channels; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        outpad=1
    )
    return Decoder4(conv1, residual_stack, conv_trans1, conv_trans2)
end

"""
Model call for Decoder4.
Arguments
- inputs::AbstractArray{Float32,3}: Input 3‑D Float32 tensor.

Returns
- AbstractArray{Float32,3}: Decoded output as a 3‑D Float32 tensor.
"""
function (decoder::Decoder4)(inputs::AbstractArray{Float32, 3})
    x = decoder.conv1(inputs)  
    x = decoder.residual_stack(x)  

    x = decoder.conv_trans1(x)  
    x = relu.(x) 
    return decoder.conv_trans2(x) 
end


struct Decoder8 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_transa::ConvTranspose
    conv_trans1::ConvTranspose
    conv_trans2::ConvTranspose
end

"""
Decoder8(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
         kernel_size_ds::Int, kernel_size_res::Int)

Construct a Decoder8 composed of an initial convolution, a residual stack, and three transposed convolutions for upsampling.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden/channel features used throughout the decoder.
- out_channels::Int: number of output channels.
- num_residual_layers::Int: number of layers in the residual stack.
- num_residual_hiddens::Int: number of hidden units inside each residual layer.
- kernel_size_ds::Int: kernel size for the transposed (upsampling) convolutions.
- kernel_size_res::Int: kernel size for the initial convolution and residual blocks.

Returns
- Decoder8: an instance of the Decoder8 struct containing the configured layers.
"""
function Decoder8(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    kernel_size_ds::Int, kernel_size_res::Int)
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
    conv_transa = ConvTranspose(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 256, 512),
        outpad=1
    )
    conv_trans1 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        outpad=1
    )
    conv_trans2 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => out_channels; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        outpad=1
    )
    return Decoder8(conv1, residual_stack, conv_transa, conv_trans1, conv_trans2)
end

"""
    (decoder::Decoder8)(inputs::AbstractArray{Float32,3})

Model pass for Decoder8.

Arguments
- inputs::AbstractArray{Float32,3}: Input tensor of Float32 values.

Returns
- AbstractArray{Float32,3}: Decoded output tensor of Float32 values.
"""
function (decoder::Decoder8)(inputs::AbstractArray{Float32, 3})
    x = decoder.conv1(inputs)  
    x = decoder.residual_stack(x)  

    x = decoder.conv_transa(x)
    x = relu.(x)

    x = decoder.conv_trans1(x)  
    x = relu.(x)  
    return decoder.conv_trans2(x) 
end


struct Decoder16 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_transa::ConvTranspose
    conv_transb::ConvTranspose
    conv_trans1::ConvTranspose
    conv_trans2::ConvTranspose
end

"""
Construct a Decoder16 model consisting of an initial convolution, a residual stack,
and a sequence of transposed convolutions for upsampling.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden channels/features.
- out_channels::Int: number of output channels.
- num_residual_layers::Int: number of residual layers in the residual stack.
- num_residual_hiddens::Int: number of hidden channels inside residual blocks.
- kernel_size_ds::Int: kernel size for the down/up-sampling (transposed conv) layers.
- kernel_size_res::Int: kernel size for the initial conv and residual layers.

Returns
- Decoder16: an instance of the Decoder16 model containing the assembled layers.
"""
function Decoder16(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    kernel_size_ds::Int, kernel_size_res::Int)
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
    conv_transa = ConvTranspose(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 128, 256),
        outpad=1
    )
    conv_transb = ConvTranspose(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 256, 512),
        outpad=1
    )
    conv_trans1 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        outpad=1
    )
    conv_trans2 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => out_channels; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        outpad=1
    )
    return Decoder16(conv1, residual_stack, conv_transa, conv_transb, conv_trans1, conv_trans2)
end

"""
Model pass of Decoder16.

Arguments
- inputs::AbstractArray{Float32,3}: 3D Float32 input tensor to be decoded.

Returns
- AbstractArray{Float32,3}: decoded 3D Float32 output tensor.
"""
function (decoder::Decoder16)(inputs::AbstractArray{Float32, 3})
    x = decoder.conv1(inputs)  # First convolutional layer
    x = decoder.residual_stack(x)  # Residual stack

    x = decoder.conv_transa(x)
    x = relu.(x)

    x = decoder.conv_transb(x)
    x = relu.(x)

    x = decoder.conv_trans1(x)  
    x = relu.(x)  
    return decoder.conv_trans2(x)  
end


struct Decoder32 <: Decoder
    conv1::Conv
    residual_stack::ResidualStack
    conv_transa::ConvTranspose
    conv_transb::ConvTranspose
    conv_transc::ConvTranspose
    conv_trans1::ConvTranspose
    conv_trans2::ConvTranspose
end

"""
Decoder32(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    kernel_size_ds::Int, kernel_size_res::Int)

Construct a 1D decoder composed of an initial convolution, a residual stack, and a sequence of transposed convolutions.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden channels/features.
- out_channels::Int: number of output channels.
- num_residual_layers::Int: number of residual layers in the residual stack.
- num_residual_hiddens::Int: hidden channels inside each residual layer.
- kernel_size_ds::Int: kernel size for transposed (upsampling) convolutions.
- kernel_size_res::Int: kernel size for residual convolutions.

Returns
- Decoder32: an instance of the Decoder32 model.
"""
function Decoder32(in_channels::Int, num_hiddens::Int, out_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    kernel_size_ds::Int, kernel_size_res::Int)
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
    conv_transa = ConvTranspose(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 64, 128),
        outpad=1
    )
    conv_transb = ConvTranspose(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 128, 256),
        outpad=1
    )
    conv_transc = ConvTranspose(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_paddingT(kernel_size_ds, 1, 2, 256, 512),
        outpad=1
    )
    conv_trans1 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 512, 1024),
        outpad=1
    )
    conv_trans2 = ConvTranspose(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => out_channels; 
        stride=2, 
        pad=compute_paddingT(kernel_size_ds, 1, 2, 1024, 2048),
        outpad=1
    )
    return Decoder32(conv1, residual_stack, conv_transa, conv_transb, conv_transc, conv_trans1, conv_trans2)
end

"""
Call method for Decoder32.

Arguments
- decoder::Decoder32: Decoder instance containing convolutional, residual and transposed convolution layers.
- inputs::AbstractArray{Float32,3}: 3D Float32 input tensor (features × length × batch).

Returns
- AbstractArray{Float32,3}: Decoded 3D Float32 output tensor after upsampling/transposed convolutions.
"""
function (decoder::Decoder32)(inputs::AbstractArray{Float32, 3})
    x = decoder.conv1(inputs)  # First convolutional layer
    x = decoder.residual_stack(x)  # Residual stack

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