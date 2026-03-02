abstract type Encoder end

struct Encoder4 <: Encoder
    conv1::Conv
    conv2::Conv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
Encoder4(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int)

Construct an encoder made of two strided low-pass 1D convolutions, a residual convolution, and a dilated residual stack.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden/output channels for the conv layers.
- num_residual_layers::Int: number of residual layers in the residual stack.
- num_residual_hiddens::Int: number of hidden channels inside each residual block.
- kernel_size_ds::Int: kernel size used for the downsampling (strided) convolutions.
- kernel_size_res::Int: kernel size used for the residual convolution and residual blocks.

Returns
- Encoder4: a composite encoder instance containing (conv1, conv2, conv3, residual_stack).
"""
function Encoder4(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int)
    conv1 = Conv(
        (kernel_size_ds,), 
        in_channels => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024)
    )
    conv2 = Conv(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => num_hiddens; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
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
    return Encoder4(conv1, conv2, conv3, residual_stack)
end


"""
Encode inputs using an Encoder4.

Arguments
- encoder::Encoder4 — encoder instance with conv1, conv2, conv3 and residual_stack fields.
- inputs::AbstractArray{Float32,3} — 3‑D input tensor of Float32.

Returns
- AbstractArray{Float32,3} — encoded output after the convolutional layers and residual stack.
"""
function (encoder::Encoder4)(inputs::AbstractArray{Float32, 3})
    x = encoder.conv1(inputs)
    x = relu.(x)

    x = encoder.conv2(x)
    x = relu.(x)

    x = encoder.conv3(x)
    return encoder.residual_stack(x)
end


struct Encoder8 <: Encoder
    conv1::Conv
    conv2::Conv
    conva::Conv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
Create an Encoder8 composed of downsampling and residual convolutions.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden channels/features.
- num_residual_layers::Int: number of residual layers in the residual stack.
- num_residual_hiddens::Int: number of hidden units inside each residual block.
- kernel_size_ds::Int: kernel size used for downsampling convolutions.
- kernel_size_res::Int: kernel size used for residual convolutions.

Returns
- Encoder8: an encoder object consisting of stacked Conv layers and a ResidualStack.
"""
function Encoder8(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int)
    conv1 = Conv(
        (kernel_size_ds,), 
        in_channels => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024)
    )
    conv2 = Conv(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => num_hiddens; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
    )
    conva = Conv(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 512, 256)
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
    return Encoder8(conv1, conv2, conva, conv3, residual_stack)
end

"""
Encode inputs using an Encoder8 instance by applying successive convolutional layers
with ReLU activations followed by the residual stack.

Arguments
- inputs::AbstractArray{Float32,3}: 3D input tensor of Float32 values.

Returns
- AbstractArray{Float32,3}: encoded 3D feature tensor after convolutions and residual stack.
"""
function (encoder::Encoder8)(inputs::AbstractArray{Float32, 3})
    x = encoder.conv1(inputs)
    x = relu.(x)

    x = encoder.conv2(x)
    x = relu.(x)

    x = encoder.conva(x)
    x = relu.(x)

    x = encoder.conv3(x)
    return encoder.residual_stack(x)
end


struct Encoder16 <: Encoder
    conv1::Conv
    conv2::Conv
    conva::Conv
    convb::Conv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
Construct an Encoder16 composed of strided 1D convolutional downsampling layers
and a residual stack.

Arguments
- in_channels::Int: number of input channels
- num_hiddens::Int: number of hidden channels/features
- num_residual_layers::Int: number of residual layers in the ResidualStack
- num_residual_hiddens::Int: number of hidden channels inside residual blocks
- kernel_size_ds::Int: kernel size for downsampling convolutions
- kernel_size_res::Int: kernel size for residual and final convolution

Returns
- Encoder16: an Encoder16 instance containing conv1, conv2, conva, convb, conv3 and a ResidualStack
"""

function Encoder16(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int)
    conv1 = Conv(
        (kernel_size_ds,), 
        in_channels => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024),
    )
    conv2 = Conv(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => num_hiddens; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
    )
    conva = Conv(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 512, 256)
    )
    convb = Conv(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 256, 128)
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
    return Encoder16(conv1, conv2, conva, convb, conv3, residual_stack)
end


"""
Encode input through the Encoder16 convolutional pipeline and residual stack.

Arguments
- inputs::AbstractArray{Float32,3}: 3-D Float32 input tensor.

Returns
- AbstractArray{Float32,3}: 3-D Float32 tensor of encoded features.
"""
function (encoder::Encoder16)(inputs::AbstractArray{Float32, 3})
    x = encoder.conv1(inputs)
    x = relu.(x)

    x = encoder.conv2(x)
    x = relu.(x)

    x = encoder.conva(x)
    x = relu.(x)

    x = encoder.convb(x)
    x = relu.(x)

    x = encoder.conv3(x)
    return encoder.residual_stack(x)
end


struct Encoder32 <: Encoder
    conv1::Conv
    conv2::Conv
    conva::Conv
    convb::Conv
    convc::Conv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
Encoder32(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int)

Construct a 1D convolutional encoder with multiple strided downsampling convolutions followed by a residual stack.

Arguments
- in_channels::Int: number of input channels.
- num_hiddens::Int: number of hidden feature channels.
- num_residual_layers::Int: number of residual layers in the residual stack.
- num_residual_hiddens::Int: hidden channels inside each residual block.
- kernel_size_ds::Int: kernel size for downsampling convolutions.
- kernel_size_res::Int: kernel size for residual and final convolutions.

Returns
- Encoder32: instance combining Conv layers and a ResidualStack for 1D inputs.
"""
function Encoder32(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size_ds::Int, kernel_size_res::Int)
    conv1 = Conv(
        (kernel_size_ds,), 
        in_channels => num_hiddens ÷ 2; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 2048, 1024)
    )
    conv2 = Conv(
        (kernel_size_ds,), 
        num_hiddens ÷ 2 => num_hiddens; 
        stride=2, 
        pad=compute_padding(kernel_size_ds, 1, 2, 1024, 512),
    )
    conva = Conv(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 512, 256)
    )
    convb = Conv(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 256, 128)
    )
    convc = Conv(
        (kernel_size_ds,),
        num_hiddens => num_hiddens;
        stride=2,
        pad=compute_padding(kernel_size_ds, 1, 2, 128, 64)
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
    return Encoder32(conv1, conv2, conva, convb, convc, conv3, residual_stack)
end

"""
Apply an Encoder32 to a 3‑D Float32 input tensor by running it through the encoder's
convolutional layers with ReLU activations and a final residual stack.

Arguments
- inputs::AbstractArray{Float32, 3}: 3‑D input tensor to be encoded.

Returns
- AbstractArray{Float32, 3}: Encoded feature tensor produced by the encoder.
"""
function (encoder::Encoder32)(inputs::AbstractArray{Float32, 3})
    x = encoder.conv1(inputs)
    x = relu.(x)

    x = encoder.conv2(x)
    x = relu.(x)

    x = encoder.conva(x)
    x = relu.(x)

    x = encoder.convb(x)
    x = relu.(x)

    x = encoder.convc(x)
    x = relu.(x)

    x = encoder.conv3(x)
    return encoder.residual_stack(x)
end