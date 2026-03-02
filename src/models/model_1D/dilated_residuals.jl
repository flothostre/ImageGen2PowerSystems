# Define the DilatedResidual block
struct DilatedResidual
    lnorm1::LayerNorm
    mconv1::Conv
    lnorm2::LayerNorm
    conv::Conv
    lnorm3::LayerNorm
    mconv2::Conv
    contribution_factor::Float32
end

"""
    DilatedResidual(in_channels::Int, num_hiddens::Int, num_residual_hiddens::Int)

Creates a residual block (skip connections) for a neural network.

# Arguments
- `in_channels::Int`: The number of input channels for the residual block.
- `num_hiddens::Int`: The number of hidden units in the intermediate layers of the block.
- `num_residual_hiddens::Int`: The number of hidden units in the residual layers.

# Returns
A residual block that can be used as part of a neural network architecture.
"""
function DilatedResidual(in_channels::Int, num_residual_hiddens::Int, dilation::Int, kernel_size::Int, contribution_factor::Float32)
    lnorm1 = LayerNorm(in_channels)
    mconv1 = Conv(
        (1,),
        in_channels => num_residual_hiddens; 
        stride=1, 
        pad=compute_padding(1, 1), 
        bias=false,
        dilation=1
        )

    lnorm2 = LayerNorm(num_residual_hiddens)
    conv = Conv(
        (kernel_size,),
        num_residual_hiddens => num_residual_hiddens;
        stride=1,
        pad=compute_padding(kernel_size, dilation),
        bias=false,
        dilation=dilation
    )

    lnorm3 = LayerNorm(num_residual_hiddens)
    mconv2 = Conv(
        (1,),
        num_residual_hiddens => in_channels;
        stride=1,
        pad=compute_padding(1, 1),
        bias=false,
        dilation=1,
        init=zeros32
    )
    contribution_factor = contribution_factor

    return DilatedResidual(lnorm1, mconv1, lnorm2, conv, lnorm3, mconv2, contribution_factor)
end

"""
Apply a DilatedResidual block to a 3D Float32 tensor.

Arguments
- residual::DilatedResidual: the residual block instance containing normalization, activation, and convolution layers, plus a contribution_factor.
- x::AbstractArray{Float32,3}: input tensor (3D Float32 array).

Returns
- AbstractArray{Float32,3}: output tensor of the same shape as `x`, computed as the input plus the block's transformed output scaled by `contribution_factor`.
"""
function (residual::DilatedResidual)(x::AbstractArray{Float32, 3})
    x_in = x
    x = permutedims(residual.lnorm1(permutedims(x, (2,1,3))), (2,1,3))
    x = gelu.(x)
    x = residual.mconv1(x)

    x = permutedims(residual.lnorm2(permutedims(x, (2,1,3))), (2,1,3))
    x = gelu.(x)
    x = residual.conv(x)

    x = permutedims(residual.lnorm3(permutedims(x, (2,1,3))), (2,1,3))
    x = gelu.(x)
    x = residual.mconv2(x)

    return x_in .+ x .* residual.contribution_factor
end

# Define the DilatedResidualStack
struct DilatedResidualStack
    layers::Vector{DilatedResidual}
end

"""
    DilatedResidualStack(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int)

Creates a residual stack module for use in a neural network which consists of multiple residual blocks.

# Arguments
- `in_channels::Int`: The number of input channels to the residual stack.
- `num_hiddens::Int`: The number of hidden units in the intermediate layers of the stack.
- `num_residual_layers::Int`: The number of residual layers in the stack.
- `num_residual_hiddens::Int`: The number of hidden units in each residual layer.

# Returns
A residual stack module.
"""
function DilatedResidualStack(in_channels::Int, num_residual_layers::Int, num_residual_hiddens::Int, dilations::Vector{Int}, kernel_size::Int, dires_contribution_factor::Float64)
    layers = [
        DilatedResidual(
            in_channels, 
            num_residual_hiddens,
            dilations[i],
            kernel_size, 
            Float32(dires_contribution_factor) 
        ) for i in 1:num_residual_layers
        ]
    return DilatedResidualStack(layers)
end

function (stack::DilatedResidualStack)(x::AbstractArray{Float32, 3})
"""
Apply the DilatedResidualStack to a 3‑D Float32 input by sequentially applying each layer.

Arguments
- stack::DilatedResidualStack: the stack containing the residual layers
- x::AbstractArray{Float32,3}: input tensor

Returns
- AbstractArray{Float32,3}: output tensor after processing by the stack
"""
    for layer in stack.layers
        x = layer(x)
    end
    return x 
end