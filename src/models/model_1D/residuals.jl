# Define the Residual block
struct Residual
    block::Chain
end

"""
    Residual(in_channels::Int, num_hiddens::Int, num_residual_hiddens::Int)

Creates a residual block (skip connections) for a neural network.

# Arguments
- `in_channels::Int`: The number of input channels for the residual block.
- `num_hiddens::Int`: The number of hidden units in the intermediate layers of the block.
- `num_residual_hiddens::Int`: The number of hidden units in the residual layers.

# Returns
A residual block that can be used as part of a neural network architecture.
"""
function Residual(in_channels::Int, num_hiddens::Int, num_residual_hiddens::Int, kernel_size::Int)
    block = Chain(
        x -> relu.(x),  # First ReLU activation
        Conv(
            (kernel_size,),
            in_channels => num_residual_hiddens;
            stride=1, 
            pad=compute_padding(kernel_size, 1),
            bias=false,
        ),  
        x -> relu.(x),  
        Conv(
            (1,), 
            num_residual_hiddens => num_hiddens; 
            stride=1, 
            bias=false,
        )  
    )
    return Residual(block)
end

"""
    (::Residual)(x)

Applies the `Residual` layer to the input `x` as callable object.

# Arguments
- `x`: The input data to which the residual operation will be applied.
"""
function (residual::Residual)(x::AbstractArray{Float32, 3})
    return x .+ residual.block(x)  
end


# Define the ResidualStack
struct ResidualStack
    layers::Vector{Residual}
end

"""
    ResidualStack(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int)

Creates a residual stack module for use in a neural network which consists of multiple residual blocks.

# Arguments
- `in_channels::Int`: The number of input channels to the residual stack.
- `num_hiddens::Int`: The number of hidden units in the intermediate layers of the stack.
- `num_residual_layers::Int`: The number of residual layers in the stack.
- `num_residual_hiddens::Int`: The number of hidden units in each residual layer.

# Returns
A residual stack module.
"""
function ResidualStack(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int, kernel_size::Int)
    layers = [
        Residual(
            in_channels, 
            num_hiddens, 
            num_residual_hiddens,
            kernel_size
        ) for _ in 1:num_residual_layers
        ]
    return ResidualStack(layers)
end

"""
    (::ResidualStack)(x)

Applies the `ResidualStack` to the input `x`. This function allows an instance of 
`ResidualStack` to be called as a function, enabling a functional programming style.

# Arguments
- `x`: The input data to be processed by the `ResidualStack`.

# Returns
The output after applying the `ResidualStack` to the input `x`.
"""
function (stack::ResidualStack)(x::AbstractArray{Float32, 3})
    for layer in stack.layers
        x = layer(x)  # Apply each residual block
    end
    return relu.(x)  # Final ReLU activation
end