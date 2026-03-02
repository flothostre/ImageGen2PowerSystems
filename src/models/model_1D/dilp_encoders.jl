# and now to the encoder definitons
struct DILPEncoder4 <: Encoder
    dirs1::DilatedResidualStack
    conv1::LPConv
    dirs2::DilatedResidualStack
    conv2::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
DILPEncoder4(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an encoder composed of cascaded low-pass convolutions and dilated residual stacks.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels/features.
- num_residual_layers::Int: Number of residual layers in the final residual stack.
- num_residual_hiddens::Int: Hidden channels inside residual layers.
- dires_contribution_factor::Float64: Scaling factor for dilated residual contributions.
- kernel_size_ds::Int: Kernel size for downsampling (LPConv) layers.
- kernel_size_res::Int: Kernel size for residual convolutions.
- taps::Int=63: Number of taps for the low-pass filters.
- beta::Float64=8.0: Beta parameter for the low-pass filter design.
- cutoff_factor::Float64=0.9: Cutoff frequency factor for the low-pass filters.

Returns
- DILPEncoder4: An instance of the encoder composed of the configured components.
"""
function DILPEncoder4(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    dirs1 = DilatedResidualStack(
        in_channels,
        4,
        num_residual_hiddens,
        [1, 1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
    )
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
    dirs2 = DilatedResidualStack(
        num_hiddens ÷ 2,
        3,
        num_residual_hiddens,
        [1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
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
    return DILPEncoder4(dirs1, conv1, dirs2, conv2, conv3, residual_stack)
end

"""
Call operator for DILPEncoder4.

Arguments
- lpencoder::DILPEncoder4: encoder instance.
- x::AbstractArray{Float32,3}: 3‑D Float32 input tensor.

Returns
- AbstractArray{Float32,3}: encoded feature map after convolutional layers and residual stack.
"""
function (lpencoder::DILPEncoder4)(x::AbstractArray{Float32, 3})
    x = lpadencoder.dirs1(x)
    x = lpencoder.conv1(x)
    x = relu.(x)

    x = lpencoder.dirs2(x)
    x = lpencoder.conv2(x)
    x = relu.(x)

    x = lpencoder.conv3(x)
    return lpencoder.residual_stack(x)
end


struct DILPEncoder8 <: Encoder
    dirs1::DilatedResidualStack
    conv1::LPConv
    dirs2::DilatedResidualStack
    conv2::LPConv
    dirs3::DilatedResidualStack
    conva::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
DILPEncoder8(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an encoder composed of cascaded low-pass convolutions and dilated residual stacks.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels in intermediate feature maps.
- num_residual_layers::Int: Number of layers in the final ResidualStack.
- num_residual_hiddens::Int: Hidden channels inside residual blocks.
- dires_contribution_factor::Float64: Scaling factor for dilated residual contributions.
- kernel_size_ds::Int: Kernel size for downsampling LPConv layers.
- kernel_size_res::Int: Kernel size for residual convolutions.
- taps::Int=63: Number of taps for LPConv filters.
- beta::Float64=8.0: Beta parameter for LPConv filters.
- cutoff_factor::Float64=0.9: Cutoff factor for LPConv filters.

Returns
- DILPEncoder8: Constructed encoder instance composed of the configured submodules.
"""
function DILPEncoder8(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    dirs1 = DilatedResidualStack(
        in_channels,
        4,
        num_residual_hiddens,
        [1, 1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
    )
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
    dirs2 = DilatedResidualStack(
        num_hiddens ÷ 2,
        4,
        num_residual_hiddens,
        [1, 2, 4, 8],
        kernel_size_res,
        dires_contribution_factor
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
    dirs3 = DilatedResidualStack(
        num_hiddens,
        3,
        num_residual_hiddens,
        [1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
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
    return DILPEncoder8(dirs1, conv1, dirs2, conv2, dirs3, conva, conv3, residual_stack)
end

"""
Call method for DILPEncoder8.

Arguments
- lpencoder::DILPEncoder8: the encoder instance.
- x::AbstractArray{Float32,3}: input 3D Float32 tensor.

Returns
- AbstractArray{Float32,3}: encoded output after directional layers, convolutions, activations and residual stack.
"""
function (lpencoder::DILPEncoder8)(x::AbstractArray{Float32, 3})
    x = lpencoder.dirs1(x)
    x = lpencoder.conv1(x)
    x = relu.(x)

    x = lpencoder.dirs2(x)
    x = lpencoder.conv2(x)
    x = relu.(x)

    x = lpencoder.dirs3(x)
    x = lpencoder.conva(x)
    x = relu.(x)

    x = lpencoder.conv3(x)
    return lpencoder.residual_stack(x)
end


struct DILPEncoder16 <: Encoder
    dirs1::DilatedResidualStack
    conv1::LPConv
    dirs2::DilatedResidualStack
    conv2::LPConv
    dirs3::DilatedResidualStack
    conva::LPConv
    dirs4::DilatedResidualStack
    convb::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
DILPEncoder16(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
              dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an encoder composed of cascaded low-pass convolutions and dilated residual stacks.

Arguments
- in_channels::Int: number of input channels
- num_hiddens::Int: base number of hidden channels used throughout the encoder
- num_residual_layers::Int: number of residual layers in the final residual stack
- num_residual_hiddens::Int: hidden channels inside each residual layer
- dires_contribution_factor::Float64: scaling factor for dilated residual contributions
- kernel_size_ds::Int: kernel size for downsampling LPConv layers
- kernel_size_res::Int: kernel size for residual convolutions
- taps::Int=63: (keyword) number of filter taps for LPConv
- beta::Float64=8.0: (keyword) beta parameter for LPConv
- cutoff_factor::Float64=0.9: (keyword) cutoff frequency factor for LPConv

Returns
- DILPEncoder16: an instance of the configured encoder
"""
function DILPEncoder16(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    dirs1 = DilatedResidualStack(
        in_channels,
        4,
        num_residual_hiddens,
        [1, 1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
    )
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
    dirs2 = DilatedResidualStack(
        num_hiddens ÷ 2,
        4,
        num_residual_hiddens,
        [1, 2, 4, 8],
        kernel_size_res,
        dires_contribution_factor
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
    dirs3 = DilatedResidualStack(
        num_hiddens,
        4,
        num_residual_hiddens,
        [1, 2, 4, 8],
        kernel_size_res,
        dires_contribution_factor
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
    dirs4 = DilatedResidualStack(
        num_hiddens,
        3,
        num_residual_hiddens,
        [1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
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
    return DILPEncoder16(dirs1, conv1, dirs2, conv2, dirs3, conva, dirs4, convb, conv3, residual_stack)
end

"""
Call operator for DILPEncoder16.

Arguments
- x::AbstractArray{Float32,3}: input 3D Float32 tensor.

Returns
- AbstractArray{Float32,3}: encoded output produced by the encoder's convolutional and residual layers.
"""
function (lpencoder16::DILPEncoder16)(x::AbstractArray{Float32, 3})
    x = lpencoder16.dirs1(x)
    x = lpencoder16.conv1(x)
    x = relu.(x)

    x = lpencoder16.dirs2(x)
    x = lpencoder16.conv2(x)
    x = relu.(x)

    x = lpencoder16.dirs3(x)
    x = lpencoder16.conva(x)
    x = relu.(x)

    x = lpencoder16.dirs4(x)
    x = lpencoder16.convb(x)
    x = relu.(x)

    x = lpencoder16.conv3(x)
    return lpencoder16.residual_stack(x)
end


struct DILPEncoder32 <: Encoder
    dirs1::DilatedResidualStack
    conv1::LPConv
    dirs2::DilatedResidualStack
    conv2::LPConv
    dirs3::DilatedResidualStack
    conva::LPConv
    dirs4::DilatedResidualStack
    convb::LPConv
    dirs5::DilatedResidualStack
    convc::LPConv
    conv3::Conv
    residual_stack::ResidualStack
end

"""
DILPEncoder32(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
              dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63,
              beta::Float64=8.0, cutoff_factor::Float64=0.9)

Construct an encoder composed of cascaded low-pass convolutions and dilated residual stacks.

Arguments
- in_channels::Int: Number of input channels.
- num_hiddens::Int: Number of hidden channels/features.
- num_residual_layers::Int: Number of layers in the final residual stack.
- num_residual_hiddens::Int: Hidden size inside each residual block.
- dires_contribution_factor::Float64: Scaling factor for dilated residual contributions.
- kernel_size_ds::Int: Kernel size for downsampling (LPConv) layers.
- kernel_size_res::Int: Kernel size for residual convolutions.
- taps::Int=63: (keyword) Number of taps for LPConv filters.
- beta::Float64=8.0: (keyword) Beta parameter for LPConv.
- cutoff_factor::Float64=0.9: (keyword) Cutoff factor for LPConv.

Returns
- DILPEncoder32: An instance representing the assembled encoder model.
"""
function DILPEncoder32(in_channels::Int, num_hiddens::Int, num_residual_layers::Int, num_residual_hiddens::Int,
    dires_contribution_factor::Float64, kernel_size_ds::Int, kernel_size_res::Int; taps::Int=63, beta::Float64=8.0, cutoff_factor::Float64=0.9)
    dirs1 = DilatedResidualStack(
        in_channels,
        4,
        num_residual_hiddens,
        [1, 1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
    )
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
    dirs2 = DilatedResidualStack(
        num_hiddens ÷ 2,
        4,
        num_residual_hiddens,
        [1, 2, 4, 8],
        kernel_size_res,
        dires_contribution_factor
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
    dirs3 = DilatedResidualStack(
        num_hiddens,
        4,
        num_residual_hiddens,
        [1, 2, 4, 8],
        kernel_size_res,
        dires_contribution_factor
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
    dirs4 = DilatedResidualStack(
        num_hiddens,
        4,
        num_residual_hiddens,
        [1, 2, 4, 8],
        kernel_size_res,
        dires_contribution_factor
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
    dirs5 = DilatedResidualStack(
        num_hiddens,
        3,
        num_residual_hiddens,
        [1, 2, 4],
        kernel_size_res,
        dires_contribution_factor
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
    return DILPEncoder32(dirs1, conv1, dirs2, conv2, dirs3, conva, dirs4, convb, dirs5,convc, conv3, residual_stack)
end

"""
Call operator of DILPEncoder32.

Arguments
- x::AbstractArray{Float32, 3}: 3‑dimensional Float32 input tensor to be encoded.

Returns
- AbstractArray{Float32, 3}: encoded Float32 tensor after the convolutional layers and residual stack.
"""
function (lpencoder32::DILPEncoder32)(x::AbstractArray{Float32, 3})
    x = lpencoder32.dirs1(x)
    x = lpencoder32.conv1(x)
    x = relu.(x)

    x = lpencoder32.dirs2(x)
    x = lpencoder32.conv2(x)
    x = relu.(x)

    x = lpencoder32.dirs3(x)
    x = lpencoder32.conva(x)
    x = relu.(x)

    x = lpencoder32.dirs4(x)
    x = lpencoder32.convb(x)
    x = relu.(x)

    x = lpencoder32.dirs5(x)
    x = lpencoder32.convc(x)
    x = relu.(x)

    x = lpencoder32.conv3(x)
    return lpencoder32.residual_stack(x)
    
end