# here we include some auxiliary functions for the VQVAE for gradient flow and autodiff

# first, some CUDA kernels...
"""
    gpu_argmin_rows_kernel(
        A::CuDeviceMatrix{T},
        rowidx::CuDeviceVector{Int32},
        m::Int32,
        n::Int32
    ) where {T}

CUDA kernel that computes the index of the minimum value in each row of a matrix `A` and stores the result in `rowidx`.

# Arguments
- `A::CuDeviceMatrix{T}`: Input matrix of type `T` stored on the GPU.
- `rowidx::CuDeviceVector{Int32}`: Output vector (on the GPU) where the index of the minimum value for each row will be stored.
- `m::Int32`: Number of rows in `A`.
- `n::Int32`: Number of columns in `A`.
"""
function gpu_argmin_rows_kernel(
    A::CuDeviceMatrix{T},
    rowidx::CuDeviceVector{Int32},    
    m::Int32,
    n::Int32
) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i ≤ m
            # Initialize with the first column in row i
            minval = A[i, 1]
            minj   = Int32(1)

            # Scan across columns 2..n in this row
            @inbounds for j in 2:n
                v = A[i, j]
                if v < minval
                    minval = v
                    minj   = Int32(j)
                end
            end
            rowidx[i] = minj
        end
    return
end

"""
    gpu_argmin_rows(A::CuArray{T,2}) where {T}

Finds the index of the minimum value in each row of the given 2D CuArray `A` using a custom CUDA kernel.
Convenience wrapper for the kernel `gpu_argmin_rows_kernel`.

# Arguments
- `A::CuArray{T,2}`: A 2D CUDA array (matrix) of element type `T`.

# Returns
- `rowidx::CuArray{Int32,1}`: A 1D CUDA array of length equal to the number of rows in `A`, where each element contains the column index (as `Int32`) of the minimum value in the corresponding row.
"""
function gpu_argmin_rows(A::CuArray{T,2}) where {T}
    m, n = size(A)
    rowidx = CUDA.zeros(Int32, m)
    threads = 256
    blocks = cld(m, threads)
    @cuda threads=threads blocks=blocks gpu_argmin_rows_kernel(
        A,       
        rowidx,  
        Int32(m),
        Int32(n)
    )
    return rowidx
end

Zygote.@adjoint function gpu_argmin_rows(A::CuArray{T,2}) where {T} 
    rowidx = gpu_argmin_rows(A)
    function gpu_argmin_rows_pullback(ȳ)
        return(nothing, nothing)
    end
    rowidx, gpu_argmin_rows_pullback
end

"""
    gpu_argmin_cols_kernel(
        A::CuDeviceMatrix{T},
        colidx::CuDeviceVector{Int32},
        m::Int32,
        n::Int32
    ) where {T}

CUDA kernel that computes the index of the minimum value in each column of a matrix `A` and stores the result in `colidx`.

# Arguments
- `A::CuDeviceMatrix{T}`: Input matrix stored on the GPU.
- `colidx::CuDeviceVector{Int32}`: Output vector (on the GPU) to store the row indices of the minimum values for each column.
- `m::Int32`: Number of rows in `A`.
- `n::Int32`: Number of columns in `A`.
"""
function gpu_argmin_cols_kernel(
    A::CuDeviceMatrix{T},
    colidx::CuDeviceVector{Int32},
    m::Int32,
    n::Int32
) where {T}
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if j ≤ n
            # Initialize with the first row in column j
            minval = A[1, j]
            mini   = Int32(1)

            # Scan rows 2..m in this column
            @inbounds for i in 2:m
                v = A[i, j]
                if v < minval
                    minval = v
                    mini   = Int32(i)
                end
            end
            colidx[j] = mini
        end
    return
end

"""
    gpu_argmin_cols(A::CuArray{T,2}) where {T}

Finds the index of the minimum element in each column of the given 2D CuArray `A` using a custom CUDA kernel.
Convenience wrapper for the kernel `gpu_argmin_cols_kernel`.

# Arguments
- `A::CuArray{T,2}`: A 2D CUDA array of element type `T`.

# Returns
- `colidx::CuArray{Int32,1}`: A 1D CUDA array of length equal to the number of columns in `A`, where each element contains the row index (as `Int32`) of the minimum value in the corresponding column.
"""
function gpu_argmin_cols(A::CuArray{T,2}) where {T}
    m, n = size(A)
    colidx = CUDA.zeros(Int32, n)
    threads = 256
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks gpu_argmin_cols_kernel(
        A,       
        colidx,  
        Int32(m),
        Int32(n)
    )
    return colidx
end

Zygote.@adjoint function gpu_argmin_cols(A::CuArray{T,2}) where {T} 
    colidx = gpu_argmin_cols(A)
    function gpu_argmin_cols_pullback(ȳ)
        return(nothing, nothing)
    end
    colidx, gpu_argmin_cols_pullback
end


# now, some gradient flow definitions...
"""
    stopgrad(x)

Returns `x` without modifying it. 
"""
stopgrad(x) = x

ChainRulesCore.@non_differentiable stopgrad(x)

"""
    straight_through(quantized, inputs)

Applies the straight-through estimator for quantization in neural networks.
The goal is to work as identity in the forward pass, while allowing gradients to flow through as if it were the original input tensor skipping parts of the computational graph.

# Arguments
- `quantized`: The quantized tensor (typically the output of a quantization operation).
- `inputs`: The original input tensor before quantization.

# Returns
- A tensor with the same values as `quantized` in the forward pass, but gradients as if it were `inputs` during backpropagation.
"""
function straight_through(quantized, inputs)
    quantized = inputs .+ quantized .- inputs
    return quantized
end

Zygote.@adjoint function straight_through(quantized, inputs)
    quantized = straight_through(quantized, inputs)
    function straight_through_pullback(ȳ)
        return (ȳ,ȳ,)
    end
    return quantized, straight_through_pullback
end

"""
compute_padding(kernel_size::Int, dilation::Int, stride::Int, lin::Int, lout::Int)

Compute the required padding for a strided 1D convolution given kernel size, dilation, stride,
input length and output length.

Arguments
- kernel_size::Int: convolution kernel size
- dilation::Int: dilation factor
- stride::Int: stride length
- lin::Int: input length
- lout::Int: output length

Returns
- Int: computed padding size
"""
function compute_padding(kernel_size::Int, dilation::Int, stride::Int, lin::Int, lout::Int)
    @assert stride * lout == lin "stride and lengths need to match"
    return Int(ceil((stride * (lout - 1) - lin + dilation * (kernel_size -1) + 1) / 2))
end

function compute_padding(kernel_size::Int, dilation::Int)
    return Int(ceil(dilation * (kernel_size -1)) / 2)
end
"""
compute_padding(kernel_size::Int, dilation::Int)

Compute symmetric padding size for a convolution given kernel size and dilation.

# Arguments
- kernel_size::Int: Size of the convolution kernel.
- dilation::Int: Dilation factor applied to the kernel.

# Returns
- Int: Number of padding elements to apply on each side.
"""

"""
Compute symmetric padding for a transposed strided convolution.

Arguments
- kernel_size::Int — odd kernel size
- dilation::Int — dilation factor
- stride::Int — stride (requires lout == stride * lin)
- lin::Int — input length
- lout::Int — output length

Returns
- Int — padding size per side
"""
function compute_paddingT(kernel_size::Int, dilation::Int, stride::Int, lin::Int, lout::Int)
    @assert  lout == stride *lin "stride and lengths need to match"
    @assert kernel_size % 2 == 1 "kernel size needs to be odd"
    #return Int(ceil((stride * lin - 1 - lout + dilation * (kernel_size - 1) + 1) / 2))  
    return Int((dilation * (kernel_size - 1)) ÷ 2)
end

"""
Compute symmetric padding for a transposed 1D convolution given kernel size and dilation.

Arguments
- kernel_size::Int: size of the convolution kernel
- dilation::Int: dilation factor

Returns
- Int: padding size (ceil(dilation * (kernel_size - 1) / 2))
"""
function compute_paddingT(kernel_size::Int, dilation::Int)
    return Int(ceil(dilation * (kernel_size -1)) / 2)
end