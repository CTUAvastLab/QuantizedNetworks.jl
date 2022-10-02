struct BinDense{M<:ClippedMatrix,B,F}
    weight::M
    batchnorm::B
    bin::F
end

function BinDense((in, out)::Pair{<:Integer,<:Integer}; bin=binarize, init = glorot_uniform)
    W = ClippedArray(2 .* init(out, in) .- 1)
    B = BatchNorm(out)
    return BinDense(W, B, bin)
end

@functor BinDense

function (l::BinDense)(x::AbstractVecOrMat)
    Wbin = l.bin.(l.weight)
    return hardtanh.(l.batchnorm(Wbin * x))
end

function (l::BinDense)(x::AbstractArray)
    reshape(l(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::BinDense)
    print(io, "BinDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.batchnorm == identity || print(io, ", ", l.batchnorm)
    print(io, ")")
end
