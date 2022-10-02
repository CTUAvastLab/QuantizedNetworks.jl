struct BinDense{M<:ClippedMatrix,F}
    weight::M
    quantizer::F
end

function BinDense(
    (in, out)::Pair{<:Integer,<:Integer};
    quantizer=binarize,
    init = glorot_uniform
)

    return BinDense(ClippedArray(init(out, in)), quantizer)
end

@functor BinDense

function (l::BinDense)(x::AbstractVecOrMat)
    Wbin = l.quantizer.(l.weight)
    return l.quantizer.(Wbin * x)
end

function (l::BinDense)(x::AbstractArray)
    reshape(l(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::BinDense)
    print(io, "BinDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    print(io, ")")
end
