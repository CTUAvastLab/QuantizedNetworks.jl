struct QuantizedDense{F,M<:AbstractMatrix,B,Q}
    weight::M
    bias::B
    σ::F
    quantizer::Q

    function QuantizedDense(
        weight::M,
        bias=false,
        σ::F=identity,
        quantizer::Q=Ternary(),
    ) where {M<:AbstractMatrix,F,Q}

        b = create_bias(weight, bias, size(weight, 1))
        return new{F,M,typeof(b),Q}(weight, b, σ, quantizer)
    end
end

function QuantizedDense(
    (in, out)::Pair{<:Integer,<:Integer},
    σ=identity;
    init=ClippedArray,
    bias=false,
    quantizer=Ternary()
)

    return QuantizedDense(init(out, in), bias, σ, quantizer)
end

@functor QuantizedDense

function (l::QuantizedDense)(x::AbstractVecOrMat)
    σ = NNlib.fast_act(l.σ, x)  # replaces tanh => tanh_fast, etc
    weight = l.quantizer(l.weight)
    return σ.(weight * x .+ l.bias)
end

function (l::QuantizedDense)(x::AbstractArray)
    return reshape(l(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::QuantizedDense)
    print(io, "QuantizedDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    kwargs = String[]
    l.bias == false && push!(kwargs, "bias=false")
    l.quantizer == identity || push!(kwargs, "quantizer=$(l.quantizer)")
    if isempty(kwargs)
        print(io, ")")
    else
        print(io, "; ", join(kwargs, ", "), ")")
    end
end

# Conversion to standard Dense layer
function Flux.Dense(l::QuantizedDense)
    return Dense(l.quantizer(copy(l.weight)), copy(l.bias), l.σ)
end
