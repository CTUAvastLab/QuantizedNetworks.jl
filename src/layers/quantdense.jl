struct QuantDense{F, M, B, Q1, S, Q2, N}
    weight::M
    bias::B
    σ::F

    weight_quantizer::Q1
    weight_sparsifier::S
    output_quantizer::Q2
    batchnorm::N
end

function QuantDense(
    weight::AbstractArray,
    bias,
    σ = identity;
    weight_quantizer = Ternary(),
    weight_sparsifier = identity,
    output_quantizer = Sign(),
    batchnorm::Bool = true,
 )

    return QuantDense(
        weight,
        create_bias(weight, bias, size(weight,1)),
        batchnorm ? identity : σ,
        weight_quantizer,
        weight_sparsifier,
        output_quantizer,
        batchnorm ? BatchNorm(size(weight, 1), σ) : identity,
    )
end

function QuantDense(
    (in, out)::Pair{<:Integer, <:Integer},
    σ = identity;
    init = ClippedArray,
    bias = false,
    kwargs...
)
    return QuantDense(init(out, in), bias, σ; kwargs...)
end

function QuantDense(in::Integer, out::Integer, σ = identity; kwargs...)
    return QuantDense(in => out, σ; kwargs...)
end

function QuantDense(
    l::Dense;
    weight = copy(l.weight),
    bias = copy(l.bias),
    σ = l.σ,
    kwargs...
)
    return QuantDense(weight, bias, σ; kwargs...)
end

@functor QuantDense

function (l::QuantDense)(x::AbstractVecOrMat)
    σ = NNlib.fast_act(l.σ, x)  # replaces tanh => tanh_fast, etc
    wq = l.weight_quantizer(l.weight)
    wqs = l.weight_sparsifier(wq)

    return l.output_quantizer(l.batchnorm(σ.(wqs * x .+ l.bias)))
end

function (l::QuantDense)(x::AbstractArray)
    return reshape(l(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

# TODO improve printing
function Base.show(io::IO, l::QuantDense)
    print(io, "QuantDense(", size(l.weight, 2), " => ", size(l.weight, 1))

    l.σ == identity || print(io, ", ", l.σ)
    l.batchnorm == identity || print(io, ", ", l.batchnorm.λ)

    kwargs = String[]
    if isa(l.weight, ClippedArray)
        push!(kwargs, "weight_lims=$((l.weight.lo, l.weight.hi))")
    end
    if l.bias == false
        push!(kwargs, "bias=false")
    else
        if isa(l.bias, ClippedArray)
            push!(kwargs, "bias_lims=$((l.bias.lo, l.bias.hi))")
        end
    end
    push!(kwargs, "$(l.weight_quantizer)")
    push!(kwargs, "$(l.output_quantizer)")
    l.batchnorm == identity && push!(kwargs, "batchnorm=false")
    print(io, "; ", join(kwargs, ", "), ")")
end
