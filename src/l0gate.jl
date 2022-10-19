struct L0Gate{T, S}
    logα::T
    β::S
    dims::Union{Colon, Int}
    active::RefValue{Union{Bool,Nothing}}

    function L0Gate(logα::T, β::S, dims, active) where {T, S}
        return new{T, S}(logα, β, dims, Ref{Union{Bool,Nothing}}(active))
    end
end

L0Gate(logα = 10; β = 2/3, dims = :, active = nothing) = L0Gate(logα, β, dims, active)

Flux.@functor L0Gate

isactive(c::L0Gate) = isnothing(c.active[]) ? false : c.active[]

function (c::L0Gate)(x)
    z = if isactive(c)
        l0gate_train(x, c.logα, c.β; dims=c.dims)
    else
        l0gate_test(x, c.logα, c.β)
    end
    return z .* x
end

function Flux.testmode!(c::L0Gate, mode=true)
    c.active[] = (isnothing(mode) || mode == :auto) ? nothing : !mode
    return c
end

_shape(s, ::Colon) = size(s)
_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) in enumerate(size(s)))...)
shift(x::T, lo::Real = -0.1, hi::Real = 1.1) where {T} = T(x * (hi - lo) + lo)

function l0gate_train(x::AbstractArray{T}, logα, β; dims = :) where {T}
    u = Zygote.@ignore rand(T, _shape(x, dims))
    s = @. sigmoid((log(u) - log(1 - u) + T(logα)) / T(β))
    return @. clamp(shift(s), zero(T), one(T))
end

function l0gate_test(::AbstractArray{T}, logα, β) where {T}
    return @. clamp(shift(sigmoid(logα)), zero(T), one(T))
end
