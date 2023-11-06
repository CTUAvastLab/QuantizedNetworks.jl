"""
    struct L0Gate{T, S}

Represents an L0Gate which applies L0 regularization during neural network training.

    L0Gate(logα = 10; β = 2/3, dims = :, active = nothing)

# Fields
- `logα::T`: Controls the strength of L0 regularization. (defaults to `10`)
- `β::S`: Controls "temperature" of a sigmoid function. (defaults to `2/3`)
- `dims::Union{Colon, Int}`: Specifies the dimensions to which L0 regularization is applied. (defaults to `:`)
- `active::RefValue{Union{Bool, Nothing}}`: Indicates whether the L0Gate is active (regularization is applied). (defaults to `nothing`)
"""
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
"""
    isactive(c::L0Gate)

Checks if the L0Gate is active (applies regularization).
"""
isactive(c::L0Gate) = isnothing(c.active[]) ? false : c.active[]

"""
    (c::L0Gate)(x)

Implements a callable behavior for L0Gate objects.
Depending on whether the L0Gate is active or not, it calls different functions (l0gate_train or l0gate_test) 
to apply L0 regularization to the input x.
"""
function (c::L0Gate)(x)
    z = if isactive(c)
        l0gate_train(x, c.logα, c.β; dims=c.dims)
    else
        l0gate_test(x, c.logα, c.β)
    end
    return z .* x
end

"""
    Flux.testmode!(c::L0Gate, mode=true)

Sets the testing mode for the L0Gate object.
If mode is true, it sets the active field to nothing, effectively turning off L0 regularization during testing.
If mode is false, it sets the active field to true, enabling L0 regularization during testing.
If mode is :auto, it toggles the active field.
"""
function Flux.testmode!(c::L0Gate, mode=true)
    c.active[] = (isnothing(mode) || mode == :auto) ? nothing : !mode
    return c
end

"""
    _shape(s, ::Colon)

Computes the size of an array when applying L0 regularization to all dimensions
"""
_shape(s, ::Colon) = size(s)

"""
    _shape(s, dims)

Computes the size of an array when applying L0 regularization to specified dimensions.
"""
_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) in enumerate(size(s)))...)

"""
    shift(x::T, lo::Real = -0.1, hi::Real = 1.1) where {T}

Shifts the input x to a specified range [lo, hi].
"""
shift(x::T, lo::Real = -0.1, hi::Real = 1.1) where {T} = T(x * (hi - lo) + lo)

"""
    l0gate_train(x::AbstractArray{T}, logα, β; dims = :) where {T}

Applies L0 regularization during training.
"""
function l0gate_train(x::AbstractArray{T}, logα, β; dims = :) where {T}
    u = Zygote.@ignore rand(T, _shape(x, dims))
    s = @. sigmoid((log(u) - log(1 - u) + T(logα)) / T(β))
    return @. clamp(shift(s), zero(T), one(T))
end

"""
    l0gate_test(::AbstractArray{T}, logα, β) where {T}
    
Applies L0 regularization during testing.
"""
function l0gate_test(::AbstractArray{T}, logα, β) where {T}
    return @. clamp(shift(sigmoid(logα)), zero(T), one(T))
end
