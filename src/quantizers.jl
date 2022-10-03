abstract type Quantizer{T<:Real} end

Base.broadcastable(q::Quantizer) = Ref(q)

"""
    apply(q::Quantizer, x)

Apply given quantizer `q` to `x`. See concrete quantizers for more details: [`Sign`](@ref)
"""
function apply end

"""
    Sign(lo, hi, threshold)

deterministic binary quantizer that return `lo` when the given input is less than zero and `hi` otherwise. To apply the quantizer use the `apply` function.

# References

- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)

# Examples
```julia
julia> q = Sign()
Sign{Int64}(-1, 1, 1)

julia> apply.(q, [-2, 0.5, 2])
3-element Vector{Float64}:
 -1.0
  1.0
  1.0
```
"""
struct Sign{T} <: Quantizer{T}
    lo::T
    hi::T
    threshold::T

    Sign(lo::Real, hi::Real, threshold::Real) = Sign(promote(lo, hi, threshold)...)

    function Sign(lo::T, hi::T, threshold::T) where {T<:Real}
        lo < hi || throw(ArgumentError("`lo` must be less than `hi`"))
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(lo, hi, threshold)
    end
end

Sign(; lo::Real=-1, hi::Real=1, threshold::Real=1) = Sign(lo, hi, threshold)

apply(q::Sign, x::T) where {T<:Real} = ifelse(x < 0, T(q.lo), T(q.hi))

function ChainRulesCore.rrule(::typeof(apply), ::typeof(Sign), x)
    function apply_sign_pullback(Δy)
        return NoTangent(), NoTangent(), Δy .* (abs.(x) .<= q.threshold)
    end
    return apply.(q, x), apply_sign_pullback
end
