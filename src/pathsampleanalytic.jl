using ChainRulesCore, Zygote, Flux, StatsBase


function psa_binary(x::Real)
    return(((tanh(x) + 1) / 2))
end 

function psa_binary(x::Matrix)
    return(_psa_binary((@. tanh(x) + 1) / 2))
end 


function _psa_binary(p::T) where {T<:Real}
    rand() > (1-p) ? one(T) : -one(T)
end

function _psa_binary(p::AbstractMatrix)
    _psa_binary.(p)
end

function ChainRulesCore.rrule(::typeof(_psa_binary), p::Real)
    o = _psa_binary(p)
    function _psa_binary_pullback(Δy)
        return NoTangent(), 2 * o * Δy
    end
    o, _psa_binary_pullback
end

function ChainRulesCore.rrule(::typeof(_psa_binary), x::AbstractMatrix)
    project_x = ProjectTo(x)
    o = _psa_binary.(x)
    function _psa_binary_pullback(Δy)
        return NoTangent(), 2 .* o .* Δy
    end
    o, _psa_binary_pullback
end


psa_ternary(x::Real) = _psa_ternary(tanh(x))
psa_ternary(x::AbstractMatrix) = _psa_ternary(tanh.(x))

 function _psa_ternary(x::Real)
    fx = floor(x)
    δ = fx + (_psa_binary(x - fx) + 1) / 2
 end

 function _psa_ternary(x::AbstractMatrix)
    fx = floor.(x)
    δ = fx + (_psa_binary(x - fx) .+ 1) / 2
 end


gradient(x -> sum(psa_binary(x)), randn(3,3))
gradient(x -> sum(psa_ternary(x)), randn(3,3))
