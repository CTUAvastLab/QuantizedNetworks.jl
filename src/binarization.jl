# deterministic binarization
binarize(x::Real) = x >= 0 ? one(x) : -one(x)

@scalar_rule(binarize(x), one(x) * (abs(x) <= 1))

function ChainRulesCore.rrule(
    ::typeof(Broadcast.broadcasted),
    ::typeof(binarize),
    x::Union{Numeric,Broadcast.Broadcasted}
)

    function broadcasted_binarize_pullback(ȳ)
        return NoTangent(), NoTangent(), ȳ .* (abs.(x) .<= 1)
    end
    return binarize.(x), broadcasted_binarize_pullback
end

# stochastic binarization
clip(x::Real, xmin::Real, xmax::Real) = max(xmin, min(xmax, x))

function binarize_stochastic(x::Real)
    p = clip((x + 1) / 2, zero(x), one(x)) # hard sigmoid
    return rand(Bernoulli(p)) ? one(x) : -one(x)
end

@scalar_rule(binarize_stochastic(x), one(x) * (abs(x) <= 1))

function ChainRulesCore.rrule(
    ::typeof(Broadcast.broadcasted),
    ::typeof(binarize_stochastic),
    x::Union{Numeric,Broadcast.Broadcasted}
)

    function broadcasted_binarize_stochastic_pullback(ȳ)
        return NoTangent(), NoTangent(), ȳ .* (abs.(x) .<= 1)
    end
    return binarize_stochastic.(x), broadcasted_binarize_stochastic_pullback
end
