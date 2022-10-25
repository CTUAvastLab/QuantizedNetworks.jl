struct Bop{T} <: AbstractOptimiser
    rho::T
    threshold::T
    momentum::IdDict
end

Bop(ρ = 1f-4, τ = 1f-8, momentum = IdDict()) = Bop{typeof(ρ)}(ρ, τ, momentum)

function Flux.Optimise.apply!(b::Bop, x, Δ)
    ρ, τ = b.rho, b.threshold
    mt = get!(() -> zero(x), b.momentum, x)::typeof(x)

    @. mt = (1 - ρ) * mt + ρ * Δ
    @. Δ = ifelse(abs(mt) > τ && sign(mt) == sign(Δ), -one(x), one(x))
    return Δ
end

function Flux.Optimise.apply!(b::Bop, x, Δ)
    ρ, τ = b.rho, b.threshold
    mt = get!(() -> zero(x), b.momentum, x)::typeof(x)

    @. mt = ρ * (Δ - mt)
    @. Δ = ifelse((-sign(x * mt - τ) * x) < 0, -one(x), one(x))
    return Δ
end

struct CaseOptimizer <: AbstractOptimiser
    optimizers
    default

    function CaseOptimizer(optimizers::Pair...; default = AdaBelief())
        return new(optimizers, default)
    end
end

function Flux.Optimise.apply!(o::CaseOptimizer, x, Δ)
    for (condition, optimizer) in o.optimizers
        if condition(x)
            return apply!(optimizer, x, Δ)
        end
    end
    return apply!(o.default, x, Δ)
end

isbinary(x::Real) = isone(abs(x))
isbinary(x) = all(isbinary, x)
