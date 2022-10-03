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
