struct Bop{T} <: AbstractRule
    rho::T
    threshold::T
end

Bop(ρ = 1e-4, τ = 1f-8) = Bop{typeof(ρ)}(ρ, τ)
Optimisers.init(o::Bop, x::AbstractArray) = (zero(x), )

function Optimisers.apply!(b::Bop, state, x, dx)
    ρ, τ = b.rho, b.threshold
    mt, = state
    @.. mt = (1 - ρ) * mt + ρ * dx

    return (mt, ), @lazy ifelse(abs(mt) > τ && sign(mt) == sign(dx), -1, 1)
end
