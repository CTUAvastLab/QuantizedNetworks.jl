"""
    Bop{T}

`Bop` is a custom binary optimizer type that implements a variant of the stochastic gradient descent `SGD` optimizer with a binary threshold for momentum updates, to decide the direction of the updates. 
If the momentum exceeds the threshold and has the same sign as the gradient, the update is set to a positive constant (one(x)); otherwise, it is set to a negative constant (-one(x)).
Allows you to control the direction of parameter updates based on the momentum history.
It is compatible with the `Flux.jl` machine learning library.

    Bop(ρ, τ, momentum)
# Fields
- ρ (rho): learning rate hyperparameter (default is 1e-4).
- τ (tau): binary threshold hyperparameter for momentum updates (default is 1e-8).
- momentum: dictionary that stores the momentum for each parameter (default is an empty dictionary).
"""
struct Bop{T} <: AbstractOptimiser
    rho::T
    threshold::T
    momentum::IdDict
end

Bop(ρ = 1f-4, τ = 1f-8, momentum = IdDict()) = Bop{typeof(ρ)}(ρ, τ, momentum)

"""
    Flux.Optimise.apply!(b::Bop, x, Δ)

A custom apply! function, which is required for optimizers in Flux.jl.
- `x`: The parameters (model weights, bias) to be updated.
- `Δ`: The gradients or updates for the parameters.
"""
function Flux.Optimise.apply!(b::Bop, x, Δ)
    ρ, τ = b.rho, b.threshold
    mt = get!(() -> zero(x), b.momentum, x)::typeof(x)

    @. mt = (1 - ρ) * mt + Δ * ρ
    @. Δ = ifelse((abs(mt) > τ) && (sign(mt) == sign(Δ)), one(x), -one(x))
    return Δ
end

"""
    CaseOptimizer

A custom optimizer that works with different optimization strategies based on conditions.
It selects the appropriate optimizer from a collection of conditions and associated optimizer objects. 
If none of the conditions match, it uses a default optimizer. 
This flexibility enables you to adapt the optimization strategy during training based on specific conditions or requirements.

# Fields
- `optimizers`: A collection of condition-to-optimizer mappings. This field stores pairs of conditions and corresponding optimizer objects.
- `default`: The default optimizer, an instance of `AdaBelief`, to use when no conditions match any of the conditions defined in the optimizers field.
"""
struct CaseOptimizer <: AbstractOptimiser
    optimizers
    default

    function CaseOptimizer(optimizers::Pair...; default = AdaBelief())
        return new(optimizers, default)
    end
end

"""
    Flux.Optimise.apply!(o::CaseOptimizer, x, Δ)

A custom apply! function for the CaseOptimizer, which is required for optimizers in Flux.jl.
Arguments:
- `x`: The parameters (model weights, bias) to be updated.
- `Δ`: The gradients or updates for the parameters.
"""
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
