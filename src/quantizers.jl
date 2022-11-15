abstract type AbstractQuantizer{E<:AbstractEstimator} end

Base.broadcastable(q::AbstractQuantizer) = Ref(q)
NNlib.fast_act(q::AbstractQuantizer, ::AbstractArray) = q

(q::AbstractQuantizer)(x) = forward_pass(q, x)

"""
    forward_pass(q::Quantizer, x)

Applies quantizer `q` to value `x`.
"""
function forward_pass(q::AbstractQuantizer, x)
    T = nonmissingtype(eltype(x))
    return T.(forward_pass.(q, x))
end

"""
    pullback(q::Quantizer, x)

Returns gradient of `q` with respect to `x`.
"""
function pullback(q::AbstractQuantizer, x)
    T = nonmissingtype(eltype(x))
    return T.(pullback.(q, x))
end

function ChainRulesCore.rrule(q::AbstractQuantizer, x)
    y = q(x)
    project_y = ProjectTo(y)

    function quantizer_pullback(Δy)
        return NoTangent(), project_y(Δy .* pullback(q, x))
    end
    return y, quantizer_pullback
end

@doc raw"""
    Sign(estimator::AbstractEstimator = STE())

deterministic binary quantizer that return `-1` when the given input is less than zero or `Missing` and `1` otherwise
```math
sign(x) = \begin{cases}
    -1 & x < 0 \\
    1 & x \geq 0
\end{cases}
```
The type of the inputs is preserved with exception of `Missing` input.

# Estimators

Estimators are used to estimate non-existing gradient of the Sign function. They are used only on backward pass.

- `STE(threshold::Real = 2)`: Straight-Through Estimator approximates the sign function using the clip function
```math
clip(x) = \begin{cases}
    -1 & x < \text{threshold} \\
    1 & x > \text{threshold} \\
    x & \text{otherwise}
\end{cases}
```
with the gradient is defined as a clipped identity
```math
\frac{\partial clip}{\partial x} = \begin{cases}
    1 & \left|x\right| \leq \text{threshold} \\
    0 & \left|x\right| > \text{threshold}
\end{cases}
```
- `PolynomialSTE()`: Polynomial estimater approximates the sign function using the piecewise polynomial function
```math
poly(x) = \begin{cases}
    -1 & x < -1 \\
    2x + x^2 & -1 \leq x < 0 \\
    2x - x^2 & 0 \leq x < 1 \\
    1 & \text{otherwise}
\end{cases}
```
with the gradient is defined as
```math
\frac{\partial poly}{\partial x} = \begin{cases}
    2 + 2x & -1 \leq x < 0 \\
    2 - 2x & 0 \leq x < 1 \\
    0 & \text{otherwise}
\end{cases}
```
- `SwishSTE(β=5)`: SignSwish estimator approximates the sign function using the boundes swish function
```math
sswish_{\beta}(x) = 2\sigma(\beta x) \left(1 + \beta x (1 - \sigma(\beta x))\right)
```
where $\sigma(x)$ is the sigmoid function and $\beta > 0$ controls how fast the function asymptotes to `−1` and `+1`. The gradient is defined as
```math
\frac{\partial sswish_{\beta}}{\partial x} =
\frac{\beta\left( 2-\beta x \tanh \left(\frac{\beta x}{2}\right) \right)}{1+\cosh (\beta x)}
```

# References

- [`Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1`](https://arxiv.org/abs/1602.02830)
- [`Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm`](https://arxiv.org/abs/1808.00278)
- [`Regularized Binary Network Training`](https://arxiv.org/abs/1812.11800)

# Examples

```jldoctest
julia> using QuantizedNetworks: pullback

julia> x = [-2.0, -0.5, 0.0, 0.5, 1.0, missing];

julia> q = Sign()
Sign(STE(2))

julia> q(x)
6-element Vector{Float64}:
 -1.0
 -1.0
  1.0
  1.0
  1.0
 -1.0

julia> pullback(q, x)
6-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0

julia> pullback(Sign(PolynomialSTE()), x)
6-element Vector{Float64}:
 0.0
 1.0
 2.0
 1.0
 0.0
 0.0
```
"""
struct Sign{E<:AbstractEstimator} <: AbstractQuantizer{E}
    estimator::E

    function Sign(estimator::E=STE()) where {E}
        return new{E}(estimator)
    end
end

Base.show(io::IO, q::Sign) = print(io, "Sign($(q.estimator))")

forward_pass(::Sign, x::Missing) = -1
forward_pass(::Sign, x::Real) = ifelse(x < 0, -one(x), one(x))

pullback(q::Sign, x::Missing) = 0
function pullback(q::Sign{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end

function pullback(::Sign{<:PolynomialSTE}, x::T)::T where {T<:Real}
    return abs(2 - 2abs(x)) * (abs(x) <= 1)
end

function pullback(q::Sign{<:SwishSTE}, x::T)::T where {T<:Real}
    β = q.estimator.β
    return (β * (2 - β * x * tanh((β * x) / 2))) / (1 + cosh(β * x))
end

function ChainRulesCore.rrule(q::Sign{<:StochasticSTE}, x)
    T = nonmissingtype(eltype(x))
    y = q(x .+ (2 .* rand(T, size(x)) .- 1))

    return y, Δy -> (NoTangent(), Δy)
end

@doc raw"""
    Heaviside(estimator::AbstractEstimator = STE())

deterministic binary quantizer that return `0` when the given input is less than zero or `Missing` and `1` otherwise
```math
heaviside(x) = \begin{cases}
    0 & x \leq 0 \\
    1 & x > 0
\end{cases}
```
The type of the inputs is preserved with exception of `Missing` input.

# Estimators

Estimators are used to estimate non-existing gradient of the heaviside function. They are used only on backward pass.

- `STE(threshold::Real = 2)`: Straight-Through Estimator approximates the heaviside function using the clip function
```math
clip(x) = \begin{cases}
    0 & x < \text{threshold} \\
    1 & x > \text{threshold} \\
    x & \text{otherwise}
\end{cases}
```
with the gradient is defined as a clipped identity
```math
\frac{\partial clip}{\partial x} = \begin{cases}
    1 & \left|x\right| \leq \text{threshold} \\\
    0 & \left|x\right| > \text{threshold}
\end{cases}
```

# Examples

```jldoctest
julia> using QuantizedNetworks: pullback

julia> x = [-2.0, -0.5, 0.0, 0.5, 1.0, missing];

julia> q = Heaviside()
Heaviside(STE(2))

julia> q(x)
6-element Vector{Float64}:
 0.0
 0.0
 0.0
 1.0
 1.0
 0.0

julia> pullback(q, x)
6-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0
```
"""
struct Heaviside{E<:AbstractEstimator} <: AbstractQuantizer{E}
    estimator::E

    function Heaviside(estimator::E=STE()) where {E}
        return new{E}(estimator)
    end
end

Base.show(io::IO, q::Heaviside) = print(io, "Heaviside($(q.estimator))")
forward_pass(::Heaviside, x::Missing) = 0
forward_pass(::Heaviside, x::Real) = ifelse(x <= 0, zero(x), one(x))

pullback(q::Heaviside, x::Missing) = 0
function pullback(q::Heaviside{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end

@doc raw"""
    Ternary(Δ::T=0.05, estimator::AbstractEstimator = STE())

deterministic ternary quantizer that return `-1` when the given input is less than `Δ`, `1` whe the input in greater than Δ, and `0` otherwise. For `Missing` input, the output is `0`.
```math
ternary(x) = \begin{cases}
    -1 & x < -\Delta \\
    1 & x > \Delta \\
    0 & \text{otherwise}
\end{cases}
```
The type of the inputs is preserved with exception of `Missing` input.

# Estimators

Estimators are used to estimate non-existing gradient of the ternary function. They are used only on backward pass.

- `STE(threshold::Real = 2)`: Straight-Through Estimator approximates the ternary function using the clip function
```math
clip(x) = \begin{cases}
    -1 & x < \text{threshold} \\
    1 & x > \text{threshold} \\
    x & \text{otherwise}
\end{cases}
```
with the gradient is defined as a clipped identity
```math
\frac{\partial clip}{\partial x} = \begin{cases}
    1 & \left|x\right| \leq \text{threshold} \\\
    -1 & \left|x\right| > \text{threshold}
\end{cases}
```

# Examples

```jldoctest
julia> using QuantizedNetworks: pullback

julia> x = [-2.0, -0.5, 0.0, 0.5, 1.0, missing];

julia> q = Ternary()
Ternary(0.05, STE(2))

julia> q(x)
6-element Vector{Float64}:
 -1.0
 -1.0
  0.0
  1.0
  1.0
  0.0

julia> pullback(q, x)
6-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0
```
"""
struct Ternary{E<:AbstractEstimator,T} <: AbstractQuantizer{E}
    Δ::T
    estimator::E

    function Ternary(Δ::T=0.05, estimator::E=STE()) where {T<:Real,E}
        Δ > 0 || throw(ArgumentError("`Δ` must be positive"))
        return new{E,T}(Δ, estimator)
    end
end

Base.show(io::IO, q::Ternary) = print(io, "Ternary$((q.Δ, q.estimator))")

forward_pass(::Ternary, x::Missing) = 0
function forward_pass(q::Ternary, x::Real)
    return if x < -q.Δ
        -one(x)
    elseif x > q.Δ
        one(x)
    else
        zero(x)
    end
end

pullback(q::Ternary, x::Missing) = 0
function pullback(q::Ternary{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end
