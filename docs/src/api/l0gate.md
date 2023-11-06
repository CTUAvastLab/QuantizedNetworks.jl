# L0 gate

```@docs
L0Gate
```

### Functions

```@docs
QuantizedNetworks.isactive(c::L0Gate)
QuantizedNetworks.Flux.testmode!(c::L0Gate, mode=true)
```

### Helper functions
```@docs
QuantizedNetworks._shape(s, ::Colon)
QuantizedNetworks._shape(s, dims)
QuantizedNetworks.shift(x::T, lo::Real = -0.1, hi::Real = 1.1) where {T}
QuantizedNetworks.l0gate_train(x::AbstractArray{T}, logα, β; dims = :) where {T}
QuantizedNetworks.l0gate_test(::AbstractArray{T}, logα, β) where {T}
```