# Layers

## QuantDense
```@docs
QuantDense
```
### Logic
```@docs
QuantizedNetworks.nn2logic(layer::QuantDense)
```

### Examples

```jldoctest
julia> using Random, QuantizedNetworks; Random.seed!(3);

julia> x = rand(Float32, 1, 2);

julia> qd = QuantDense(1 => 2)
QuantDense(1 => 2, identity; weight_lims=(-1.0f0, 1.0f0), bias=false, Ternary(0.05, STE(2)), Sign(STE(2)))

julia> qd(x)
2×2 Matrix{Float32}:
 -1.0  -1.0
  1.0   1.0

julia> d = QuantizedNetworks.nn2logic(qd)
Dense(1 => 2, Sign(STE(2)))  # 4 parameters

julia> d([3])
2-element Vector{Float32}:
 -1.0
  1.0
```

## FeatureQuantizer
```@docs
FeatureQuantizer
```

### Forward pass
```@docs
QuantizedNetworks._forward_pass(w, b, x)
```

### Backpropagation
```@docs
QuantizedNetworks.ChainRulesCore.rrule(::typeof(QuantizedNetworks._forward_pass), w, b, x)
```

### Examples
```jldoctest
julia> using Random, QuantizedNetworks; Random.seed!(3);

julia> x = Float32.([1 2 ;3 4]);

julia> fq = FeatureQuantizer(2,2);

julia> fq(x)
4×2 Matrix{Float32}:
  1.0   1.0
  1.0  -1.0
 -1.0  -1.0
  1.0   1.0
```