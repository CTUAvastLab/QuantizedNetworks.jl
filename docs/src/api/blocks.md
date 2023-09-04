# Blocks

## DenseBlock
```@docs
DenseBlock
```
### Standard Dense Layer
```@docs
QuantizedNetworks.Flux.Dense(l::DenseBlock)
```

### Examples
```jldoctest
julia> using Random, QuantizedNetworks; Random.seed!(3);

julia> db = DenseBlock(2=>2)
DenseBlock(Chain(QuantizedDense(2 => 2; bias=false, quantizer=Ternary(0.05, STE(2))), BatchNorm(2), Sign(STE(2))))

julia> x = rand(Float32, 2, 4)
2×4 Matrix{Float32}:
 0.940675  0.100403   0.789168  0.582228
 0.999979  0.0921143  0.698426  0.496285

julia> db(x)
2×4 Matrix{Float32}:
 -1.0   1.0   1.0   1.0
  1.0  -1.0  -1.0  -1.0
```


## FeatureBlock
```@docs
FeatureBlock
```

### Examples
```jldoctest
julia> using Random, QuantizedNetworks; Random.seed!(3);

julia> fb = FeatureBlock(2, 2)
FeatureBlock(Parallel(vcat, FeatureQuantizer(2 => 4; quantizer=Sign(STE(2)))))

julia> x = rand(Float32, 2, 1)
2×1 Matrix{Float32}:
 0.8521847
 0.7965402

julia> fb(x)
4×1 Matrix{Float32}:
  1.0
  1.0
  1.0
 -1.0
```