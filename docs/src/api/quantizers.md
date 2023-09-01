# Quantizers 
```@docs
QuantizedNetworks.AbstractQuantizer
QuantizedNetworks.forward_pass(q::AbstractQuantizer, x)
QuantizedNetworks.pullback(q::AbstractQuantizer, x)
```
## Binary
```@docs
QuantizedNetworks.Sign
QuantizedNetworks.Heaviside
```
## Ternary
```@docs
QuantizedNetworks.Ternary
```

## References

- [`Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1`](https://arxiv.org/abs/1602.02830)
- [`Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm`](https://arxiv.org/abs/1808.00278)
- [`Regularized Binary Network Training`](https://arxiv.org/abs/1812.11800)