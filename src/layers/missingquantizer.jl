struct MissingQuantizer{Q}
    quantizer::Q
end

MissingQuantizer(; quantizer = Sign()) = MissingQuantizer(quantizer)

Flux.@functor MissingQuantizer

function (q::MissingQuantizer)(x)
    T = truetype(x)
    return q.output_quantizer(ifelse.(ismissing.(x), -one(T), one(T)))
end
