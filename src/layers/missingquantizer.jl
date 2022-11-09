struct MissingQuantizer{Q}
    quantizer::Q
end

MissingQuantizer(; quantizer = Sign()) = MissingQuantizer(quantizer)

Flux.@functor MissingQuantizer

function (q::MissingQuantizer)(x)
    T = nonmissingtype(eltype(x))
    return q.quantizer(ifelse.(ismissing.(x), -one(T), one(T)))
end

function Base.show(io::IO, q::MissingQuantizer)
    return print(io, "MissingQuantizer(; quantizer=$(q.quantizer))")
end
