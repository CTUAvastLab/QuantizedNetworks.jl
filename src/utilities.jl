truetype(x) = eltype(x)
truetype(::AbstractArray{T}) where {T} = typeof(zero(T))
