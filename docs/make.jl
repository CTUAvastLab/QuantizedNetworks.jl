push!(LOAD_PATH,"../src/")

using Documenter
using QuantizedNetworks

# set metadata for docstring
DocMeta.setdocmeta!(
    QuantizedNetworks,
    :DocTestSetup,
    :(using QuantizedNetworks);
    recursive = true
)

# content
api = joinpath.("./api/", [
    "utilities.md",
    "quantizers.md",
    "layers.md",
])

makedocs(
    sitename = "QuantizedNetworks",
    format = Documenter.HTML(),
    modules = [QuantizedNetworks],
    pages = [
        "Home" => "index.md",
        "Api" => api,
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
