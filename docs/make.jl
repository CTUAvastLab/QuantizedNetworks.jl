using Pkg
Pkg.develop(path = "../")

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
    "estimators.md",
    "quantizers.md",
    "layers.md",
    "blocks.md",
    "l0gate.md",
    "optimizers.md"
])

examples = joinpath.("./examples/", [
    "mnist.md",
    "flower.md"
])

makedocs(
    sitename = "QuantizedNetworks",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        collapselevel=1,
        ansicolor=true
    ),
    modules = [QuantizedNetworks],
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "Api" => api,
    ],
    doctest = true,  # Disable doctests
    clean = true,

)

deploydocs(;
    repo="github.com/CTUAvastLab/QuantizedNetworks.git"
)
