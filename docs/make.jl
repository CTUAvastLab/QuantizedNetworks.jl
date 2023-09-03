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
    "estimators.md",
    "quantizers.md",
    "layers.md",
    "blocks.md"
])

makedocs(
    sitename = "QuantizedNetworks",
    format = Documenter.HTML(prettyurls=false),
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
