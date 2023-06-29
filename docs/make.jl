using QuantumUtilities
using Documenter

DocMeta.setdocmeta!(QuantumUtilities, :DocTestSetup, :(using QuantumUtilities); recursive=true)

makedocs(;
    modules=[QuantumUtilities],
    authors="Federico Cerisola <federico@cerisola.net",
    repo="https://github.com/cerisola/QuantumUtilities.jl/blob/{commit}{path}#{line}",
    sitename="QuantumUtilities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cerisola.github.io/QuantumUtilities.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cerisola/QuantumUtilities.jl",
    devbranch="main",
)
