## Purpose of SR2StanPluto.jl

As stated many times by the author in his [online lectures](https://www.youtube.com/watch?v=ENxTrFf9a7c&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI), StatisticalRethinking is a hands-on course. This project is intended to assist with the hands-on aspect of learning the key ideas in StatisticalRethinking. 

SR2StanPluto is a Julia project that uses Pluto notebooks for this purpose. Each notebook demonstrates Julia versions of `code snippets` and `mcmc models` contained in the R package "rethinking" associated with the book [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath.

If you prefer to work with scripts instead of notebooks, a utility in the `src` directory is provided (`generate_scripts.jl`) to create scripts from all notebooks and store those in a newly created `scripts` directory. Note that this is a simple tool and will override all files in the `scripts` directory. For exploration purposes I suggest to move some of those scripts to e.g. the `research` directory.

This Julia project uses Stan (the `cmdstan` executable) as the underlying mcmc implementation. A companion project ( [SR2TuringPluto.jl](https://github.com/StatisticalRethinkingJulia/SR2TuringPluto.jl) ) uses Turing.jl.

## Installation

To (locally) reproduce and use this project, do the following (just once):

1. Download this [project](https://github.com/StatisticalRethinkingJulia/SR2StanPluto.jl) from Github and move to the downloaded directory, e.g.:

```
$ git clone https://github.com/StatisticalRethinkingJulia/SR2StanPluto.jl SR2StanPluto
$ cd SR2StanPluto
```

The next step assumes your `basic`Julia environment includes `Pkg` and `Pluto`.

2. Start a Pluto notebook server.
```
$ julia

julia> using Pluto
julia> Pluto.run()
```

Or you can start a new project and install Pluto:
$ julia 


3. A Pluto page should open in a browser.

## Usage

Select a notebook in the `open a file` entry box, e.g. type `./` and step to `./notebooks/00/clip-00-01-03s.jl`. 

A good notebook to initially glance over if `./notebooks/intros/intro-stan/intro-stan-01s.jl`.

The `data` directory, in DrWatson accessible through `datadir()`, can be used for locally generated data, exercises, etc. 

All "rethinking" data files are stored and maintained in StatisticalRethinking.jl and can be accessed via `sr_datadir(...)`. DrWatson provides several other handy shortcuts, e.g. projectdir().

A typical set of opening lines in each notebook:
```
using Pkg, DrWatson

# Note: Below sequence is important. First activate the project
# followed by `using` or `import` statements. Pretty much all
# scripts use StatisticalRethinking. If mcmc sampling is
# needed, it must be loaded before StatisticalRethinking:

# Load additional packages first, e.g.:
using GLM 

# Or 
#     using ParetoSmoothedImportanceSampling
#     using StructuralCausalModels
#     using MonteCarloMeasurements
#     using BSplines
#     using Optim
#     using ParetoSmooth

# Replace StanSample by StanQuap if stan_quap() is used.
using StanSample

using StatisticalRethinking
using StatisticalRethinkingPlots

# To access e.g. the Howell1.csv data file:
df = CSV.read(sr_datadir("Howell1.csv"), DataFrame)
df = df[df.age .>= 18, :]
```

## Naming conventions

All R snippets (fragments) have been organized in clips. Each clip is a notebook.

Clips are named as `clip-cc-fs-ls[s|t|d].jl` where

* `cc`               : Chapter number
* `fs`               : First snippet in clip
* `ls`               : Last snippet in clip
* `[s|sl|t|d|m]`     : Mcmc flavor used (s : Stan, t : Turing)

Note: `d` is reserved for a combination Soss/DynamicHMC, `sl` is reserved for Stan models using the `logpdf` formulation and `m` for Mamba.

The notebooks containing the clips are stored by chapter. In addition to clips, in the early notebook chapters it is also shown how to create some of the figures in the book, e.g. `Fig2.5s.jl` in `notebooks/chapter/02`.

Special introductory notebooks have been included in `notebooks/intros`, e.g.
`intro-stan/intro-stan-01s.jl` and `intro-R-users/distributions.jl`. It is suggested to at least glance over the `intro-stan` notebooks.

Great introductory notebooks showing Julia and statistics ( based on the [Statistics with Julia](https://statisticswithjulia.org/index.html) book ) can be found in [StatisticsWithJuliaPlutoNotebooks](https://github.com/StatisticalRethinkingJulia/StatisticsWithJuliaPlutoNotebooks.jl).

One goal for the changes in StatisticalRethinking v3 was to make it easier to compare and mix and match results from different mcmc implementations. Hence consistent naming of models and results is important. The models and the results of simulations are stored as follows:

Models and results:

0. stan5_1           : Stan language program
1. m5_1s             : The sampled StanSample model
2. q5_1s             : Stan quap model (NamedTuple similar to Turing)

Draws:

3. chns5_1s          : MCMCChains object (4000 samples from 4 chains)
4. part5_1s          : Stan samples (Particles notation)
5. quap5_1s          : Quap samples (Particles notation)
6. nt5_1s            : NamedTuple with samples values
7. ka5_1s            : KeyedArray object (see AxisArrays.jl)
8. da5_1s            : DimArray object (see DimensionalData.jl)
9. st5_1s            : StanTable 0bject

The default for `read_samples(m1_1s)` is a StanTable chains object.

Results as a DataFrame:

10. prior5_1s_df      : Prior samples (DataFrame)
11. post5_1s_df       : Posterior samples (DataFrame)
12. quap5_1s_df       : Quap approximation to posterior samples (DataFrame)
13.pred5_1s_df       : Posterior predictions (DataFrame)

As before, the `s` at the end indicates Stan.

By default `read_samples(m5_1s)` returns a StanTable with the results. In general
it is safer to specify the desired format, i.e. `read_samples(m5_1s, :table)` as
the Julia eco-sytem is still evolving rapidly with new options.

Using `read_samples(m5_1s, :...)` makes it easy to convert samples to other formats.

## Status

SR2StanPluto.jl is compatible with the 2nd edition of the book.

StructuralCausalModels.jl ans ParetoSmoothedImportanceSampling.jl are included as experimental dependencies in the StatisticalRethinking.jl v3+ package. Both definitely WIP!

Any feedback is appreciated. Please open an issue.

## Acknowledgements

Of course, without the excellent textbook by Richard McElreath, this package would not have been possible. The author has also been supportive of this work and gave permission to use the datasets.

This repository and format is derived from previous versions of StatisticalRethinking.jl, work by Karajan, and many other contributors.

## Versions

### Version 4.0.0 (Under development)

1. SR2StanPluto v4+ requires StatisticalRethinking v4+.

### versions 2 & 3

1. Many additions for 2nd edition of Statistical Rethinking book.
2. Version 3 switched to using StanSample and StanQuap

### Version 1

1. Initial versions (late Nov 2020).

