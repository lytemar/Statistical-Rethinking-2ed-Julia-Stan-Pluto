## Note

After many years I have decided to step away from my work with Stan and Julia. My plan is to be around until the end of 2024 for support if someone decides to step in and take over further development and maintenance work.

At the end of 2024 I'll archive the different packages and projects included in the Github organisations StanJulia, StatisticalRethingJulia and RegressionAndOtherStoriesJulia if no one is interested (and time-wise able!) to take on this work.

I have thoroughly enjoyed working on both Julia and Stan and see both projects mature during the last 15 or so years. And I will always be grateful for the many folks who have helped me on numerous occasions. Both the Julia and the Stan community are awesome to work with! Thanks a lot!

## Purpose of SR2StanPluto.jl (v5.7)

As stated many times by the author in his [online lectures](https://www.youtube.com/watch?v=ENxTrFf9a7c&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI), StatisticalRethinking is a hands-on course. This project is intended to assist with the hands-on aspect of learning the key ideas in StatisticalRethinking. 

SR2StanPluto is a Julia project that uses Pluto notebooks for this purpose. Each notebook demonstrates Julia versions of `code snippets` and `mcmc models` contained in the R package "rethinking" associated with the book [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath.

This Julia project uses Stan (the `cmdstan` executable) as the underlying mcmc implementation. Please see Stan.jl and/or StanSample.jl for details.

## Important note

From v5 onwards the basis will no longer be StatisticalRethinking.jl but RegressionAndOtherStories.jl. Both packages have very similar content, but StatisticalRethinking.jl uses Plots.jl while RegressionAndOtherStories.jl is using (Cairo)Makie.jl.

Tagged version 4.2.0 is the last more or less complete set of scripts covering `the old` chapters 1 to 11.

## Using Pluto's package management (or not!)

Most notebooks include a line like `#Pkg.activate("~/.julie/dev/SR2StanPluto"))`. On the Github repo it is usualy commented out. While developing code I run notebooks inside a project environment (by uncommenting that line).  Occasionally I update the package (SR2StanPluto) in the REPL when new versions of the project itself or its dependencies become available. It requires SR2StanPluto.jl to be installed (in e.g. `~/.julia/dev` in the above example).

If you frequently switch between different notebooks, uncommenting (or adding) the `Pkg.activate(...)` line is also much faster.

## Usage

To (locally) reproduce and use this project, do the following:

1. Download this [project](https://github.com/StatisticalRethinkingJulia/SR2StanPluto.jl) from Github and move to the downloaded directory, e.g.:

```
$ cd ./julia/dev
$ git clone https://github.com/StatisticalRethinkingJulia/SR2StanPluto.jl SR2StanPluto
$ cd SR2StanPluto
```
Or select a particular tagged version, i.e. `...//SR2StanPluto.jl@4.2.0 ...`.

The next step assumes your `basic` Julia environment includes `Pkg` and `Pluto`.

2. Start a Pluto notebook server.
```
$ cd notebooks
$ julia

julia> using Pluto
julia> Pluto.run()
```
3. A Pluto page should open in a browser.

4. Select a notebook in the `open a file` entry box, e.g. type `./`, select a notebook collection and a notebook.. 

## Usage details

All "rethinking" data files are stored and maintained in StatisticalRethinking.jl and can be accessed via `sr_datadir(...)`. See `notebooks/00-Preface.jl` for an example.

By default `read_samples(m5_1s)` returns a StanTable with the results. In general
it is safer to specify the desired format, i.e. `read_samples(m5_1s, :table)` as
the Julia eco-sytem is still evolving rapidly with new options.

Using `read_samples(m5_1s, :...)` makes it easy to convert samples to other formats.

In version 5 I expect to mainly use the output_formats :dataframe and :nesteddataframe.

For InferenceObjects.jl there is a separate function `inferencedata(m1_1s)`. 

See the [StanExampleNotebooks.jl](https://github.com/StanJulia/StanExampleNotebooks.jl) for an example Pluto notebook.

## Status

SR2StanPluto.jl is compatible with the 2nd edition of the book. Not all notebooks have been converted to Makie.jl yet. These are in the SR1 subdirectory. In the SR3 subdirectory I've started to look at the SR 2023 lecture examples, mainly as illustrated in the "Statistical Rethinking 2023 Python Notes".

Beginning with v5.3.0 StructuralCausalModels.jl is repaced by CausalInference.jl as an extension. To display DAGs, GraphViz.jl and CairoMakie.jl are used. This is a long term project!

ParetoSmoothedImportanceSampling.jl is included as a dependency in the StatisticalRethinking.jl v3+ package. 

Definitely WIP! See also below version 5 info.

Any feedback is appreciated. Please open an issue if you have questions or suggestions.

## Acknowledgements

Of course, without the excellent textbook by Richard McElreath, this project would not have been possible. The author has also been supportive of this work and gave permission to use the datasets.

This repository and format is influenced by previous versions of StatisticalRethinking.jl, work by Karajan, Max Lapan and many other contributors.

## Versions

### Version 6

1. Updates for chapters 7 and 8 (in SR2).

### Version 5.4.0-5.5.11

1. Further updates in using CausalInference and GraphViz.

### Version 5.3

1. Switch to (Cairo)Makie.jl, Graphs.jl, CausalInference.jl, GraphViz.jl (and more probably).

### Version 5

1. Version 5 is a breaking change!
2. A new look is taken at packages available in the Julia ecosystem.

### Version 4

1. SR2StanPluto v4+ requires StatisticalRethinking v4+.

### versions 2 & 3

1. Many additions for 2nd edition of Statistical Rethinking book.
2. Version 3 switched to using StanSample and StanQuap

### Version 1

1. Initial versions (late Nov 2020).

