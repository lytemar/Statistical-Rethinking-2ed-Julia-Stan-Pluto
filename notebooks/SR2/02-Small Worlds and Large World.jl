### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 718fb06b-e579-4a7d-8328-c366715dce10
using Pkg

# ╔═╡ 899a24c4-1316-471f-b3ba-5ae685f903f3
Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 8be42366-9118-426b-9407-f5eb17ff80f0
begin
	# General packages
	using Distributions
	using LaTeXStrings

	# Grphics related
	using PlutoUI
	using CairoMakie
	
	# Specific for SR2StanPluto
	using StanSample
	using StanQuap
	using StanPathfinder
	
	# Projects
	using RegressionAndOtherStories
end

# ╔═╡ f700b150-8382-44af-893e-1cbd7d97610d
md" ## Chapter 2 - Small Worlds and Large World."

# ╔═╡ 18b85845-f1ca-4a19-afb0-6e4282e9228f
md" ##### Set page layout for notebook."

# ╔═╡ bda8c10f-eef0-4f0a-a8ee-219792ac4b34
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 3500px;
    	padding-left: max(80px, 0%);
    	padding-right: max(200px, 18%);
	}
</style>
"""

# ╔═╡ 4fdc6ba2-f1fb-4525-87c8-0aa5800fa4b5
md" ### 2.1 The garden of forking data."

# ╔═╡ 20c4e1eb-3420-4ab7-aa4d-0a2f4848d65f
md" #### Julia code snippet 2.1"

# ╔═╡ 4ef0e151-9fc4-4ab3-81a2-efb29c7a3bb3
begin
    ways = [0, 3, 8, 9, 0]
    ways = ways ./ sum(ways)
end

# ╔═╡ 46bad21c-43a2-4fba-9882-6ba9a8ade36d
md" ##### One blue, 3 white balls"

# ╔═╡ 3f24f098-be77-4fdc-a005-1de3df58d3b3
[1 * 3 * 1, 4 * 4 * 4]

# ╔═╡ ff195304-6c9f-4a12-8c52-54f6fa916ffe
md" ##### Two blue, 2 white balls"

# ╔═╡ c6e2b7df-2fae-43d1-821d-4c39a615b7b2
[2 * 2 * 2, 4 * 4 * 4]

# ╔═╡ 8e7db533-67e2-45b8-a9db-c7ab720e86a7
md" ##### Three blue, 1 white balls"

# ╔═╡ 8f01a72a-1b71-48b7-8764-9841a44aec4c
[3 * 1 * 3, 4 * 4 * 4]

# ╔═╡ 0e848dd0-10f9-4a14-a177-99ad2658c7ea
md" ### 2.2 Building a model."

# ╔═╡ 3cdfdba9-a25e-435a-9e8a-927297e6d341
md" ### 2.3 Components of the model."

# ╔═╡ c2b719c4-7ff8-434c-af3c-52431016efe2
md" #### Julia code snippet 2.2"

# ╔═╡ 5d237149-f37a-4cac-b730-78f309f93b31
let
    b = Binomial(9, 0.5)
    pdf(b, 6)
end

# ╔═╡ 2e1778a1-31e3-4137-8699-3d367a652c02
mean(rand(Binomial(3, 1/4), 100000))/3

# ╔═╡ b1db9ec1-09df-4ba9-8d80-e9f79d714a78
mean(rand(Binomial(3, 2/4), 100000))/3

# ╔═╡ 388d1102-1dc9-41d3-815b-3953462976f1
mean(rand(Binomial(3, 3/4), 100000))/3

# ╔═╡ 3de688bc-9383-42db-b0b3-6f0bef53c373
md" #### Julia code snippet 2.3"

# ╔═╡ 58d16e75-43c2-4328-bc67-82ac40c1de31
md" ##### Size of the grid."

# ╔═╡ 3235bfbf-2b53-4762-9842-105608343660
n_size = 30

# ╔═╡ 663b0e04-d4de-4d36-b688-c4acdabf8179
md" ##### Grid and prior."

# ╔═╡ 83a06085-b389-4aa3-aefc-451511c36dc5
p_grid = range(0, 1; length=n_size)

# ╔═╡ b8a3eab4-9f42-4d35-bd90-fb1df8ad9abc
prior = [pdf(Uniform(0, 1), i) for i in p_grid] ./ n_size

# ╔═╡ b2495576-c63d-4eda-8e62-bb7dd517ecd4
md" ##### Compute likelihood at each value in grid."

# ╔═╡ a2d4e3d2-a42c-4e42-8889-f63a2fb0b2c0
likelihood = [pdf(Binomial(9, p), 6) for p in p_grid];

# ╔═╡ ef3b5609-c9e9-4a22-a0a1-15a3d03205a7
let
	f = Figure(;size = default_figure_resolution)
	ax = Axis(f[1, 1]; title="Distribution of likelihood on the grid")
	hideydecorations!(ax, ticks = false)
	lines!(p_grid, likelihood)
	lines!(p_grid, prior; linestyle=:dash)
	f
end

# ╔═╡ 6a658e52-e842-47e1-a3c1-d0f2dab1c979
md" ##### Compute the product of likelihood and prior and standardize."

# ╔═╡ 268cb94b-4f87-406b-a5fd-934f1327d99a
let
	unstd_posterior = likelihood .* prior
	posterior = unstd_posterior / sum(unstd_posterior)
end

# ╔═╡ 961b498c-486b-42a0-8d20-bb4d9a87624a
md" ##### Reproduce figure 2.5."

# ╔═╡ 2cb6682f-ca62-4c55-ae3e-426f3b4b69fe
let
	f = Figure(;size = default_figure_resolution)
	obs = "WLWWWLWLW"
	w = 0
	l = 0
	r = 1
	c = 1
	# Below prior is not proper!
	lprior = prior * 15
	for (i, ob) in enumerate(obs)
		ob == 'W' ? (w += 1) : (l += 1)
		likelihood = [pdf(Binomial(i, p), w) for p in p_grid]
		c = i
		if i > 3
			if i < 7
				r = 2; c = i - 3
			else
				r = 3; c = i - 6
			end
		end
		ax = Axis(f[r, c]; title="N = $i:   \"$(obs[1:i])\"", xlabel="Proportion water", ylabel="Plausability")
		i < 2 && hideydecorations!(ax, ticks = false)
		lines!(p_grid, likelihood)
		lines!(p_grid, lprior; linestyle=:dash)
		lprior = copy(likelihood)
	end
	f
end

# ╔═╡ 3c2c38c3-8da4-4c43-a8d9-a6bfe466f6d7
md" ### 2.4 Making the model go"

# ╔═╡ 09aa2045-6d1b-460a-b6d5-a45de36d0be9
md" #### Julia code snippet 2.4"

# ╔═╡ 8e3d0ba3-dfb7-49d2-8b9b-afec7337f514
@bind N PlutoUI.Slider(5:1:1000, default=20)

# ╔═╡ 4124aed5-9b68-460c-af5b-155e8ec5cc53
let
	p_grid = range(0, 1; length=N)
	prior = repeat([1.0], N)
	likelihood = [pdf(Binomial(9, p), 6) for p in p_grid]
	unstd_posterior = likelihood .* prior
	posterior = unstd_posterior / sum(unstd_posterior / N)	
	f = Figure(;size = default_figure_resolution)
	ax = Axis(f[1, 1]; title="Posterior distribution of probability of water ($N points)",
	    xlabel="probability of water", 
	    ylabel="posterior probability")
	lines!(p_grid, posterior)
	scatter!(p_grid, posterior)
	f
end

# ╔═╡ 5dcea199-bad2-4b87-be43-ef417d487c14
md" ##### 2.4.3 Grid approximation"

# ╔═╡ 30f5e835-c43b-42f5-b56a-6bac186d11e5
md" #### Julia code snippet 2.5"

# ╔═╡ 1e556e16-37b0-4de0-a3f1-e9defde82f05
let
    size = 100
    p_grid = range(0, 1; length=size)

    # prior is different - 0 if p < 0.5, 1 if >= 0.5
    prior1 = convert(Vector{AbstractFloat}, p_grid .>= 0.5)

    # another prior to try (uncomment the line below)
    prior2 = exp.(-5*abs.(p_grid .- 0.5))

    # the rest is the same
    likelyhood = [pdf(Binomial(9, p), 6) for p in p_grid]

	f = Figure(;size = default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="probability of water", ylabel="posterior probability", title="$size points")
    unstd_posterior = likelyhood .* prior1
    posterior = unstd_posterior / sum(unstd_posterior / size)
	lines!(p_grid, posterior)
	scatter!(p_grid, posterior)
	ax = Axis(f[1, 2]; xlabel="probability of water", ylabel="posterior probability", title="$size points")
    unstd_posterior = likelyhood .* prior2
    posterior = unstd_posterior / sum(unstd_posterior / size)
	lines!(p_grid, posterior)
	scatter!(p_grid, posterior)
	f

end

# ╔═╡ b9d2bb69-4afc-4a82-85bd-5cdcd8f094aa
md" ##### 2.4.4 Quadratic approximation"

# ╔═╡ d50cd0f1-0d5a-4f7c-8242-72388f0e6565
md" #### Julia code snippet 2.6"

# ╔═╡ 8440c903-5fee-4399-9fd9-c2b491e83949
m2_1 = "
data{
    int W;
    int L;
}
parameters {
    real<lower=0, upper=1> p;
}
model {
    p ~ uniform(0, 1);
    W ~ binomial(W + L, p);
}
";

# ╔═╡ 8e0e6cae-838c-473d-a222-6f6eac13d989
begin
    m2_1s = SampleModel("m2_1s", m2_1)
    rc2_1s = stan_sample(m2_1s; data = (W=6, L=3))
	success(rc2_1s) && describe(m2_1s, [:p, :p_tilde])
end

# ╔═╡ 451c3c98-255c-4623-b0fa-19917f5573f7
if success(rc2_1s)
    post2_1s = read_samples(m2_1s, :nesteddataframe)
	ms2_1s = model_summary(post2_1s, [:p])
end

# ╔═╡ 16dfa2a7-be8f-46ea-964d-0b2b0697c1e5
begin
	qm, sm, om = stan_quap("m2_1s", m2_1; data = (W=6, L=3, P=10, p_tilde=0.1:0.1:1), init = (p = 0.5,))
	om.optim |> display
	qm
end

# ╔═╡ 7f7cb8a4-1f86-4c46-8755-2662e4daaf30
fit(Normal, post2_1s.p)

# ╔═╡ a5b33ca6-3119-4a22-8f0f-1e41c0c56d2a
fit_mle(Normal, post2_1s.p)

# ╔═╡ f7926385-36cd-4402-9930-921e797de2f4
md" Quadratic approximation using Stan's pathfinder variational method"

# ╔═╡ f63325a6-96a1-4d03-803e-f65a7f8f7624
let
	global pm = PathfinderModel("m2_1p", m2_1)
	rc = stan_pathfinder(pm; data = (W=6, L=3))
	
	if all(success.(rc))
		global df_variational = StanSample.convert_a3d(read_pathfinder(pm)..., Val(:dataframe))
	    create_pathfinder_profile_df(pm)
	end
end

# ╔═╡ ebda8b2d-e4f2-4c6f-88a5-abc2bcc4b38c
df_variational

# ╔═╡ 909210c1-4af0-44a1-8d1c-0ad9675bfc6a
fit(Normal, df_variational.p)

# ╔═╡ 07c86b5b-be1a-43dd-949b-84828bd9fa21
md"

!!! note

See https://github.com/StanJulia/StanExampleNotebooks.jl/tree/main/notebooks/Stan-intros for more details."

# ╔═╡ 8d0aaa63-8ca8-4f6d-b829-5985c0ce5ec7
md" #### Julia code snippet 2.7"

# ╔═╡ 3eaca463-2743-456f-a6aa-705799f99592
let
    x = range(0, 1; length=101)
	
	f = Figure(;size = default_figure_resolution)
	ax = Axis(f[1, 1]; title="W=6. L=3")
    W = 6; L = 3
	qm, sm, om = stan_quap("m2_1s", m2_1; data = (W=6, L=3), init = (p = 0.5,))
    b = Beta(W+1, L+1)
    lines!(x, pdf.(b, x))
	lines!(x, pdf.(qm.distr, x))
	
	ax = Axis(f[1, 2]; title="W=12. L=6")
    W = 12; L = 6
	qm, sm, om = stan_quap("m2_1s", m2_1; data = (W=12, L=6), init = (p = 0.5,))
    b = Beta(W+1, L+1)
    lines!(x, pdf.(b, x))
	lines!(x, pdf.(qm.distr, x))
	
	ax = Axis(f[1, 3]; title="W=24. L=12")
	qm, sm, om = stan_quap("m2_1s", m2_1; data = (W=24, L=12), init = (p = 0.5,))
	W = 24; L = 12
    b = Beta(W+1, L+1)
    lines!(x, pdf.(b, x))
	lines!(x, pdf.(qm.distr, x))
	f
end

# ╔═╡ cb3102a3-dbfa-4bbd-96cb-b4c680154df4
let
	W = 6; L = 3
	qm, sm, om = stan_quap("m2_1s", m2_1; data = (W=6, L=3), init = (p = 0.5,))
	√qm.vcov
end

# ╔═╡ ae452813-f237-4568-aea6-82f3723622b5
md" ##### Quadratic approximation."

# ╔═╡ 81b47785-3f4d-4a0c-b35e-85b0de760029
md" #### Julia code snippet 2.8"

# ╔═╡ d19b9ff8-cbdf-45cf-878b-9edc630d9c69
begin
    n_samples = 4000
    p = Vector{Float64}(undef, n_samples)
    p[1] = 0.5
    W, L = 6, 3

    for i ∈ 2:n_samples
        p_old = p[i-1]
        p_new = rand(Normal(p_old, 0.1))
        if p_new < 0
            p_new = abs(p_new)
        elseif p_new > 1
            p_new = 2-p_new
        end

        q0 = pdf(Binomial(W+L, p_old), W)
        q1 = pdf(Binomial(W+L, p_new), W)
        u = rand(Uniform())
        p[i] = (u < q1 / q0) ? p_new : p_old
    end
end

# ╔═╡ 57985c29-eb61-40bc-9478-f092650ad68d
md" #### Julia code snippet 2.9"

# ╔═╡ 3c227bff-f7db-49f5-aaad-f2a58fcf4151
let
	x = range(0, 1; length=101)
	f = Figure(;size =  default_figure_resolution)
	ax = Axis(f[1, 1]; title = "Posterior sample density (blue) and Beta analytical density (darkred)")
    density!(p; color = (:lightblue, 0.3), strokecolor = :blue, strokewidth = 3)
    b = Beta(W+1, L+1)
    lines!(x, pdf.(b, x); color=:darkred, linewidth = 3)
	f
end

# ╔═╡ Cell order:
# ╟─f700b150-8382-44af-893e-1cbd7d97610d
# ╟─18b85845-f1ca-4a19-afb0-6e4282e9228f
# ╠═bda8c10f-eef0-4f0a-a8ee-219792ac4b34
# ╠═718fb06b-e579-4a7d-8328-c366715dce10
# ╠═899a24c4-1316-471f-b3ba-5ae685f903f3
# ╠═8be42366-9118-426b-9407-f5eb17ff80f0
# ╟─4fdc6ba2-f1fb-4525-87c8-0aa5800fa4b5
# ╟─20c4e1eb-3420-4ab7-aa4d-0a2f4848d65f
# ╠═4ef0e151-9fc4-4ab3-81a2-efb29c7a3bb3
# ╟─46bad21c-43a2-4fba-9882-6ba9a8ade36d
# ╠═3f24f098-be77-4fdc-a005-1de3df58d3b3
# ╟─ff195304-6c9f-4a12-8c52-54f6fa916ffe
# ╠═c6e2b7df-2fae-43d1-821d-4c39a615b7b2
# ╠═8e7db533-67e2-45b8-a9db-c7ab720e86a7
# ╠═8f01a72a-1b71-48b7-8764-9841a44aec4c
# ╟─0e848dd0-10f9-4a14-a177-99ad2658c7ea
# ╟─3cdfdba9-a25e-435a-9e8a-927297e6d341
# ╟─c2b719c4-7ff8-434c-af3c-52431016efe2
# ╠═5d237149-f37a-4cac-b730-78f309f93b31
# ╠═2e1778a1-31e3-4137-8699-3d367a652c02
# ╠═b1db9ec1-09df-4ba9-8d80-e9f79d714a78
# ╠═388d1102-1dc9-41d3-815b-3953462976f1
# ╟─3de688bc-9383-42db-b0b3-6f0bef53c373
# ╟─58d16e75-43c2-4328-bc67-82ac40c1de31
# ╟─3235bfbf-2b53-4762-9842-105608343660
# ╟─663b0e04-d4de-4d36-b688-c4acdabf8179
# ╠═83a06085-b389-4aa3-aefc-451511c36dc5
# ╠═b8a3eab4-9f42-4d35-bd90-fb1df8ad9abc
# ╟─b2495576-c63d-4eda-8e62-bb7dd517ecd4
# ╠═a2d4e3d2-a42c-4e42-8889-f63a2fb0b2c0
# ╠═ef3b5609-c9e9-4a22-a0a1-15a3d03205a7
# ╟─6a658e52-e842-47e1-a3c1-d0f2dab1c979
# ╠═268cb94b-4f87-406b-a5fd-934f1327d99a
# ╟─961b498c-486b-42a0-8d20-bb4d9a87624a
# ╠═2cb6682f-ca62-4c55-ae3e-426f3b4b69fe
# ╟─3c2c38c3-8da4-4c43-a8d9-a6bfe466f6d7
# ╟─09aa2045-6d1b-460a-b6d5-a45de36d0be9
# ╠═8e3d0ba3-dfb7-49d2-8b9b-afec7337f514
# ╠═4124aed5-9b68-460c-af5b-155e8ec5cc53
# ╟─5dcea199-bad2-4b87-be43-ef417d487c14
# ╟─30f5e835-c43b-42f5-b56a-6bac186d11e5
# ╠═1e556e16-37b0-4de0-a3f1-e9defde82f05
# ╟─b9d2bb69-4afc-4a82-85bd-5cdcd8f094aa
# ╟─d50cd0f1-0d5a-4f7c-8242-72388f0e6565
# ╠═8440c903-5fee-4399-9fd9-c2b491e83949
# ╠═8e0e6cae-838c-473d-a222-6f6eac13d989
# ╠═451c3c98-255c-4623-b0fa-19917f5573f7
# ╠═16dfa2a7-be8f-46ea-964d-0b2b0697c1e5
# ╠═7f7cb8a4-1f86-4c46-8755-2662e4daaf30
# ╠═a5b33ca6-3119-4a22-8f0f-1e41c0c56d2a
# ╟─f7926385-36cd-4402-9930-921e797de2f4
# ╠═f63325a6-96a1-4d03-803e-f65a7f8f7624
# ╠═ebda8b2d-e4f2-4c6f-88a5-abc2bcc4b38c
# ╠═909210c1-4af0-44a1-8d1c-0ad9675bfc6a
# ╟─07c86b5b-be1a-43dd-949b-84828bd9fa21
# ╟─8d0aaa63-8ca8-4f6d-b829-5985c0ce5ec7
# ╠═3eaca463-2743-456f-a6aa-705799f99592
# ╠═cb3102a3-dbfa-4bbd-96cb-b4c680154df4
# ╟─ae452813-f237-4568-aea6-82f3723622b5
# ╟─81b47785-3f4d-4a0c-b35e-85b0de760029
# ╠═d19b9ff8-cbdf-45cf-878b-9edc630d9c69
# ╟─57985c29-eb61-40bc-9478-f092650ad68d
# ╠═3c227bff-f7db-49f5-aaad-f2a58fcf4151
