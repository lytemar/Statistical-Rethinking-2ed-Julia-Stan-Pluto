### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 6df5faaf-d262-45eb-8e8d-523335528419
using Pkg

# ╔═╡ afab9c0a-401d-465a-9cd8-55330a4a6ce4
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ 772eeae8-c5f8-11ee-2f4c-390f6136f2d1
begin
	using Distributions
	using GLM
	
	using CairoMakie
	
	using StanSample
	using RegressionAndOtherStories
end

# ╔═╡ 0df948dd-785b-4807-b995-edc3405b1e9e
md" # Julia package ConjugatePriors.jl"

# ╔═╡ 379027c4-ff5f-4daa-890c-10763740aa28
md" ###### Widen the cells."

# ╔═╡ fde4fcee-98ba-4b08-975b-3c1aa55814ce
html"""
<style>
    main {
        margin: 0 auto;
        max-width: 2000px;
        padding-left: max(160px, 10%);
        padding-right: max(160px, 15%);
    }
</style>
"""

# ╔═╡ 0df304eb-f7fc-4437-becf-1c0d63e2867b
stan_01 = "
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ normal(alpha + beta * x, sigma);
}
generated quantities {
  array[N] real y_rep = normal_rng(alpha + beta * x, sigma);
}
";

# ╔═╡ 09a829d5-dd3e-43e2-8ea9-38a852f67126
begin
	N = 200
	alpha = 3.0
	beta = 2.0
	x = LinRange(-10, 10, N)
	y = alpha .+ beta .* x .+ rand(Normal(0, 1), N)
	data = (N=N, x=x, y=y)
end;

# ╔═╡ db1c234e-646e-4ced-b5cc-8381134020b7
begin
	sm = SampleModel("gq_1", stan_01)
	rc = stan_sample(sm; data)
	if success(rc)
		df = read_samples(sm, :nesteddataframe)
	end
	df
end

# ╔═╡ 551d810d-a315-4fbb-813a-c908102d5410
describe(df)

# ╔═╡ 4a7a059b-910f-481d-aa11-570cb31d8385
size(df.y_rep[1])

# ╔═╡ dc73b8ca-2318-4dbd-ac68-5ba95473937f
mean(df.y_rep)

# ╔═╡ 802ac3c7-5a2b-471f-8ffb-ec134c67036d
std(df.y_rep)

# ╔═╡ 9ce99a3f-719e-46d7-836c-7319be2e4c6a
stan_02 = "
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
parameters {
  real<lower=0> lambda;
}
model {
  lambda ~ gamma(1, 1);
  y ~ poisson(lambda);
}
generated quantities {
  int<lower=0> y_tilde = poisson_rng(lambda);
}
";

# ╔═╡ e98aa616-c7fa-4aa1-aca2-3329d8b9beaa
begin
	sm2 = SampleModel("gq_2", stan_02)
	rc2 = stan_sample(sm2; data=(N=9, y=[1, 0, 0, 1, 0, 0, 0, 1, 1]))
	if success(rc2)
		df2 = read_samples(sm2, :dataframe)
	end
	df2
end

# ╔═╡ df26b625-a523-4583-9e6f-aa90269e96b8
stan_03 = "
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  p ~ exponential(1);
  y ~ binomial(N, p);
}
generated quantities {
  int<lower=0> p_tilde = binomial_rng(N, p);
}
";

# ╔═╡ 2f4f3f8b-22a3-4613-a326-a28a76ad9947
begin
	sm3 = SampleModel("gq_3", stan_03)
	rc3 = stan_sample(sm3; data=(N=9, y=rand(Binomial(9, 0.6), 9)))
	if success(rc3)
		df3 = read_samples(sm3, :dataframe)
	end
	df3
end

# ╔═╡ c5a528cc-0261-4298-b815-2b4715e57373
describe(df3)

# ╔═╡ 20adf8b5-854f-42ab-8f3f-0ec0debe47bf
stan_04 = "
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  p ~ exponential(1);
  y ~ bernoulli(p);
}
generated quantities {
  int<lower=0> p_tilde = bernoulli_rng(p);
}
";

# ╔═╡ f8c1a84b-cf87-4a25-817d-4cbc49a26c46
begin
	sm4 = SampleModel("gq_4", stan_04)
	rc4 = stan_sample(sm4; data=(N=20, y=convert.(Int, rand(Bernoulli(0.6), 20))))
	if success(rc4)
		df4 = read_samples(sm4, :dataframe)
	end
	df4
end

# ╔═╡ a0562a7f-f459-487e-9fec-44188ca2da89
describe(df4)

# ╔═╡ fce0027b-86c7-4fb3-95c4-aa89f677bf3b
let
	f = Figure(; size=default_figure_resolution)
	pvalues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	ax = Axis(f[1, 1:length(pvalues)])
	xlims!(0, 1)
	density!(df4.p)
	for (i, p) in enumerate(pvalues)
		ax = Axis(f[2, i]; title="p = $(pvalues[i])")
		xlims!(0, 1)
		rc = stan_sample(sm4; data=(N=20, y=convert.(Int, rand(Bernoulli(p), 20))))
		if success(rc4)
			df = read_samples(sm4, :dataframe)
		end
		hist!(df.p; bins=10)
	end
	f
end

# ╔═╡ Cell order:
# ╟─0df948dd-785b-4807-b995-edc3405b1e9e
# ╠═379027c4-ff5f-4daa-890c-10763740aa28
# ╠═fde4fcee-98ba-4b08-975b-3c1aa55814ce
# ╠═6df5faaf-d262-45eb-8e8d-523335528419
# ╠═afab9c0a-401d-465a-9cd8-55330a4a6ce4
# ╠═772eeae8-c5f8-11ee-2f4c-390f6136f2d1
# ╠═0df304eb-f7fc-4437-becf-1c0d63e2867b
# ╠═09a829d5-dd3e-43e2-8ea9-38a852f67126
# ╠═db1c234e-646e-4ced-b5cc-8381134020b7
# ╠═551d810d-a315-4fbb-813a-c908102d5410
# ╠═4a7a059b-910f-481d-aa11-570cb31d8385
# ╠═dc73b8ca-2318-4dbd-ac68-5ba95473937f
# ╠═802ac3c7-5a2b-471f-8ffb-ec134c67036d
# ╠═9ce99a3f-719e-46d7-836c-7319be2e4c6a
# ╠═e98aa616-c7fa-4aa1-aca2-3329d8b9beaa
# ╠═df26b625-a523-4583-9e6f-aa90269e96b8
# ╠═2f4f3f8b-22a3-4613-a326-a28a76ad9947
# ╠═c5a528cc-0261-4298-b815-2b4715e57373
# ╠═20adf8b5-854f-42ab-8f3f-0ec0debe47bf
# ╠═f8c1a84b-cf87-4a25-817d-4cbc49a26c46
# ╠═a0562a7f-f459-487e-9fec-44188ca2da89
# ╠═fce0027b-86c7-4fb3-95c4-aa89f677bf3b
