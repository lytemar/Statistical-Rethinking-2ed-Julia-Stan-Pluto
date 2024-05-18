### A Pluto.jl notebook ###
# v0.19.39

using Markdown
using InteractiveUtils

# ╔═╡ c600966f-c2bf-4683-9351-d6f5f18f1e30
using Pkg

# ╔═╡ 433031b8-424d-429e-9b5c-5d1c4faeea67
#Pkg.activate(expanduser("~/.julia/dev/SR2StanPluto"))

# ╔═╡ f115095c-2762-44d1-882e-e3ee1f02640b
begin
	# Notebook specific
	using GLM
	using MonteCarloMeasurements
	
	# Graphics related
	using CairoMakie

	# Causal inference support
	using GraphViz
	using CausalInference

	# Stan specific
	using SplitApplyCombine
	using StanSample
	using StanQuap
	
	# Project support libraries
	using StatisticalRethinking: sr_datadir, create_observation_matrix
	using RegressionAndOtherStories
end

# ╔═╡ 969d4bb6-0a0b-4540-b125-90be7b5779a7
md" ## 7.1 - The problem with parameters."

# ╔═╡ 2dac121d-11d0-4dd6-bf10-2f5121a44576
md" ##### Set page layout for notebook."

# ╔═╡ 3dd68075-470a-4e45-adf3-a110aecd9bb3
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 3500px;
    	padding-left: max(10px, 5%);
    	padding-right: max(5px, 20%);
	}
</style>
"""

# ╔═╡ d919a139-c5d3-4c32-bb9f-48115f75119b
md" #### Julia code snippet 7.01"

# ╔═╡ ae857001-30c9-4079-bd79-9afb818dd842
begin
	sppnames = [:afarensis, :africanus, :hapilis,
		:boisei, :rudolfensis, :ergaster, :sapiens]
	brainvol = [438, 452, 612, 521, 752, 871, 1350]
	masskg = [37, 35.5, 34.5, 41.5, 55.5, 61, 53.5]
	df = DataFrame(species = sppnames, brain = brainvol, mass = masskg)
end

# ╔═╡ bc2fe2eb-fb4c-4e4f-9f73-824b716c71ed
md" #### Julia code snippet 7.02"

# ╔═╡ 95a362ed-d23d-41c4-8fc9-745ef76c7241
begin
	df[!, :brain_s] = df.brain / maximum(df.brain) 
	scale_df_cols!(df, :mass)
	data = (N = size(df, 1), brain = df.brain_s, mass = df.mass_s)
end

# ╔═╡ f594c32d-d56a-4e90-8433-b2ae5a853843
df

# ╔═╡ dc95e7b0-96b5-4259-9cb9-1389769c165e
begin
	f = Figure(;size=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="body mass [kg]", ylabel="brain vol [cc]")
	scatter!(df.mass, df.brain)
	
	for (ind, species) in enumerate(df.species)
		yadj =8
		if species == :afarensis
			yadj = -38
		end
		annotations!(String(df[ind, :species]); position=(df[ind, :mass] - 0.9, df[ind, :brain] + yadj))
	end
	f
end

# ╔═╡ af4fabee-f2e5-4a6c-9fad-3bfb7d237f01
md" #### Julia code snippet 7.03"

# ╔═╡ 98812b6b-d394-4b90-b46b-986b04b6f56a
stan7_1 = "
data {
 int < lower = 1 > N; 			// Sample size
 vector[N] brain; 				// Outcome
 vector[N] mass; 				// Predictor
}

parameters {
 real a;                        // Intercept
 real bA;                       	// Slope (regression coefficients)
 real < lower = 0 > log_sigma;    	// Error SD
}

model {
  vector[N] mu;               		// mu is a vector
  a ~ normal(0.5, 1);         		//Priors
  bA ~ normal(0, 10);
  log_sigma ~ normal(0, 1);
  mu = a + bA * mass;
  brain ~ normal(mu , log_sigma);   // Likelihood
}
";

# ╔═╡ f186df35-4abb-4ba7-acbd-000156a9f1ba
let
	global m7_1s = SampleModel("m7.1s", stan7_1)
	global rc7_1s = stan_sample(m7_1s; data)
	success(rc7_1s) && describe(m7_1s, [:a, :bA, :log_sigma])
end

# ╔═╡ 58c4a78b-5024-41d0-aff9-4b1245cf64e2
if success(rc7_1s)
	post7_1s_df = read_samples(m7_1s, :dataframe)
	ms7_1s = model_summary(post7_1s_df, [:a, :bA, :log_sigma])
end

# ╔═╡ 22e354e5-7aaf-447a-8538-a4d66f8cecd4
m7_1_ols = lm(@formula(brain_s ~ 1 + mass_s), df)

# ╔═╡ e07bf075-394c-4e96-a509-e2bf0ec9e29e
m7_1_fit = fit(Normal, ms7_1s["a", "mean"] .+ ms7_1s["bA", "mean"] .* df.mass_s)

# ╔═╡ 9a196854-d7e3-4a99-8a7a-31b4cb24b011
describe(post7_1s_df)

# ╔═╡ cb7ddc89-c71b-41d7-9940-b48114793b38
log(mean(post7_1s_df.log_sigma))

# ╔═╡ 41eb1a4e-1bbd-4e92-9c8f-2c8c5c7c4e8a
stan7_1a = "
data {
 int < lower = 1 > N; 			// Sample size
 vector[N] brain; 				// Outcome
 vector[N] mass; 				// Predictor
}

parameters {
 real a; // Intercept
 real bA; // Slope (regression coefficients)
 real log_sigma; 
}

model {
  vector[N] mu;               	// mu is a vector
  a ~ normal(0.5, 1);         	//Priors
  bA ~ normal(0, 10);
  log_sigma ~ normal(0, 1);
  mu = a + bA * mass;
  brain ~ normal(mu , exp(log_sigma));   // Likelihood
}
";

# ╔═╡ 526277b4-d89f-4f29-bc3e-427802a3fbc6
begin
	m7_1as = SampleModel("m7.1as", stan7_1a)
	rc7_1as = stan_sample(m7_1as; data)
	success(rc7_1as) && describe(m7_1as, [:a, :bA, :log_sigma])
end

# ╔═╡ 5fbaa60b-0164-402f-bea7-69f3e0450dc5
if success(rc7_1as)
	post7_1as_df = read_samples(m7_1as, :dataframe)
	ms7_1as_df = model_summary(post7_1as_df, [:a, :bA, :log_sigma])
end

# ╔═╡ bf77a18b-24ff-4cca-8863-c5ae3300256f
exp(mean(post7_1as_df.log_sigma))

# ╔═╡ 71fffab5-23ac-43d2-a185-d41ccde39a7a
stan7_1b = "
data{
	int < lower = 1 > N; 			// Sample size
    vector[N] brain;
    vector[N] mass;
}
parameters{
    real a;
    real bA;
    real log_sigma;
}
model{
    vector[N] mu;
    a ~ normal( 0.5 , 1 );
    bA ~ normal( 0 , 10 );
    log_sigma ~ normal( 0 , 1 );
    for ( i in 1:N ) {
        mu[i] = a + bA * mass[i];
    }
    brain ~ normal( mu , exp(log_sigma) );
}
";

# ╔═╡ 5ffd89a9-577d-4da7-b074-71f3125e7441
let
	data = (N = size(df, 1), brain = df.brain_s, mass = df.mass_s)
	global m7_1bs = SampleModel("m7.1bs", stan7_1b)
	global rc7_1bs = stan_sample(m7_1bs; data)
	success(rc7_1bs) && describe(m7_1bs, [:a, :bA, :log_sigma])
end

# ╔═╡ d6471315-72f3-400a-ac3b-ab5149e0f685
if success(rc7_1bs)
	post7_1bs_df = read_samples(m7_1bs, :dataframe)
	ms7_1bs = model_summary(post7_1bs_df, [:a, :bA, :log_sigma])
end

# ╔═╡ 6bb25243-e9ec-4678-9174-0a876c8bfc9f
let
	x_s = -1.3:0.1:1.7
	y_s = ms7_1bs[:a, :mean] .+ ms7_1bs[:bA, :mean] .* x_s
	f = Figure(;size=default_figure_resolution)
	ax = Axis(f[1, 1])
	scatter!(df.mass_s, df.brain_s)
	lines!(x_s, y_s)
	f
end

# ╔═╡ 38700e2c-6382-48b9-99bf-14d3ec91b867
md" #### Julia code snippet 7.4"

# ╔═╡ 2bc6cb6f-39bc-42bc-b144-36ee2b6e2072
m1 = lm(@formula(brain_s ~ mass_s), df)

# ╔═╡ 93939e51-7f79-42e2-a557-5feef7bfc3c6
m2 = lm(@formula(brain ~ mass), df)

# ╔═╡ f590cd6d-1b58-469a-a582-373392a62aaf
deviance(m1)

# ╔═╡ 9d747c73-b112-4f73-a275-6f55269b66fa
deviance(m2)

# ╔═╡ 246c12fe-7d70-4c6b-aff3-0e5495abe5a4
md" #### Julia code snippet 7.5"

# ╔═╡ c301f616-dfbe-4399-949b-fc0e9c96f42b
function r2_is_bad(pred, df::DataFrame)
	s = mean(pred, dims=2)
	r = s - df.brain_s
	1 - var(r; corrected=false) / var(df.brain_s; corrected=false)
end

# ╔═╡ c505ed37-949e-4502-b343-a238fba678e1
md" #### Julia code snippet 7.6"

# ╔═╡ 788c5fc3-9073-467b-9173-067dd105de92
let
	preds = [ms7_1bs[:a, :mean] .+ ms7_1bs[:bA, :mean] .* x for x in df.mass_s]
	r2_is_bad(preds, df)
end

# ╔═╡ 1dc42f48-1f06-4b3c-87d7-9e5a9a6a0f0f
md" #### Julia code snippet 7.7 - 7.10"

# ╔═╡ 6c254059-d2db-497a-ac8b-0d582e939816
stan7_2 = "
data {
	int < lower = 1 > N; 			// Sample size
	int < lower = 1 > K;			// Degree of polynomial
	int < lower = 1 > L;			// Number of predicted brain values
	vector[N] brain; 				// Outcome
	matrix[N, K] mass; 				// Predictor
	matrix[L, K] mass_steps;        // Predictor steps
}
parameters {
 real a;                            // Intercept
 vector[K] b;                       // K slope(s)
 real log_sigma;    	            // Error
}
transformed parameters {
    vector[N] mu;
	vector[L] brain_pred;
    mu = a + mass * b;
	brain_pred = a + mass_steps * b;
}
model {
  a ~ normal(0.5, 1);         	//Priors
  b ~ normal(0, 10);
  log_sigma ~ normal(0, 1);
  brain ~ normal(mu , exp(log_sigma));   // Likelihood
}
";

# ╔═╡ 2bddd8c1-5b58-4b5f-9a17-733baf55365d
md" #### Julia code snippet for figure 7.3"

# ╔═╡ c5387611-2eb0-4bc9-96a0-06a056f28482
md"
!!! note
Note that this figure doesn't reach R2 = 1 and bounds shrinking as much as figure 7.3 in the book. Run below cell a few times to see the R2 values vary."

# ╔═╡ 6b3921fb-83de-4751-850c-33b6d8e9c9fe
let
	f = Figure(; size=default_figure_resolution)
	r2array = Float64[]
	for k in 1:6
		lr = k > 3 ? 2 : 1
		mc = k > 3 ? k - 3 : k
		N = size(df, 1)
		mass = zeros(N, k)
		for j in 1:N
			mass[j, :] = [df.mass_s[j]^i for i in 1:k]
		end
		mass_step_range = LinRange(minimum(df.mass_s), maximum(df.mass_s), 100)
		l = length(mass_step_range)
		mass_steps = zeros(l, k)
		for j in 1:l
			mass_steps[j, :] = [mass_step_range[j]^i for i in 1:k]
		end
		data = (N = N, K=k, L=l, brain = df.brain_s, mass = mass, mass_steps=mass_steps)
		sm = SampleModel("m7_2", stan7_2)
		rc = stan_sample(sm; data)
		global ndf = read_samples(sm, :nesteddataframe)
		#dft = DataFrame(m=df.mass_s, b = mean(ndf.mu), u=mean(ndf.mu) .+ std(ndf.mu), 
		#	l=mean(ndf.mu) .- std(ndf.mu))
		#sort!(dft, :m)
		dft = DataFrame(m=mass_step_range, b = mean(ndf.brain_pred), 
			u=mean(ndf.brain_pred) .+ std(ndf.brain_pred), 
			l=mean(ndf.brain_pred) .- std(ndf.brain_pred))
		
		mus = mean(ndf.mu)
		r = mus - df.brain_s
		append!(r2array, 1 - var(r; corrected=false) / var(df.brain_s; corrected=false))
		
		ax = Axis(f[lr, mc]; xlabel="Body mass [kg]", ylabel="Brain volume [cc]", 
			title="R2 = $(round(r2array[end]; digits=2)), k=$k")
		#ylims!(-0.1, 1.4)
		Makie.scatter!(df.mass_s, df.brain_s)
		Makie.lines!(dft.m, dft.b)
		Makie.band!(dft.m, dft.l, dft.u; color=(:grey, 0.3))
		
	end
	f
end

# ╔═╡ fd0323b5-e48e-4f10-99d5-92013a66c488
mean(ndf.mu)

# ╔═╡ d7b677bc-9eaf-4395-8dd3-ef5b8914aefa
mean(ndf.b)

# ╔═╡ Cell order:
# ╟─969d4bb6-0a0b-4540-b125-90be7b5779a7
# ╟─2dac121d-11d0-4dd6-bf10-2f5121a44576
# ╠═3dd68075-470a-4e45-adf3-a110aecd9bb3
# ╠═c600966f-c2bf-4683-9351-d6f5f18f1e30
# ╠═433031b8-424d-429e-9b5c-5d1c4faeea67
# ╠═f115095c-2762-44d1-882e-e3ee1f02640b
# ╟─d919a139-c5d3-4c32-bb9f-48115f75119b
# ╠═ae857001-30c9-4079-bd79-9afb818dd842
# ╟─bc2fe2eb-fb4c-4e4f-9f73-824b716c71ed
# ╠═95a362ed-d23d-41c4-8fc9-745ef76c7241
# ╠═f594c32d-d56a-4e90-8433-b2ae5a853843
# ╠═dc95e7b0-96b5-4259-9cb9-1389769c165e
# ╟─af4fabee-f2e5-4a6c-9fad-3bfb7d237f01
# ╠═98812b6b-d394-4b90-b46b-986b04b6f56a
# ╠═f186df35-4abb-4ba7-acbd-000156a9f1ba
# ╠═58c4a78b-5024-41d0-aff9-4b1245cf64e2
# ╠═22e354e5-7aaf-447a-8538-a4d66f8cecd4
# ╠═e07bf075-394c-4e96-a509-e2bf0ec9e29e
# ╠═9a196854-d7e3-4a99-8a7a-31b4cb24b011
# ╠═cb7ddc89-c71b-41d7-9940-b48114793b38
# ╠═41eb1a4e-1bbd-4e92-9c8f-2c8c5c7c4e8a
# ╠═526277b4-d89f-4f29-bc3e-427802a3fbc6
# ╠═5fbaa60b-0164-402f-bea7-69f3e0450dc5
# ╠═bf77a18b-24ff-4cca-8863-c5ae3300256f
# ╠═71fffab5-23ac-43d2-a185-d41ccde39a7a
# ╠═5ffd89a9-577d-4da7-b074-71f3125e7441
# ╠═d6471315-72f3-400a-ac3b-ab5149e0f685
# ╠═6bb25243-e9ec-4678-9174-0a876c8bfc9f
# ╟─38700e2c-6382-48b9-99bf-14d3ec91b867
# ╠═2bc6cb6f-39bc-42bc-b144-36ee2b6e2072
# ╠═93939e51-7f79-42e2-a557-5feef7bfc3c6
# ╠═f590cd6d-1b58-469a-a582-373392a62aaf
# ╠═9d747c73-b112-4f73-a275-6f55269b66fa
# ╟─246c12fe-7d70-4c6b-aff3-0e5495abe5a4
# ╠═c301f616-dfbe-4399-949b-fc0e9c96f42b
# ╟─c505ed37-949e-4502-b343-a238fba678e1
# ╠═788c5fc3-9073-467b-9173-067dd105de92
# ╟─1dc42f48-1f06-4b3c-87d7-9e5a9a6a0f0f
# ╠═6c254059-d2db-497a-ac8b-0d582e939816
# ╟─2bddd8c1-5b58-4b5f-9a17-733baf55365d
# ╟─c5387611-2eb0-4bc9-96a0-06a056f28482
# ╠═6b3921fb-83de-4751-850c-33b6d8e9c9fe
# ╠═fd0323b5-e48e-4f10-99d5-92013a66c488
# ╠═d7b677bc-9eaf-4395-8dd3-ef5b8914aefa
