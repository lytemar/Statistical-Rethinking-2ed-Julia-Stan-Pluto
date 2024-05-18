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
md" ## 7.1b - Rethinking section 7.1."

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


# ╔═╡ 4ca22108-83f5-4edc-815d-101c78639824
stan7_1 = "
data{
    int<lower=1> N;
    vector[N] brain_std;
    vector[N] mass_std;
}
parameters{
    vector[1] b;
    real a;
	real log_sigma;
}
model{
    vector[N] mu;
    b ~ normal( 0 , 10 );
    a ~ normal( 0.5 , 1 );
	log_sigma ~ normal(0, 1);
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i];
    }
    brain_std ~ normal( mu , exp(log_sigma) );
}
generated quantities{
    vector[N] mu;
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i];
    }
}
";

# ╔═╡ 17889d4d-04f9-4b3f-8d93-8524e1bb15f7
md" ##### Quick check if results from StanQuap's MAP estimates are in line with R"

# ╔═╡ bdf0c993-5eee-480d-928b-40212180a6d0
let
	N = size(df, 1)
	data = (N = N, brain_std = df.brain_s, mass_std = df.mass_s)
	init = (a = 0.0, b = zeros(1), sigma = 2)
	global q7_1c, m7_1c, o7_1c = stan_quap("m7.1s", stan7_1; data, init)
	mean.([o7_1c.optim[v] for v in Symbol.(["a", "b.1", "log_sigma"])])
end

# ╔═╡ 9e9cdf8b-51ee-4544-854e-a74c7a4cd6bf
q7_1c

# ╔═╡ a287a5dd-dc99-4877-9e24-47ae28138670
md" R:  0.5285431  0.1671091 -1.7067065"

# ╔═╡ cb5c5048-9d7a-4a4f-9ad0-01dd633a439e
ndf = read_samples(m7_1c, :nesteddataframe)

# ╔═╡ c25cf2bd-bef7-4edb-a139-3ff692c7735e
[mean(ndf.a), mean(ndf.b), mean(ndf.mu)]

# ╔═╡ ceca761b-d23b-429f-bb94-1b79173962e9
[std(ndf.a), std(ndf.b), std(ndf.mu)]

# ╔═╡ 7f4ffc11-6105-4265-844e-29982f6befc6
stan7_2 = "
data{
    int<lower=1> N;
    vector[N] brain_std;
    vector[N] mass_std;
}
parameters{
    vector[2] b;
    real a;
	real log_sigma;
}
model{
    vector[N] mu;
    b ~ normal( 0 , 10 );
    a ~ normal( 0.5 , 1 );
	log_sigma ~ normal(0, 1);
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2;
    }
    brain_std ~ normal( mu , exp(log_sigma) );
}
generated quantities{
    vector[N] mu;
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2;
    }
}
";|

# ╔═╡ c11b3bec-2e84-4bb2-97be-3f6b90f4c021
let
	N = size(df, 1)
	data = (N = N, brain_std = df.brain_s, mass_std = df.mass_s)
	init = (a = 0.0, b = zeros(2), sigma = 2)
	global q7_2c, m7_2c, o7_2c = stan_quap("m7.2s", stan7_2; data, init)
	mean.([o7_2c.optim[v] for v in Symbol.(["a", "b.1", "b.2", "log_sigma"])])
end

# ╔═╡ 6f7da2ec-fd9d-4479-874f-39e3129064d6
q7_2c

# ╔═╡ 84b08ed3-f571-41d0-aaee-6b32273826e2
ndf2 = read_samples(m7_2c, :nesteddataframe)

# ╔═╡ 146bba10-8e10-46ca-b111-e7321cd97760
[mean(ndf2.a), mean(ndf2.b), mean(ndf2.mu)]

# ╔═╡ 5e0a9743-6318-439b-bbf1-f415a681854f
md" R:  0.19512884 -0.09790736  0.61211382 -1.74971017 "

# ╔═╡ 0f844e0d-8dbb-40fe-b511-df22b455d376
stan7_3 = "
data{
    int<lower=1> N;
    vector[N] brain_std;
    vector[N] mass_std;
}
parameters{
    vector[3] b;
    real a;
	real log_sigma;
}
model{
    vector[N] mu;
    b ~ normal( 0 , 10 );
    a ~ normal( 0.5 , 1 );
	log_sigma ~ normal(0, 1);
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3;
    }
    brain_std ~ normal( mu , exp(log_sigma) );
}
generated quantities{
    vector[N] mu;
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3;
    }
}
";

# ╔═╡ ff144f4b-d7b7-4d4a-ba6f-d3f5e30b18a8
stan7_4 = "
data{
    int<lower=1> N;
    vector[N] brain_std;
    vector[N] mass_std;
}
parameters{
    vector[4] b;
    real a;
	real log_sigma;
}
model{
    vector[N] mu;
    b ~ normal( 0 , 10 );
    a ~ normal( 0.5 , 1 );
	log_sigma ~ normal(0, 1);
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3 + b[4] * mass_std[i]^4;
    }
    brain_std ~ normal( mu , exp(log_sigma) );
}
generated quantities{
    vector[N] mu;
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3 +
			b[4] * mass_std[i]^4;
    }
}
";

# ╔═╡ 82a0bc7d-3252-4176-ba8b-669f0caa6712
stan7_5 = "
data{
    int<lower=1> N;
    vector[N] brain_std;
    vector[N] mass_std;
}
parameters{
    vector[5] b;
    real a;
	real log_sigma;
}
model{
    vector[N] mu;
    b ~ normal( 0 , 10 );
    a ~ normal( 0.5 , 1 );
	log_sigma ~ normal(0, 1);
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3 +
			b[4] * mass_std[i]^4 + b[5] * mass_std[i]^5;
    }
    brain_std ~ normal( mu , exp(log_sigma) );
}
generated quantities{
    vector[N] mu;
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3 +
			b[4] * mass_std[i]^4 + b[5] * mass_std[i]^5;
    }
}
";

# ╔═╡ 5a947711-0686-40a7-8968-8f1058143b6b
stan7_6 = "
data{
    int<lower=1> N;
    vector[N] brain_std;
    vector[N] mass_std;
}
parameters{
    vector[6] b;
    real a;
	real log_sigma;
}
model{
    vector[N] mu;
    b ~ normal( 0 , 10 );
    a ~ normal( 0.5 , 1 );
	log_sigma ~ normal(0, 1);
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3 +
			b[4] * mass_std[i]^4 + b[5] * mass_std[i]^5 + b[6] * mass_std[i]^6;
    }
    brain_std ~ normal( mu , 0.001 );
}
generated quantities{
    vector[N] mu;
    for ( i in 1:N ) {
        mu[i] = a + b[1] * mass_std[i] + b[2] * mass_std[i]^2 + b[3] * mass_std[i]^3 +
			b[4] * mass_std[i]^4 + b[5] * mass_std[i]^5+ b[6] * mass_std[i]^6;
    }
}
";

# ╔═╡ e2720ff8-2eee-4f55-9222-7c2baa95227c
models = [stan7_1, stan7_2, stan7_3, stan7_4, stan7_5, stan7_6];

# ╔═╡ fe051862-5fd3-4513-9f8a-ec0147872d16
let
	f = Figure(; size=default_figure_resolution)
	r2array = Float64[]
	for k in 1:6
		l = k > 3 ? 2 : 1
		m = k > 3 ? k - 3 : k
		N = size(df, 1)
		data = (N = N, brain_std = df.brain_s, mass_std = df.mass_s)
		init = (a = 0.0, bA = ones(k), sigma = 2)
		mu_range = LinRange(minimum(df.mass_s), maximum(df.mass_s), 100)

		q7, m7, o7 = stan_quap("m", models[k]; data, init)
		ndf1 = read_samples(m7, :nesteddataframe)
		dft = DataFrame(m=df.mass_s, b = mean(ndf1.mu), u=mean(ndf1.mu) .+ std(ndf1.mu), 
			l=mean(ndf1.mu) .- std(ndf1.mu))
		sort!(dft, :m)
		
		mus = mean(ndf1.mu)
		r = mus - df.brain_s
		append!(r2array, 1 - var(r; corrected=false) / var(df.brain_s; corrected=false))
		
		ax = Axis(f[l, m]; xlabel="Body mass [kg]", ylabel="Brain volume [cc]", 
			title="R2 = $(round(r2array[end]; digits=2)), k=$k")
		#ylims!(-0.1, 1.4)
		Makie.scatter!(df.mass_s, df.brain_s)
		Makie.lines!(dft.m, dft.b)
		Makie.band!(dft.m, dft.l, dft.u; color=(:grey, 0.3))
		
	end
	f
end

# ╔═╡ 7fb60422-314e-4da7-974e-a5de6d4ae704
let
	k = 1
	N = size(df, 1)
	data = (N = N, brain_std = df.brain_s, mass_std = df.mass_s)
	init = (a = 0.0, bA = ones(k), sigma = 2)
	q7, m7, o7 = stan_quap("m", models[k]; data, init)
	global mu_range3 = LinRange(minimum(df.mass_s), maximum(df.mass_s), 5)
	global mu_rv3 = [mu_range3 .^ i for i in 1:k]
	global mu_rvc3 = combinedims(mu_rv3)
	global ndf3 = read_samples(m7, :nesteddataframe)
	global res3 = link(ndf3, (r, x) -> r.a .+ dot(r.b, mu_rvc3[x, :]), 1:length(mu_range3))
	global res3a = combinedims(res3)
	global m3, l3, u3 = estimparam(res3a)
end

# ╔═╡ 092f28f8-4129-475e-864e-a1199d6bec1b
ndf3

# ╔═╡ f82e1455-418e-4682-8d50-3ac6368d4443
[minimum(mu_range3), maximum(mu_range3)]

# ╔═╡ 99367c43-45cb-4e05-86c6-a4dd2e1c2fea
mu_rvc3

# ╔═╡ 7c7de2c5-c2e9-42ae-a696-52b71a44d170
ndf3.a[1] .+ ndf3.b[1] .* mu_range3[1]

# ╔═╡ 32aeb197-6531-43d0-9f36-97d61d9fc7e3
ndf3.a[2] .+ ndf3.b[2] .* mu_range3[1]

# ╔═╡ ca611848-05e0-4fa2-9998-38426be3d610
ndf3.a[1] .+ ndf3.b[1] .* mu_range3[2]

# ╔═╡ 716d6970-a1a8-49a6-9b9c-e1cc0185dde0
res3

# ╔═╡ d1b7171b-e425-4c92-9019-ea2723fd1771
stan7_7 = "
data {
	int < lower = 1 > N; 			// Sample size
	int < lower = 1 > K;			// Degree of polynomial
	int < lower = 1 > L;			// Number of predicted brain values
	vector[N] brain; 				// Outcome
	matrix[N, K] mass; 				// Predictor
	matrix[L, K] mass_steps;        // Predictor steps
}
parameters {
 real a;                        // Intercept
 vector[K] b;                  // K slope(s)
 real log_sigma;    	// Error SD
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

# ╔═╡ 0868c3e7-8e75-4622-a4cf-d6c52e06d8dd
let
	k = 1
	N = size(df, 1)
	global mass4 = zeros(N, k)
	for j in 1:N
		mass4[j, :] = [df.mass_s[j]^i for i in 1:k]
	end
	global mass_steps = LinRange(minimum(df.mass_s), maximum(df.mass_s), 5)
	l = length(mass_steps)
	global mass_steps4 = zeros(l, k)
	for j in 1:l
		mass_steps4[j, :] = [mass_steps[j]^i for i in 1:k]
	end
	data = (N = N, K=k, L=l, brain = df.brain_s, mass = mass4, mass_steps=mass_steps4)
	sm = SampleModel("m7_7", stan7_7)
	rc = stan_sample(sm; data)
	global ndf4 = read_samples(sm, :nesteddataframe)
	
	global mu_rv4 = [mass_steps .^ i for i in 1:k]
	global mu_rvc4 = combinedims(mu_rv4)
	global res4 = link(ndf4, (r, x) -> r.a .+ dot(r.b, mu_rvc4[x, :]), 1:length(mass_steps))
	global res4a = combinedims(res4)
	global m4, l4, u4 = estimparam(res4a)
	ndf4
end

# ╔═╡ f795ae37-b487-4d09-8bad-63a68b416e0d
ndf4

# ╔═╡ 6708bb6e-1e1e-489b-bcdf-c03250cdb482
mean(ndf4.mu)

# ╔═╡ 11c46077-9650-4fbc-aac3-b9a7401cb386
mean(ndf4.brain_pred)

# ╔═╡ 9edf97eb-f8a1-4a2d-bc4c-9e24573138f5
mass4

# ╔═╡ 52f933a0-5500-4457-9518-1c51563b9b99
mu_rvc4

# ╔═╡ b672a428-62b8-42ac-81fc-6fdb4c7a1b3f
ndf4.a[1]

# ╔═╡ 48c8ac21-dfe0-4fbd-8f5f-c1952e74bd23
ndf4.b[1, :]

# ╔═╡ 5c322982-3ef3-49fe-a9d1-a4db3c7048f6
mu_rvc4[1, :]

# ╔═╡ 30e266e3-2767-4332-ba93-9d541d62edf0
ndf4.a[1] .+ dot(ndf4.b[1], mu_rvc4[1, :])

# ╔═╡ aa6b305d-12c0-4ca1-8e7f-787e11f19bf4
ndf4.a[2] .+ dot(ndf4.b[2], mu_rvc4[1, :])

# ╔═╡ 4e19a129-2a50-490f-8c0a-d20e3aa0e84f
ndf4.a[1] .+ dot(ndf4.b[1], mu_rvc4[2, :])

# ╔═╡ 3f8f214b-6b57-4c56-a6af-333744821fe7
res4

# ╔═╡ 4c1ad384-f6e7-4c56-b157-bf5c47b1c6d9
let
	f = Figure(; size=default_figure_resolution)
	r2array = Float64[]
	for k in 1:6
		lr = k > 3 ? 2 : 1
		mc = k > 3 ? k - 3 : k
		N = size(df, 1)
		mass5 = zeros(N, k)
		for j in 1:N
			mass5[j, :] = [df.mass_s[j]^i for i in 1:k]
		end
		mass_steps = LinRange(minimum(df.mass_s), maximum(df.mass_s), 100)
		l = length(mass_steps)
		mass_steps5 = zeros(l, k)
		for j in 1:l
			mass_steps5[j, :] = [mass_steps[j]^i for i in 1:k]
		end
		data = (N = N, K=k, L=l, brain = df.brain_s, mass = mass5, mass_steps=mass_steps5)
		sm = SampleModel("m7_7", stan7_7)
		rc = stan_sample(sm; data)
		global ndf5 = read_samples(sm, :nesteddataframe)
		dft = DataFrame(m=df.mass_s, b = mean(ndf5.mu), u=mean(ndf5.mu) .+ std(ndf5.mu), 
			l=mean(ndf5.mu) .- std(ndf5.mu))
		dft = DataFrame(m=mass_steps, b = mean(ndf5.brain_pred), 
			u=mean(ndf5.brain_pred) .+ std(ndf5.brain_pred), 
			l=mean(ndf5.brain_pred) .- std(ndf5.brain_pred))
		sort!(dft, :m)
		
		mus = mean(ndf5.mu)
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
# ╠═4ca22108-83f5-4edc-815d-101c78639824
# ╟─17889d4d-04f9-4b3f-8d93-8524e1bb15f7
# ╠═bdf0c993-5eee-480d-928b-40212180a6d0
# ╠═9e9cdf8b-51ee-4544-854e-a74c7a4cd6bf
# ╠═a287a5dd-dc99-4877-9e24-47ae28138670
# ╠═cb5c5048-9d7a-4a4f-9ad0-01dd633a439e
# ╠═c25cf2bd-bef7-4edb-a139-3ff692c7735e
# ╠═ceca761b-d23b-429f-bb94-1b79173962e9
# ╠═7f4ffc11-6105-4265-844e-29982f6befc6
# ╠═c11b3bec-2e84-4bb2-97be-3f6b90f4c021
# ╠═6f7da2ec-fd9d-4479-874f-39e3129064d6
# ╠═84b08ed3-f571-41d0-aaee-6b32273826e2
# ╠═146bba10-8e10-46ca-b111-e7321cd97760
# ╠═5e0a9743-6318-439b-bbf1-f415a681854f
# ╠═0f844e0d-8dbb-40fe-b511-df22b455d376
# ╠═ff144f4b-d7b7-4d4a-ba6f-d3f5e30b18a8
# ╠═82a0bc7d-3252-4176-ba8b-669f0caa6712
# ╠═5a947711-0686-40a7-8968-8f1058143b6b
# ╠═e2720ff8-2eee-4f55-9222-7c2baa95227c
# ╠═fe051862-5fd3-4513-9f8a-ec0147872d16
# ╠═7fb60422-314e-4da7-974e-a5de6d4ae704
# ╠═092f28f8-4129-475e-864e-a1199d6bec1b
# ╠═f82e1455-418e-4682-8d50-3ac6368d4443
# ╠═99367c43-45cb-4e05-86c6-a4dd2e1c2fea
# ╠═7c7de2c5-c2e9-42ae-a696-52b71a44d170
# ╠═32aeb197-6531-43d0-9f36-97d61d9fc7e3
# ╠═ca611848-05e0-4fa2-9998-38426be3d610
# ╠═716d6970-a1a8-49a6-9b9c-e1cc0185dde0
# ╠═d1b7171b-e425-4c92-9019-ea2723fd1771
# ╠═0868c3e7-8e75-4622-a4cf-d6c52e06d8dd
# ╠═f795ae37-b487-4d09-8bad-63a68b416e0d
# ╠═6708bb6e-1e1e-489b-bcdf-c03250cdb482
# ╠═11c46077-9650-4fbc-aac3-b9a7401cb386
# ╠═9edf97eb-f8a1-4a2d-bc4c-9e24573138f5
# ╠═52f933a0-5500-4457-9518-1c51563b9b99
# ╠═b672a428-62b8-42ac-81fc-6fdb4c7a1b3f
# ╠═48c8ac21-dfe0-4fbd-8f5f-c1952e74bd23
# ╠═5c322982-3ef3-49fe-a9d1-a4db3c7048f6
# ╠═30e266e3-2767-4332-ba93-9d541d62edf0
# ╠═aa6b305d-12c0-4ca1-8e7f-787e11f19bf4
# ╠═4e19a129-2a50-490f-8c0a-d20e3aa0e84f
# ╠═3f8f214b-6b57-4c56-a6af-333744821fe7
# ╠═4c1ad384-f6e7-4c56-b157-bf5c47b1c6d9
