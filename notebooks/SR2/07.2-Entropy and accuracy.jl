### A Pluto.jl notebook ###
# v0.19.40

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
	
	# Graphics related
	using CairoMakie

	# Causal inference support
	using GraphViz

	# Stan specific
	using ParetoSmoothedImportanceSampling
	using SplitApplyCombine
	using StanSample
	
	# Project support libraries
	using StatisticalRethinking: sr_datadir, create_observation_matrix, lppd, sim_train_test
	using RegressionAndOtherStories
end

# ╔═╡ 969d4bb6-0a0b-4540-b125-90be7b5779a7
md" ## 7.2 - Entropy and accuracy."

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

# ╔═╡ fa165873-ceb3-4726-8652-847792d5b2b7
md" ### Julia code snippet 7.12"

# ╔═╡ 48b01fc0-58f0-11eb-10e1-8969cfc5022f
begin
	p = [0.3, 0.7]
	q = [0.25, 0.75]
	earth = [0.7, 0.3]
	mars = [0.01, 0.99]
end;

# ╔═╡ 3e9cb689-22d3-450f-812a-6f5c44867684
begin
	# Entropy
	H(p) = - sum(p .* log.(p))

	# Cross entropy
	H(p, q) = - sum(p .* log.(q))
end

# ╔═╡ 65e714ae-58f0-11eb-0f86-bb6dad500c12
H(p)

# ╔═╡ 90e27340-58f2-11eb-367e-cb2aae23b616
H([0.01, 0.99])

# ╔═╡ c0ee72a4-6668-11eb-1078-a5f09dfbecff
H([0.7, 0.15, 0.15])

# ╔═╡ 36fe9ea1-cb9d-455d-bc96-b775fd08182f
H(p, q)

# ╔═╡ 514bd3c4-58f3-11eb-27eb-055303c639b1
md" ##### Kullback-Leibler divergence."

# ╔═╡ 18a6c58e-0adc-490a-b568-d3c525d335aa
D(p, q) = sum(p .* log.(p ./ q))

# ╔═╡ 69d6127c-58f0-11eb-29c9-0ba1d12c61c2
D(p, q)

# ╔═╡ b6ffed52-58f0-11eb-1ab2-7902bd8a2b52
begin
	qrange = 0.001:0.01:1.0
	res = Float64[]
	for qstep in qrange
		qs = [qstep, 1-qstep]
		append!(res, [D(p, qs)])
	end
	f = Figure(; size=default_figure_resolution)
	ax = Axis(f[1, 1]; xlabel="q[1]", ylabel="Divergence q from p")
	scatter!(qrange, res)
	vlines!([0.3]; color=:darkred, linestyle=:dash)
	f
end

# ╔═╡ 564b0ea4-f1f4-4af9-b7a3-62a39abded64
D(p, q)

# ╔═╡ ad48d933-40b8-4d93-9bc0-f56c7f9e2f8c
H(p, q) - H(p)

# ╔═╡ 627cfaaf-6849-4896-8765-232bee2bf985
md" ##### Divergence from earth -> mars."

# ╔═╡ 0a66d755-6636-4e2b-a8e5-52ee5dc4516b
md"
!!! note
    Reverse arguments?
"

# ╔═╡ b90d053b-df78-4d42-b107-0d0c4368046c
D(mars, earth)

# ╔═╡ 4ea0171d-64d9-4581-b806-33d39528827a
md" ##### Divergence from mars -> earth."

# ╔═╡ 00aa94a0-34d0-42d4-80d7-c80399795335
D(earth, mars)

# ╔═╡ 32c0ac88-74d1-494e-9b45-34d8154029f4
md" #### Julia code snippet 7.01"

# ╔═╡ 9161280a-fee4-462e-971a-db31b0e30eaa
begin
	sppnames = [:afarensis, :africanus, :hapilis, :boisei, :rudolfensis, :ergaster, :sapiens]
	brainvol = [438, 452, 612, 521, 752, 871, 1350]
	masskg = [37, 35.5, 34.5, 41.5, 55.5, 61, 53.5]
	df = DataFrame(species = sppnames, brain = brainvol, mass = masskg)
	df[!, :brain_s] = df.brain / maximum(df.brain) 
	scale_df_cols!(df, :mass)
	df
end

# ╔═╡ 8b2cd951-6f07-4b4b-ae61-837ba9d56481
md" #### Julia code snippet 7.02"

# ╔═╡ 3e6fe644-81f0-493e-8771-b7edcabd8c48
let
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

# ╔═╡ e3e3dae8-917e-4f6f-b57f-4d1fd9f0d477
sig = 0.1

# ╔═╡ 9724b17a-bbcc-418a-a1fa-16a85ee94133
stan7_2 = "
data {
	 int < lower = 1 > N; 			// Sample size
	 int < lower = 1 > K;			// Degree of polynomial
	 vector[N] brain; 				// Outcome
	 matrix[N, K] mass; 			// Predictor
}

parameters {
	real a;                        // Intercept
	vector[K] b;                  // K slope(s)
	real log_sigma;
}

transformed parameters {
    vector[N] mu;
    mu = a + mass * b;
}

model {
	a ~ normal(0.5, 1);        
	b ~ normal(0, 10);
	brain ~ normal(mu , $(sig));
}
generated quantities {
	vector[N] log_lik;
	real sigma;
	for (i in 1:N)
		log_lik[i] = normal_lpdf(brain[i] | mu[i], $(sig));
	sigma = $(sig);
}
";

# ╔═╡ 385fef14-443a-4842-a5b0-2e92b63a232e
sig6 = 0.001

# ╔═╡ f713870c-62ae-42d7-85dd-8e064bd12e57
stan7_6 = "
data {
	 int < lower = 1 > N; 			// Sample size
	 int < lower = 1 > K;			// Degree of polynomial
	 vector[N] brain; 				// Outcome
	 matrix[N, K] mass; 			// Predictor
}

parameters {
	real a;                        // Intercept
	vector[K] b;                  // K slope(s)
	real log_sigma;
}

transformed parameters {
    vector[N] mu;
    mu = a + mass * b;
}

model {
	a ~ normal(0.5, 1);        
	b ~ normal(0, 10);
	brain ~ normal(mu , $(sig6));
}
generated quantities {
	vector[N] log_lik;
	real sigma;
	for (i in 1:N)
		log_lik[i] = normal_lpdf(brain[i] | mu[i], $(sig6));
	sigma = $(sig6);
}
";

# ╔═╡ 0a6fdc92-996f-4191-8dcc-9fb67952dde4
begin
	loo = Vector{Float64}(undef, 6)
	loos = Vector{Vector{Float64}}(undef, 6)
	pk = Vector{Vector{Float64}}(undef, 6)
	deviance = Vector{Float64}(undef, 6)
end;

# ╔═╡ 7424dcc6-426e-4440-960f-57f05a348337
let
	global lppd_res = Matrix{Float64}(undef, 6, 7)

	for K in 1:6
		N = size(df, 1)
		mass = create_observation_matrix(df.mass_s, K)
		data = (N = N, K = K, brain = df.brain_s, mass = mass)
		
		# `sigma` should really be `exp(log_sigma)`!
		
		if K < 6
			m7_2s = SampleModel("m7.2s", stan7_2)
		else
			m7_2s = SampleModel("m7.2s", stan7_6)
		end
		rc7_2s = stan_sample(m7_2s; data=data)

		if success(rc7_2s)
			global ndf7_2s = read_samples(m7_2s, :nesteddataframe)
		end

		global log_lik = hcat(ndf7_2s.log_lik...)'
		lppd_res[K, :] = lppd(log_lik)
		loo[K], loos[K], pk[K] = psisloo(log_lik)
		K <3 && println(size(ndf7_2s), size(mass), size(df.brain_s), size(ndf7_2s.sigma))
		lp = logprob(ndf7_2s, mass, df.brain_s, ndf7_2s.sigma, K)
		deviance[K] = -2sum(lppd(lp))
	end
	deviance
end

# ╔═╡ 96ea9398-8bc0-456a-822c-2e7ab712bf9d
-deviance/2

# ╔═╡ e821c962-03d2-4de5-a41a-cf05ac969981
sum(lppd_res', dims=1)

# ╔═╡ a47498a4-6ab7-431d-84f8-637c7aebe75f
lppd_res[1, :]

# ╔═╡ ee9e0115-1ea6-4032-bd5f-70942c370da7
md"
!!! note

	Take a look at these runs using `psisloo()` from PSIS.jl. This is explained later in chapter 7.
"

# ╔═╡ 6466752e-754d-4ea3-b6d0-8b9a4a7936d8
loo

# ╔═╡ f3ccf15a-3e63-4ced-826f-a2b77f9ab686
sum.(loos)

# ╔═╡ 024a4c39-5ffd-4e61-844f-a5c0d02322b4
pk_qualify(pk[1])

# ╔═╡ 6eb2959d-435b-45fd-bd80-b4e75434e50e
let
	f = Figure(; size=default_figure_resolution)
	r = 1
	c = 1
	for i in 1:6
		if i > 3
			r = 2
			c = i - 3
		else
			r = 1
			c = i
		end
		ax = Axis(f[r, c]; title="PSIS diagnostic plot for K = $i", xlabel="Observed datapoint", ylabel="pareto shape k")
		ylims!(0, 1.2)
		scatter!(pk[i])
		hlines!([0.5, 0.7, 1]; color=[:red, :green, :purple])
	end
	f
end

# ╔═╡ cc9c3a9c-8677-497a-954d-0a2642a5e3d9
stan7_9 = "
data {
    int<lower=1> K;
    int<lower=0> N;
    matrix[N, K] x;
    vector[N] y;
}
parameters {
    real a;
    vector[K] b;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a + x * b;
}
model {
	a ~ normal(0, 100);
	b ~ normal(0, 10);
	sigma ~ exponential(1);
    y ~ normal(mu, sigma);          // observed model
}
generated quantities {
	vector[N] log_lik;
	for (i in 1:N)
		log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
}
";

# ╔═╡ dfddb1a4-dea5-49e8-8414-257b3def29d5
begin
	m7_9s = SampleModel("m7.9s", stan7_9)
end;

# ╔═╡ ea8ff089-0b55-46fd-8a9d-255c2bdd051a
begin
	Ns = [20, 100] 				# Number of observations
	rho = [0.15, -0.4]			# Covariance between x1 and x2
	L = 50 					    # Number of simulations
	K = 5 						# Number of slopes
	dev_is = zeros(L, K)
	dev_os = zeros(L, K)
end;

# ╔═╡ 3dfcade5-fdc5-4ff7-af58-6b79a23baebf
md"
!!! note
	This might take some time ...
"

# ╔═╡ b698508e-84b1-4f43-b68e-e09141117cb4
function logprob2(ndf::DataFrame, mass::Matrix, obs::Vector, sigma::Vector{Float64}, k)
    b = Matrix(hcat(ndf.b...)')
    mu = ndf.a .+ b * mass[:, 1:k]
    logpdf.(Normal.(mu , sigma),  obs')
end


# ╔═╡ d794c3d4-2410-4b4b-a63f-ce23fd18a676
let
	res = Vector{NamedTuple}(undef, length(Ns))
	for (ind, N) in enumerate(Ns)
		for i in 1:L
			for j = 1:K
				println("N = $(Ns[ind]), run = $i, no of b parms = $j")
				y, x_train, x_test = sim_train_test(;N, K, rho)
				data = (N = size(x_train, 1), K = size(x_train, 2),
					y = y, x = x_train,
					N_new = size(x_test, 1), x_new = x_test)
				rc7_9s = stan_sample(m7_9s; data)
				if success(rc7_9s)
					
					# use `logprob()`
					
					global ndf7_9s = read_samples(m7_9s, :nesteddataframe)
					println(names(ndf7_9s))
					i < 3 && println(size(ndf7_9s), size(Matrix(x_train')), size(y), size(ndf7_9s.sigma))
					lp_train = logprob2(ndf7_9s, Matrix(x_train'), y, ndf7_9s.sigma, j)
					dev_is[i, j] = -2sum(lppd(lp_train))
					lp_test = logprob2(ndf7_9s, Matrix(x_test'), y, ndf7_9s.sigma, j)
					dev_os[i, j] = -2sum(lppd(lp_test))

				end
			end
		end
		res[ind] = (
			mean_dev_is = mean(dev_is, dims=1),
			std_dev_is = std(dev_is, dims=1),
			mean_dev_os = mean(dev_os, dims=1),
			std_dev_os =std(dev_os, dims=1)
		)
	end
end;

# ╔═╡ bafa8585-2c95-4730-ad78-f338113dd624
ndf7_9s.a

# ╔═╡ 2a46db3b-52f7-4c2d-bb53-f1e824b17184
size(Matrix(hcat(ndf7_9s.b...)))

# ╔═╡ 4795f556-4a04-4d89-b3cb-7cc7ffacd7ef
begin
	fig = Vector{Plots.Plot{Plots.GRBackend}}(undef, 2)
	for i in 1:2
		xcoord = collect(1:K) .- 0.05
		ycoord = res[i].mean_dev_is[1,:]
		ylims = i == 1 ? (0,100) : (200, 350)
		fig[i] = scatter(xcoord, ycoord, xlab = "No of b parameters",
			ylab = "Deviance", ylims = ylims, leg=:bottomleft,
			lab = "In sample", markersize=2)
		for j in 1:K
			plot!([xcoord[j], xcoord[j]],
				[ycoord[j]-res[i].std_dev_is[j], ycoord[j]+res[i].std_dev_is[j]],
				lab=false, color=:darkblue)
		end
		title!("N = $(Ns[i]), L = $(L)")
		xcoord = collect(1:K) .+ 0.05
		ycoord = res[i].mean_dev_os[1,:]
		scatter!(xcoord, ycoord, lab = "Out of sample", markersize=2)
		for j in 1:K
			plot!([xcoord[j], xcoord[j]],
				[ycoord[j]-res[i].std_dev_os[j], ycoord[j]+res[i].std_dev_os[j]],
				lab=false, color=:red)
		end
		title!("N = $(Ns[i]), L = $(L)")
	end
	plot(fig..., layout=(1,2))
end

# ╔═╡ 97e955e4-b1d2-4f0d-a626-415d7f789088
res

# ╔═╡ 2a252f12-0c11-4ec7-b810-05ba95729221
md"
!!! note
	Compare waic and psis estimates to in-sample estimate
"

# ╔═╡ 877bc3c2-4ac9-4c73-9895-0d8d5a628ed9
begin
	devs = Vector{Float64}(undef, 2)
	y, x_train, x_test = sim_train_test(;N=20, K=5, rho)
	data = (N = size(x_train, 1), K = size(x_train, 2),
		y = y, x = x_train,
		N_new = size(x_test, 1), x_new = x_test)
	rc7_9s = stan_sample(m7_9s; data)
	if success(rc7_9s)

		# use `logprob()`

		post7_9s_df = read_samples(m7_9s, :dataframe)
		lp_train = logprob(post7_9s_df, x_train, y, 2)
		devs[1] = -2sum(lppd(lp_train))
		lp_test = logprob(post7_9s_df, x_test, y, 2)
		devs[2] = -2sum(lppd(lp_test))
	end
end;

# ╔═╡ d9fb7458-a491-4da1-8ad3-524a5971a6a3
devs

# ╔═╡ 00ceebf6-64f4-4121-8c85-fc45ae3b5ad3
if success(rc7_9s)
	ndf = read_samples(m7_9s, :nesteddataframe)
	ll = hcat(ndf.log_lik...)
	waic(ll; log_lik="log_lik")
end

# ╔═╡ a679d81e-c4f4-41e6-8e78-dcf46f392913
if success(rc7_9s)
	loo7_9s, loos7_9s, pk7_9s = psisloo(ll)
	-2loo7_9s
end

# ╔═╡ acc7b3af-2e02-4879-9033-61cb26cec136
# Use `generated quantities`

if success(rc7_9s)
	nt7_9s = read_samples(m7_9s, :namedtuple)
	-2sum(lppd(nt7_9s.log_lik'))
end

# ╔═╡ 434f8823-ce55-4849-b630-7f8d646ef57d
let
	f = Figure(; size=default_figure_resolution)
	r = 1
	c = 1
	for i in 1:6
		if i > 3
			r = 2
			c = i - 3
		else
			r = 1
			c = i
		end
		ax = Axis(f[r, c]; title="PSIS diagnostic plot for K = $i", xlabel="Observed datapoint", ylabel="pareto shape k")
		ylims!(0, 1.2)
		scatter!(pk[i])
		hlines!([0.5, 0.7, 1]; color=[:red, :green, :purple])
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
# ╟─fa165873-ceb3-4726-8652-847792d5b2b7
# ╠═48b01fc0-58f0-11eb-10e1-8969cfc5022f
# ╠═3e9cb689-22d3-450f-812a-6f5c44867684
# ╠═65e714ae-58f0-11eb-0f86-bb6dad500c12
# ╠═90e27340-58f2-11eb-367e-cb2aae23b616
# ╠═c0ee72a4-6668-11eb-1078-a5f09dfbecff
# ╠═36fe9ea1-cb9d-455d-bc96-b775fd08182f
# ╠═514bd3c4-58f3-11eb-27eb-055303c639b1
# ╠═18a6c58e-0adc-490a-b568-d3c525d335aa
# ╟─69d6127c-58f0-11eb-29c9-0ba1d12c61c2
# ╠═b6ffed52-58f0-11eb-1ab2-7902bd8a2b52
# ╠═564b0ea4-f1f4-4af9-b7a3-62a39abded64
# ╠═ad48d933-40b8-4d93-9bc0-f56c7f9e2f8c
# ╟─627cfaaf-6849-4896-8765-232bee2bf985
# ╟─0a66d755-6636-4e2b-a8e5-52ee5dc4516b
# ╠═b90d053b-df78-4d42-b107-0d0c4368046c
# ╟─4ea0171d-64d9-4581-b806-33d39528827a
# ╠═00aa94a0-34d0-42d4-80d7-c80399795335
# ╟─32c0ac88-74d1-494e-9b45-34d8154029f4
# ╠═9161280a-fee4-462e-971a-db31b0e30eaa
# ╟─8b2cd951-6f07-4b4b-ae61-837ba9d56481
# ╠═3e6fe644-81f0-493e-8771-b7edcabd8c48
# ╠═e3e3dae8-917e-4f6f-b57f-4d1fd9f0d477
# ╠═9724b17a-bbcc-418a-a1fa-16a85ee94133
# ╠═385fef14-443a-4842-a5b0-2e92b63a232e
# ╠═f713870c-62ae-42d7-85dd-8e064bd12e57
# ╠═0a6fdc92-996f-4191-8dcc-9fb67952dde4
# ╠═7424dcc6-426e-4440-960f-57f05a348337
# ╠═96ea9398-8bc0-456a-822c-2e7ab712bf9d
# ╠═e821c962-03d2-4de5-a41a-cf05ac969981
# ╠═a47498a4-6ab7-431d-84f8-637c7aebe75f
# ╟─ee9e0115-1ea6-4032-bd5f-70942c370da7
# ╠═6466752e-754d-4ea3-b6d0-8b9a4a7936d8
# ╠═f3ccf15a-3e63-4ced-826f-a2b77f9ab686
# ╠═024a4c39-5ffd-4e61-844f-a5c0d02322b4
# ╠═6eb2959d-435b-45fd-bd80-b4e75434e50e
# ╠═cc9c3a9c-8677-497a-954d-0a2642a5e3d9
# ╠═dfddb1a4-dea5-49e8-8414-257b3def29d5
# ╠═ea8ff089-0b55-46fd-8a9d-255c2bdd051a
# ╟─3dfcade5-fdc5-4ff7-af58-6b79a23baebf
# ╠═d794c3d4-2410-4b4b-a63f-ce23fd18a676
# ╠═b698508e-84b1-4f43-b68e-e09141117cb4
# ╠═bafa8585-2c95-4730-ad78-f338113dd624
# ╠═2a46db3b-52f7-4c2d-bb53-f1e824b17184
# ╠═4795f556-4a04-4d89-b3cb-7cc7ffacd7ef
# ╠═97e955e4-b1d2-4f0d-a626-415d7f789088
# ╠═2a252f12-0c11-4ec7-b810-05ba95729221
# ╠═877bc3c2-4ac9-4c73-9895-0d8d5a628ed9
# ╠═d9fb7458-a491-4da1-8ad3-524a5971a6a3
# ╠═00ceebf6-64f4-4121-8c85-fc45ae3b5ad3
# ╠═a679d81e-c4f4-41e6-8e78-dcf46f392913
# ╠═acc7b3af-2e02-4879-9033-61cb26cec136
# ╠═434f8823-ce55-4849-b630-7f8d646ef57d
