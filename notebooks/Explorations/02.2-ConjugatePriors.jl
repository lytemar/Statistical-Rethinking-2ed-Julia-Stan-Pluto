### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 772eeae8-c5f8-11ee-2f4c-390f6136f2d1
begin
	using Distributions
	using ConjugatePriors
	using Test

	import ConjugatePriors: posterior, posterior_rand, posterior_mode, posterior_randmodel, fit_map

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

# ╔═╡ b51107ee-e7ab-476f-9212-536f9ade151c
n = 100

# ╔═╡ 969a9a12-62d4-4a0c-978e-b7360b357d08
w = rand(100);

# ╔═╡ eb793b07-919d-4c8b-b48a-c8c4481ee2fe
mean(w)

# ╔═╡ 6cb52843-1c7f-4dd3-a697-413dacdba245
pri = Beta(1.0, 2.0)

# ╔═╡ 5e82431b-7b89-4870-9302-e044cda9da23
x = rand(Bernoulli(0.3), n);

# ╔═╡ 2ff032b5-7a20-4671-8abf-4f091905e1b9
sum(x)

# ╔═╡ dd506774-3ef0-4ae7-8c62-663bc1376948
length(x) - sum(x)

# ╔═╡ e809a976-2cb6-4021-8536-a0a345589661
p1 = posterior(Beta(1.0, 2.0), Bernoulli, x)

# ╔═╡ b1afa30c-7c1c-4de0-aa36-87615acadf59
isa(p1, Beta)

# ╔═╡ 759ffa31-cb49-446d-9dcc-4e91ee0537a1
p1.α ≈ pri.α + sum(x)

# ╔═╡ f04a92ef-4686-4d5b-9b3b-760afc9a9f29
p1.β ≈ pri.β + (n - sum(x))

# ╔═╡ 07cf6d26-41c9-4efd-af25-0e9bd0719b28
f1 = fit_map(Beta(1.0, 2.0), Bernoulli, x)

# ╔═╡ e7933b02-116a-4203-a306-690dd5f689d0
isa(f1, Bernoulli)

# ╔═╡ 42d59940-cab7-4896-9db7-4b9058d21434
succprob(f1) ≈ mode(p1)

# ╔═╡ 3acd4d17-a330-42d7-99f1-ed4d5aea3bb3
p2 = posterior(pri, Bernoulli, x, w)

# ╔═╡ 90f6586d-b4c2-4913-a02e-9e2080afa0e9
isa(p2, Beta)

# ╔═╡ 2afc8866-01d3-4652-b33c-389a9ad6df29
p2.α ≈ pri.α + sum(x .* w)

# ╔═╡ 88b3dde4-f41b-4c4a-a723-e9ee4866cf68
p2.β ≈ pri.β + (sum(w) - sum(x .* w))

# ╔═╡ 1659fa9d-a034-4b7d-90cf-b649bb0d5a7f
f2 = fit_map(pri, Bernoulli, x, w)

# ╔═╡ 15496fde-c77e-4d1d-8b4f-e71d3afda767
isa(f2, Bernoulli)

# ╔═╡ 2d9bdc69-803d-479f-952f-fef0e5b2d5a4
succprob(f2) ≈ mode(p2)

# ╔═╡ a68e325a-4927-497b-8979-89ba5fe7cf15
x3 = rand(Binomial(9, 0.63), n)

# ╔═╡ a586660c-8976-4d60-95c4-9154d1efbf78
p3 = posterior(pri, Binomial, (9, x3))

# ╔═╡ c771e440-cb84-4791-a058-e8246571d75e
isa(p3, Beta)

# ╔═╡ 92584801-b22e-4179-9c30-61c81df4e65e
p3.α ≈ pri.α + sum(x3)

# ╔═╡ 27b07f0b-54aa-4eec-b0b4-9747ccc3d9a2
p3.β ≈ pri.β + (9n - sum(x3))

# ╔═╡ 254ba3eb-9200-40e8-aa9d-4221dda81fb8
f3 = fit_map(pri, Binomial, (9, x3))

# ╔═╡ 47dcdd9c-bc66-499b-bda4-a3664b8337d7
isa(f3, Binomial)

# ╔═╡ 15cb8e8d-957f-4a8a-b6c9-5f6444dcd313
ntrials(f3) == 9

# ╔═╡ aeaff6a5-7edb-4f93-8bdf-0f452bd083d1
succprob(f3) ≈ mode(p3)

# ╔═╡ 46791b4d-d118-40b8-9000-171c2fc9d4c5
mode(p3)

# ╔═╡ Cell order:
# ╟─0df948dd-785b-4807-b995-edc3405b1e9e
# ╠═379027c4-ff5f-4daa-890c-10763740aa28
# ╠═fde4fcee-98ba-4b08-975b-3c1aa55814ce
# ╠═772eeae8-c5f8-11ee-2f4c-390f6136f2d1
# ╠═b51107ee-e7ab-476f-9212-536f9ade151c
# ╠═969a9a12-62d4-4a0c-978e-b7360b357d08
# ╠═eb793b07-919d-4c8b-b48a-c8c4481ee2fe
# ╠═6cb52843-1c7f-4dd3-a697-413dacdba245
# ╠═5e82431b-7b89-4870-9302-e044cda9da23
# ╠═2ff032b5-7a20-4671-8abf-4f091905e1b9
# ╠═dd506774-3ef0-4ae7-8c62-663bc1376948
# ╠═e809a976-2cb6-4021-8536-a0a345589661
# ╠═b1afa30c-7c1c-4de0-aa36-87615acadf59
# ╠═759ffa31-cb49-446d-9dcc-4e91ee0537a1
# ╠═f04a92ef-4686-4d5b-9b3b-760afc9a9f29
# ╠═07cf6d26-41c9-4efd-af25-0e9bd0719b28
# ╠═e7933b02-116a-4203-a306-690dd5f689d0
# ╠═42d59940-cab7-4896-9db7-4b9058d21434
# ╠═3acd4d17-a330-42d7-99f1-ed4d5aea3bb3
# ╠═90f6586d-b4c2-4913-a02e-9e2080afa0e9
# ╠═2afc8866-01d3-4652-b33c-389a9ad6df29
# ╠═88b3dde4-f41b-4c4a-a723-e9ee4866cf68
# ╠═1659fa9d-a034-4b7d-90cf-b649bb0d5a7f
# ╠═15496fde-c77e-4d1d-8b4f-e71d3afda767
# ╠═2d9bdc69-803d-479f-952f-fef0e5b2d5a4
# ╠═a68e325a-4927-497b-8979-89ba5fe7cf15
# ╠═a586660c-8976-4d60-95c4-9154d1efbf78
# ╠═c771e440-cb84-4791-a058-e8246571d75e
# ╠═92584801-b22e-4179-9c30-61c81df4e65e
# ╠═27b07f0b-54aa-4eec-b0b4-9747ccc3d9a2
# ╠═254ba3eb-9200-40e8-aa9d-4221dda81fb8
# ╠═47dcdd9c-bc66-499b-bda4-a3664b8337d7
# ╠═15cb8e8d-957f-4a8a-b6c9-5f6444dcd313
# ╠═aeaff6a5-7edb-4f93-8bdf-0f452bd083d1
# ╠═46791b4d-d118-40b8-9000-171c2fc9d4c5
