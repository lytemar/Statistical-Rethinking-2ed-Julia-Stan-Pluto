using StanSample, ParetoSmooth, NamedTupleTools
using StatisticalRethinking
import StatisticalRethinking: pk_plot
import ParetoSmooth: loo_compare

df = CSV.read(sr_datadir("WaffleDivorce.csv"), DataFrame);
scale!(df, [:Marriage, :MedianAgeMarriage, :Divorce])
data = (N=size(df, 1), D=df.Divorce_s, A=df.MedianAgeMarriage_s,
    M=df.Marriage_s)

stan5_1 = "
data {
    int < lower = 1 > N; // Sample size
    vector[N] D; // Outcome
    vector[N] A; // Predictor
}
parameters {
    real a; // Intercept
    real bA; // Slope (regression coefficients)
    real < lower = 0 > sigma;    // Error SD
}
transformed parameters {
    vector[N] mu;               // mu is a vector
    for (i in 1:N)
        mu[i] = a + bA * A[i];
}
model {
    a ~ normal(0, 0.2);         //Priors
    bA ~ normal(0, 0.5);
    sigma ~ exponential(1);
    D ~ normal(mu , sigma);     // Likelihood
}
generated quantities {
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(D[i] | mu[i], sigma);
}
";

stan5_2 = "
data {
    int N;
    vector[N] D;
    vector[N] M;
}
parameters {
    real a;
    real bM;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    for (i in 1:N)
        mu[i]= a + bM * M[i];

}
model {
    a ~ normal( 0 , 0.2 );
    bM ~ normal( 0 , 0.5 );
    sigma ~ exponential( 1 );
    D ~ normal( mu , sigma );
}
generated quantities {
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(D[i] | mu[i], sigma);
}
";

stan5_3 = "
data {
  int N;
  vector[N] D;
  vector[N] M;
  vector[N] A;
}
parameters {
  real a;
  real bA;
  real bM;
  real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    for (i in 1:N)
        mu[i] = a + bA * A[i] + bM * M[i];
}
model {
  a ~ normal( 0 , 0.2 );
  bA ~ normal( 0 , 0.5 );
  bM ~ normal( 0 , 0.5 );
  sigma ~ exponential( 1 );
  D ~ normal( mu , sigma );
}
generated quantities{
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(D[i] | mu[i], sigma);
}
";

function loo_compare(models::Vector{SampleModel}; 
    loglikelihood_name="log_lik", model_names=nothing, sort_models=true)

    if isnothing(model_names)
        mnames = [models[i].name for i in 1:length(models)]
    end

    nmodels = length(models)

    ka = Vector{KeyedArray}(undef, nmodels)
    ll = Vector{Array{Float64, 3}}(undef, nmodels)

    for i in 1:length(models)
        ka[i] = read_samples(models[i], :keyedarray)
        ll[i] = permutedims(Array(matrix(ka[i], loglikelihood_name)), [1, 3, 2])
    end

    loo_compare(ll; model_names=mnames, sort_models)
end

function waic_compare(models::Vector{SampleModel}; 
    loglikelihood_name="log_lik", model_names=nothing, sort_models=true)

    if isnothing(model_names)
        mnames = [models[i].name for i in 1:length(models)]
    end

    nmodels = length(models)

    ka = Vector{KeyedArray}(undef, nmodels)
    ll = Vector{Array{Float64, 3}}(undef, nmodels)
    llp = Vector{Array{Float64, 3}}(undef, nmodels)
    llpr = Vector{Array{Float64, 3}}(undef, nmodels)

    for i in 1:length(models)
        ka[i] = read_samples(models[i], :keyedarray)
        ll[i] = permutedims(Array(matrix(ka[i], loglikelihood_name)), [1, 3, 2])
        llp[i] = permutedims(ll[1],[ 1, 3, 2])
        if i == 1
            ndraws, nchains, nobs = size(llp[1])
        end
        llpr[i] = reshape(llp, ndraws*nchains, nobs)
    end

    compare(llpr, Val(:waic); mnames=mnames)
end

m5_1s = SampleModel("m5.1s", stan5_1)
rc5_1s = stan_sample(m5_1s; data)

m5_2s = SampleModel("m5.2s", stan5_2)
rc5_2s = stan_sample(m5_2s; data)

m5_3s = SampleModel("m5.3s", stan5_3)
rc5_3s = stan_sample(m5_3s; data)

if success(rc5_1s) && success(rc5_2s) && success(rc5_3s)

    p5_1s = read_samples(m5_1s, :particles)
    NamedTupleTools.select(p5_1s, (:a, :bA, :sigma)) |> display
    p5_2s = read_samples(m5_2s, :particles)
    NamedTupleTools.select(p5_2s, (:a, :bM, :sigma)) |> display
    p5_3s = read_samples(m5_3s, :particles)
    NamedTupleTools.select(p5_3s, (:a, :bA, :bM, :sigma)) |> display

    models = [m5_1s, m5_2s, m5_3s]
    loglikelihood_name = :log_lik
    loo_comparison = loo_compare(models)
    println()
    for (i, psis) in enumerate(loo_comparison.psis)
        psis |> display
        pk_plot(psis.pointwise(:pareto_k))
        savefig(joinpath(@__DIR__, "m5.$(i)s.png"))
    end
    println()
    loo_comparison |> display
end
#=
With SR/ulam():
```
       PSIS    SE dPSIS  dSE pPSIS weight
m5.1u 126.0 12.83   0.0   NA   3.7   0.67
m5.3u 127.4 12.75   1.4 0.75   4.7   0.33
m5.2u 139.5  9.95  13.6 9.33   3.0   0.00
```
=#