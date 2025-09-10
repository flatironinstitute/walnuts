data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x_star;
  real<lower=0> sigma_x_star;
}
parameters {
  real a, b, mu_x;
  real<lower=0> sigma, sigma_x;
  vector<offset=mu_x, multiplier=sigma_x>[N] x;
}
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  mu_x ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);
  sigma_x ~ lognormal(0, 0.5);
  
  x ~ normal(mu_x, sigma_x);
  y ~ normal(a + b * x, sigma);
  x_star ~ normal(x, sigma_x_star);
}
