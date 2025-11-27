data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x_star;
  real<lower=0> sigma_x_star;
}
parameters {
  real a, b, mu_x;
  real<lower=0> sigma, sigma_x;
}
transformed parameters {
  real inv_var_x  = inv_square(sigma_x);
  real inv_var_xs = inv_square(sigma_x_star);
  real tilde_v    = 1.0 / (inv_var_x + inv_var_xs);              // Var(x | x*)
  real<lower=0> sd_xstar = sqrt(square(sigma_x) + square(sigma_x_star));
  real<lower=0> sd_y_cond = sqrt(square(sigma) + square(b) * tilde_v);
  vector[N] tilde_mu = tilde_v * ( inv_var_x * rep_vector(mu_x, N)
                                 + inv_var_xs * x_star );
}
model {
  // hyperpriors (use whatever you prefer; half-normal/lognormal work fine)
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  mu_x ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);
  sigma_x ~ lognormal(0, 0.5);

  // marginalized likelihood
  x_star ~ normal(mu_x, sd_xstar);                 // p(x*)
  y ~ normal(a + b * tilde_mu, sd_y_cond);         // p(y | x*)
}
