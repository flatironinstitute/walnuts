data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N, P] x;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[P] beta;
}
model {
  beta ~ normal(0, 1);
  y ~ bernoulli_logit(x * beta);
}
