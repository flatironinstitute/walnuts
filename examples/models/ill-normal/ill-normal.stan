data {
  int<lower=0> N;
}
transformed data {
  vector<lower=0>[N] sigma;
  for (n in 1:N) {
    sigma[n] = n;
  }
}
parameters {
  vector[N] alpha;
}
model {
  alpha ~ normal(0, sigma);
}
