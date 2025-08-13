data {
  int<lower=0> N;
}
transformed data {
  int<lower=0> M = N / 2;
  vector<lower=0>[M] sigma1;
  for (m in 1:M) {
    sigma1[m] = m;
  }
  vector<lower=0>[M] sigma2 = sigma1 + M;
}
parameters {
  vector[M] alpha1;
  vector[M] alpha2;
}
model {
  alpha1 ~ normal(0, sigma1);
  alpha2 ~ normal(0, sigma2);
}
