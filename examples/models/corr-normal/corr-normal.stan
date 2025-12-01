data {
  int<lower=0> N;
  real<lower=-1, upper=1> rho;
}
transformed data {
  vector[N] zero = zeros_vector(N);
  matrix[N, N] Sigma;
  for (m in 1:N) {
    for (n in 1:N) {
      Sigma[m, n] = rho^abs(m - n);
    }
  }
  matrix[N, N] L_Sigma = cholesky_decompose(Sigma);
}
parameters {
  vector[N] alpha;
}
model {
  alpha ~ multi_normal_cholesky(zero, L_Sigma);
}
