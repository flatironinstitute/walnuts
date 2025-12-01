data {
  int<lower=0> N;
}
transformed data {
  vector<lower=0>[N] alpha = exp(linspaced_vector(N, log(1.0 / N), log(N)));
}
parameters {
  simplex[N] theta;	   
}
model {
  theta ~ dirichlet(alpha);
}
