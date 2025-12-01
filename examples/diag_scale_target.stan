data {
  int<lower=1> D;
  vector<lower=0>[D] scales;
}
parameters {
  vector[D] theta;
}
model {
  theta ~ normal(0, scales);
}
