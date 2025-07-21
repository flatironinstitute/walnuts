data {
  int<lower=1> D;
  vector<lower=0>[D] scales;
}
parameters {
  vector[D] x;
}
model {
  x ~ normal(0, scales);
}
