data {
  int<lower=0> N;
}
parameters {
  vector[N] theta;
}
model {
  theta ~ std_normal();
}
