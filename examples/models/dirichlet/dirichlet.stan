data {
  int<lower=0> N;
}
transformed data {
	    //  vector<lower=0>[N] alpha = exp(linspaced_vector(N, log(1), log(1)));
}
parameters {
  simplex[N] theta;	   
}
model {
  // theta ~ dirichlet(alpha);
}