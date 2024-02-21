data {
    int<lower=1> N_data; // number of data points

    // x-values of the data is uncertain.
    vector[N_data] x;
    vector[N_data] sigma_x;

    // y-values of the data is uncertain.
    vector[N_data] y;
    vector[N_data] sigma_y;

}

parameters {
    // Mean value of the x-guassian.
    real<lower=-2.0, upper=2.0> mu_x;

    // Mean value of the y-guassian.
    real<lower=-2.0, upper=2.0> mu_y;

}

model {
    for (i in 1:N_data) {
        x[i] ~ normal(
          mu_x,  
          sigma_x[i]
        );
	y[i] ~ normal(
          mu_y, 
          sigma_y[i]
        );
    }
}



