data {
    // number of series
    int<lower=1> N_series;

    // number of years or data points in a series
    int<lower=1> N_years;

    // Data Matrix (the entire dataset)
    matrix[N_series,N_years] data_points;

    //Year
    vector[N_years] year;

}

parameters {
    // standard deviation (homoscedastic)
    real<lower=0> sigma;

    // multinomial pi_k values
    simplex[3] pi_k; 
}

model {
    for (n in 1:N_series) {
        for (m in 1:N_years) {
            real log_ps[3];
            log_ps[1] = log(pi_k[1]) + normal_lpdf(data_points[n, m] | -year[m]/100, sigma);
            log_ps[2] = log(pi_k[2]) + normal_lpdf(data_points[n, m] | +year[m]/100, sigma);
            log_ps[3] = log(pi_k[3]) + normal_lpdf(data_points[n, m] | 0.0, sigma);
            target += log_sum_exp(log_ps);
        }
    }
}

generated quantities{
    real<lower=0> tmp1 ;
    real<lower=0> tmp2 ;
    real<lower=0> tmp3 ;
    vector[N_series] pi1 ;
    vector[N_series] pi2 ;
    vector[N_series] pi3 ;
    
    for (n in 1:N_series){
        tmp1 = 0;
        tmp2 = 0;
        tmp3 = 0;
        for (m in 1:N_years){
            tmp1 += log(pi_k[1]) + normal_lpdf(data_points[n, m] | -year[m]/100, sigma);
            tmp2 += log(pi_k[2]) + normal_lpdf(data_points[n, m] | +year[m]/100, sigma);
            tmp3 += log(pi_k[3]) + normal_lpdf(data_points[n, m] | 0.0, sigma);
                }
        pi1[n] = tmp1;
        pi2[n] = tmp2;
        pi3[n] = tmp3;
            }
        }
