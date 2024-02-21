#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
from chainconsumer import ChainConsumer 
from cmdstanpy import CmdStanModel, install_cmdstan
import corner
from scipy.stats import norm


# # Quesiton 1

# In[2]:


#aquire the data 
data = []
with open('temperatures.txt', "r") as f:
    for line in f:
        # Skip over empty lines
        if not line.strip():
            continue
        # Split the line into an array of values
        values = line.split()
        # Convert the values to floats
        values = [float(v) for v in values]
        # Add the values to the data list
        data.append(np.array(values))
data = np.array(data).astype(float)
# Print the data
print(data[0])


# In[3]:


#plot the data 
fig, ax = plt.subplots(figsize=(16, 8))

#set up x-array with years
t = np.arange(1880, 2015)
for series in data:
    ax.plot(t, series)

ax.set_xlabel('Year')
ax.set_ylabel(" Change in mean temperature [deg]")
ax.set_title('Global warming data');


# # Question 2

# What we have in our data set is 1000 ficticious series (with some homoscedastic noise, constant variance) each containing 135 data points. Where the data in the series represents the change in the mean temperature every year from 1800 to 2014. Each series is drawn from one of three models. Some of the series were randomly selected to have a trend that would either increase/decrease the mean temperature by $\pm$1°C
# 
# It is important to note the the data points in the series are not independant of each other, and do depend on the previous value.
# I will also note that a 1°C change in a centery would correspond to a change in 0.01°C every year. 
# 
# Thefore a particular series will have its data drwan from one on the three models: 
# 
# 
# $$
# y_{i} \sim \mathcal{N}(y_{i-1} -0.01,\sigma)
# $$
# 
# $$
# y_{i} \sim \mathcal{N}(y_{i-1} +0.01,\sigma)
# $$
# 
# $$
# y_{i} \sim \mathcal{N}(y_{i-1},\sigma)
# $$
# 
# Where $i$ is the $i$ th data in the series and $\sigma$ is the homoscedatic noise that is the same for each dataset. 
# 
# To furthur tackle this problem we need to envoke a mixture of models which takes into account that a particular series could be drawn from 1 of the 3 models:  
# 
# $$
# y_{n, i, k} \sim \begin{cases}
#       \mathcal{N}(y_{n, i-1} -10^{-2},\sigma) & \textrm{if } k = 1 \\
#       \mathcal{N}(y_{n, i-1} + 10^{-2},\sigma) & \textrm{if } k = 2 \\
#       \mathcal{N}(y_{n, i-1},\sigma) & \textrm{if } k = 3 \, ,
#       \end{cases} 
# $$
# 
# $n$ specifies which series (of which there are 1000 of). The parameter $k$ which specifies the model that the series is drawn from. The introduction of this new parameter for ever seriers means that we have just introduces 1000 more individual parameters to fit for, luckily we can marginalise out $k$ parameters by introducing a prior. 
# 
# $$
# p(k) = \pi_k
# $$
# 
# The mixing proportions ($\pi_k$) will be bouned from 0 to 1, but there is an additional condition that the sum of the components must equal 1. To state this we say that : 
# 
# $$
# \pi_k \sim Multinomial(K)
# $$
# 
# Where $K$ is the total number of components in the model.
# 
# The overall posterior probability will be: 
# 
# $$
# p(\pi_k, \sigma | \textbf{y}, \mathcal{M}) \sim \sum_{n = 1}^{N}   \sum_{k = 1}^{K} log(\pi_ky_{n, i, k})
# $$
# 
# 
# 

# # Question 3 

# In[ ]:


##this is my stan file###

text = '''data {
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

//THis is my attempt to generate the component probabilities via stan 
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
            tmp1 += exp(log(pi_k[1]) + normal_lpdf(data_points[n, m] | -year[m]/100, sigma));
            tmp2 += exp(log(pi_k[2]) + normal_lpdf(data_points[n, m] | +year[m]/100, sigma));
            tmp3 += exp(log(pi_k[3]) + normal_lpdf(data_points[n, m] | 0.0, sigma));
                }
        pi1[n] = tmp1;
        pi2[n] = tmp2;
        pi3[n] = tmp3;
            }
        }
'''

file_path = "PS3.stan"

# Open the file in write mode and write the text
with open(file_path, "w") as file:
    file.write(text)



model = CmdStanModel(stan_file='PS3.stan')

N_years = 135
N_series = 1000
year = np.arange(0, N_years)

# Data.
data_dict = dict(
  N_years=N_years,
  N_series=N_series,
  data_points=data,
  year=year
)

fit = model.sample(data = data_dict )# runs with 4 chains


# In[5]:


pi_k = fit.pi_k.T
sigma = fit.sigma


# In[6]:


#plot the cahins and corner plots 

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("HMC chains for model parameters")
ax.plot(pi_k[0][2000:], c = 'r', label = '$\pi_1$')
ax.plot(pi_k[1][2000:], c = 'b',label = '$\pi_2$')
ax.plot(pi_k[2][2000:], c = 'g',label = '$\pi_3$')
ax.plot(sigma[2000:], c = 'purple',label = '$\sigma$')
ax.legend()
ax.set_xlabel('Chain itteration')
plt.show()

for i in range(3):
    c = ChainConsumer()
    c.add_chain(pi_k[i], parameters=[r"$\pi$"])
    fig = c.plotter.plot( figsize="column")

c = ChainConsumer()
c.add_chain(sigma, parameters=["$\sigma$"])
fig = c.plotter.plot( figsize="column")


# This result seem reasonable, to summarise the 'no trend' is drawn from most often ($\pi_1 \sim 0.41$) and by eye it seems like $\sum \pi_k = 1$. Which is good! The HMC chains have also converged! 
# 

# # Question 4
# Now I need to replot the series w.r.t the component membership probabilities.
# 
# To do this I will work from the example in the lecture 3 notes. 
# 
# $$
# p(q_i| \textbf{y},\sigma) \approx \frac{1}{M}\sum_{m=1}^{M} \frac{p(q_i)p(\textbf{y}|\textbf{\theta}^{(m)},\sigma,q_i)}{\sum_{k}p(q_i=k)p(\textbf{y}|\textbf{\theta}^{(m)},q_i=k)} \quad .
# $$
# 
# Where $M$ is the number of samples in the chain that I need to sum over. Technically we should have all the information that we need to compute this.
# 
# The numerator is the likelihood of the sampled parameters of a particular model multiplied by the probability that a series is drwan from that model. While the denomenator is the sum of the likelihoods multiplied by the probability of those models. We repeat this for e number of samples and divide by the samples to get an average. 
# 
# First lets define the log-likelihood of the data. 
# 

# In[7]:


def ln_likelihood(series, k, sigma):
    #the likelihood can be one of three models depending on the parameter k. 
    if k == 0:
        slope = -1/100
    elif k==1:
        slope = 1/100
    elif k==2:
        slope = 0
    else:
        print('k is out of range')

    # Start with the initial year:
    y_0 = series[0]

    #log likelihood
    ll = 0
    for t in range(135):
        y = series[t]
        # # log likelihood
        ll += -0.5*np.log(2 * np.pi * sigma**2 ) - 0.5*(y - (y_0 + slope))**2 /sigma**2 
        #update y_0
        y_0 = y
    return ll


# In[8]:


#this is me trying and failing 

#remove the burn in 
sigma_m = fit.sigma[2000:]
#get the values
pi_k_mu = np.array([np.mean(pi_k[0][2000:]),np.mean(pi_k[1][2000:]),np.mean(pi_k[2][2000:])])

#how many samples I will iterate over otherwise it will take too long. 
M = 100

#get weight values for the -ive model
i = 0
weight_0_list = []
for series in data: #repeat for each series 
#     print(i)
    weight_0 = []
    for m in range(M):
        num = np.log(pi_k_mu[0])+ln_likelihood(series, 0, sigma_m[m]) #calculate the numerator
        denom = 0
        for k in range(3):
            denom += np.log(pi_k_mu[k])+ln_likelihood(series, k, sigma_m[m]) #calculate the denomenator
        weight_0.append(np.exp(num - denom))# get the weight at each sample
    weight_0_list.append(np.sum(weight_0)/M) #get the average weight
#     print(weight_0_list[i])
    i += 1
print('calculated weights_0')

# repeat for +ive model
i = 0
weight_1_list = []
for series in data:
#     print(i)
    weight_1 = []
    for m in range(M):
        num = np.log(pi_k_mu[1])+ln_likelihood(series, 1, sigma_m[m])
        denom = 0
        for k in range(3):
            denom += np.log(pi_k_mu[k])+ln_likelihood(series, k, sigma_m[m])    
        weight_1.append(np.exp(num - denom))
    weight_1_list.append(np.sum(weight_1)/M)
#     print(weight_1_list[i])
    i += 1

print('calculated weights_1')

# and for 0 trend
i = 0
weight_2_list = []
for series in data:
#     print(i)
    weight_2 = []
    for m in range(M):
        num = np.log(pi_k_mu[2])+ln_likelihood(series, 2, sigma_m[m])
        denom = 0
        for k in range(3):
            denom += np.log(pi_k_mu[k])+ln_likelihood(series, k, sigma_m[m])
        weight_2.append(np.exp(num - denom))
    weight_2_list.append(np.sum(weight_2)/M)
#     print(weight_2_list[i])
    i += 1

print('calculated weights_2')        
combined_weights = np.array([weight_0_list,weight_1_list,weight_2_list])


# In[9]:


print(combined_weights.T[0])


# These values are obviously incorrect for probabilities but I tried my best and I need to work with them to continue so I will get the relative values and use that as my probability. 

# In[10]:


new_weights = []
for i in range(1000):
    total= np.sum(combined_weights.T[i])
    w0 = combined_weights.T[i][0]/total
    w1 = combined_weights.T[i][1]/total  
    w2 = combined_weights.T[i][2]/total    
    new_weights.append([w0,w1,w2])


# In[11]:


print(new_weights[0])


# In[12]:


#plot the data 
fig, ax = plt.subplots(figsize=(16, 8))

#set up x-array with years
t = np.arange(1880, 2015)
for i in range(1000):
    tmp1 = np.argmax(new_weights[i])
    tmp2 = np.max(new_weights[i])
#     print(r'alpha =', tmp2)|
    if tmp1 == 0:
        ax.plot(t, data[i], alpha =tmp2, c = 'b' )
    elif tmp1 == 1:
        ax.plot(t, data[i], alpha =tmp2, c = 'r' )
    elif tmp1 == 2:
        ax.plot(t, data[i], alpha =tmp2, c = 'black')        

        
legend_elements = [
    plt.Line2D([0], [0], color='b', lw=2, label='-ive trend'),
    plt.Line2D([0], [0], color='r', lw=2, label='+ive trend'),
    plt.Line2D([0], [0], color='black', lw=2, label='No trend')
]

ax.legend(handles=legend_elements)

ax.set_xlabel('Year')
ax.set_ylabel(" Change in mean temperature [deg]")
ax.set_title('Global warming data');


# # Question 5
# 
# 

# In[13]:


E_correct = 0
sigma_correct = 0
for i in range(1000):
    max_p = np.max(new_weights[i])
    E_correct += max_p
    sigma_correct += max_p - max_p**2
    
sigma_correct = np.sqrt(sigma_correct)
    
print(f'E_correct  = {E_correct} +\- {sigma_correct}')    


# # Question 6 

# In[14]:


p = 1 - norm.cdf(900, E_correct, sigma_correct)
print(p)


# As one might of expected from the mean and standard deviation of the expected correct values, that the probability of correctly identifying 900 or more series would be pretty much 0. 
# 
# Therefore I would not take the bet as I would see better returns in a money shredder. 
# 
