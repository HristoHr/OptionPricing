'''Solution for problem 2.2'''
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def price_and_greeks_monte_carlo():
    S = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    dt = tf.placeholder(tf.float32)
    T = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    B = tf.placeholder(tf.float32)
    Phi = tfp.distributions.Normal(0., 1.).cdf

    sigma2 = sigma ** 2
    mu = (r - (sigma2 / 2)) / sigma2

    rand_mtx = tf.placeholder(tf.float32)
    ST = S * tf.cumprod(tf.exp((r - sigma2 / 2) * dt + sigma * tf.sqrt(dt) * rand_mtx), axis=1)
    non_touch = tf.cast(tf.reduce_all(ST > B, axis=1), tf.float32)
    A = tf.reduce_mean(tf.cast(tf.maximum(ST[:, -1] - K, 0) * non_touch, tf.float32))

    y1 = tf.log((B ** 2) / (S * K)) / (sigma * tf.sqrt(T)) + (1 + mu) * sigma * tf.sqrt(T)
    y2 = y1 - sigma * tf.sqrt(T)
    C = S * (B / S) ** (2 * (mu + 1)) * Phi(y1) - K * tf.exp(-r * T) * (B / S) ** (2 * mu) * Phi(y2)

    value = (A - C)

    greeks = tf.gradients(value, [S, sigma, r, T])
    delta = greeks[0]
    vega = greeks[1] / 100
    rho = greeks[2] / 100
    theta = -greeks[3] / 365
    gamma = tf.gradients(delta, [S])
    result = value, delta, vega, rho, theta, gamma

    def pricer(S0, strike, time_to_expiry, volatility, risk_free_rate, barrier, simulations, observations):
        '''returns: predicted value, delta, vega, rho, theta, gamma'''
        np.random.seed(1213)
        random_variables = np.random.randn(simulations, observations)
        with tf.Session() as sess:
            timedelta = time_to_expiry / observations
            res = sess.run(result,
                           {
                               S: S0,
                               K: strike,
                               r: risk_free_rate,
                               sigma: volatility,
                               dt: timedelta,
                               T: time_to_expiry,
                               B: barrier,
                               rand_mtx: random_variables
                           })
            return res

    return pricer


K_ = 50.
B_ = 45.
S_ = 50.
T_ = 0.5
sigma_ = 0.2
r_ = 0.1

monte_carlo = price_and_greeks_monte_carlo()
print(monte_carlo(S_, K_, T_, sigma_, r_, B_, 10000, 1000))

'''
Sample Output:
(3.773489, 0.7109515, 0.07485082, 0.14944503, 0.0032995157, [-0.03429461])

Comments:

There are slight differences between the Monte Carlo and Black Scholes solutions.
However, this is expected.

The main problem with Monte Carlo calculations is the chance of a price to hit a barrier
in between monitoring times. The ways of improving the accuracy of the simulation are to 
increase the number of simulations and observations, to use Barrier shifting 
(shifting the trigger in order to compensate the overestimation)
or to use the Brownian Bridge technique which takes into account the probability of hitting a barrier
between monitoring times. One could also use variance reduction techniques such as taking the average of
groups of simulations and then of all simulations.



APPENDIX

Alternative solution (much harder to calculate the Greeks)

#Barrier shifting reduced results accuracy so it was removed.
#B_shift = B * np.exp(0.5826 * sigma * np.sqrt(dt))
ST = S * tf.cumprod(tf.exp((r - sigma ** 2 / 2) * dt + sigma * tf.sqrt(dt) * rand_mtx), axis=1)
non_touch = tf.cast(tf.reduce_all(ST > B_shift, axis=1), tf.float32)
call_payout = tf.cast(tf.maximum(ST[:, -1] - K, 0)*non_touch, tf.float32)
value = tf.cast(tf.exp(-r * T) * tf.reduce_mean(call_payout), tf.float32)
    
'''
