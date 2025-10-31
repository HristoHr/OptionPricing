'''Solution for problem 2.1'''
import tensorflow as tf
import tensorflow_probability as tfp


def black_scholes_price_greeks():
    S = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    T = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    B = tf.placeholder(tf.float32)
    Phi = tfp.distributions.Normal(0., 1.).cdf

    sigma2 = sigma ** 2
    mu = (r - (sigma2 / 2)) / sigma2

    # Vanilla option evaluation
    x1 = (tf.log(S / K)) / (sigma * tf.sqrt(T)) + (1 + mu) * sigma * tf.sqrt(T)
    x2 = x1 - sigma * tf.sqrt(T)
    A = S * Phi(x1) - K * tf.exp(-r * T) * Phi(x2)

    # Barrier
    y1 = tf.log((B ** 2) / (S * K)) / (sigma * tf.sqrt(T)) + (1 + mu) * sigma * tf.sqrt(T)
    y2 = y1 - sigma * tf.sqrt(T)
    C = S * (B / S) ** (2 * (mu + 1)) * Phi(y1) - K * tf.exp(-r * T) * (B / S) ** (2 * mu) * Phi(y2)

    value = (A - C)
    '''
    Delta - Derivative of an option with respect to (w.r.t.) the spot price, ∂C∂S
    Vega - Derivative of an option w.r.t. the underlying volatility, ∂C∂σ
    Rho - Derivative of an option w.r.t. the interest rate, ∂C∂ρ
    Theta - (Negative) derivative of an option w.r.t. the time to expiry, ∂C∂t
    Gamma - Second derivative of an option w.r.t. the spot price, ∂2C∂S2  
    '''

    greeks = tf.gradients(value, [S, sigma, r, T])
    delta = greeks[0]
    vega = greeks[1] / 100
    rho = greeks[2] / 100
    theta = -greeks[3] / 365
    gamma = tf.gradients(delta, [S])
    result = value, delta, vega, rho, theta, gamma

    def price(S0, strike, time_to_expiry, volatility, risk_free_rate, barrier):
        with tf.Session() as sess:
            'returns: predicted value, delta, vega, rho, theta, gamma'
            res = sess.run(result,
                           {
                               S: S0,
                               K: strike,
                               r: risk_free_rate,
                               sigma: volatility,
                               T: time_to_expiry,
                               B: barrier})

        return res

    return price


K_ = 50.
B_ = 45.
S_ = 50.
T_ = 0.5
sigma_ = 0.2
r_ = 0.1

analytical = black_scholes_price_greeks()
print(analytical(S_, K_, T_, sigma_, r_, B_))

'''
Output:
(3.8794565, 0.7612063, 0.07276109, 0.14131421, -0.011730154, [0.017268449])
'''
