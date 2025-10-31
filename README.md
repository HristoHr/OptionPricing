# Option Pricing Models with Barrier Options

This repository contains Python scripts implementing option pricing methods for both vanilla and barrier options.

## Files

- `OptionPricingBlackScholes.py`
- `OptionPricingMonteCarlo.py`

## Description

These scripts provide implementations of popular option pricing models:

- **Vanilla European Options**: Pricing of standard European call and put options using the Black-Scholes analytical formula.
  
- **Barrier Options**: Pricing of exotic barrier options, which are path-dependent options that activate or deactivate if the underlying asset price crosses a specified barrier level during the option's life.
  
  - The Black-Scholes script includes analytical pricing and Greeks calculation for barrier options.
  - The Monte Carlo script simulates asset price paths to estimate barrier option prices and Greeks, addressing the complexity of barrier crossing with advanced techniques such as Brownian Bridge and variance reduction methods to improve accuracy.

## What are Barrier Options?

Barrier options are a type of exotic option where the payoff depends on the underlying asset price breaching a predefined barrier level. Examples include:

- **Knock-In Options**: Become active only if the asset price crosses the barrier.
- **Knock-Out Options**: Become void if the asset price crosses the barrier.

These options are useful for more tailored hedging and speculative strategies compared to vanilla options.

## Usage

The included implementations provide methods to price and calculate risk sensitivities (Greeks) for both option types. You can use these scripts to:

- Compute option prices for European vanilla options.
- Model and price barrier options using analytical formulas or Monte Carlo simulation.
- Analyze how different parameters like volatility, barrier level, and time to expiry affect option values.

Feel free to explore the code to understand the mathematical formulations and adapt them for your financial modeling needs.
