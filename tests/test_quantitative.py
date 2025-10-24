"""
Tests for quantitative models (Black-Scholes and Monte Carlo).
"""
import pytest
import numpy as np
from option_pricing_pipeline.models.quantitative import (
    BlackScholesModel,
    MonteCarloSimulation
)


class TestBlackScholesModel:
    """Test Black-Scholes model."""
    
    def test_call_price_atm(self):
        """Test call price for at-the-money option."""
        bs = BlackScholesModel()
        price = bs.price_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        
        # Price should be positive
        assert price > 0
        # For ATM call, price should be reasonable (roughly 5-15% of stock price)
        assert 5 < price < 15
    
    def test_put_price_atm(self):
        """Test put price for at-the-money option."""
        bs = BlackScholesModel()
        price = bs.price_put(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        
        assert price > 0
        assert 5 < price < 15
    
    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*e^(-rT)."""
        bs = BlackScholesModel()
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        call = bs.price_call(S, K, T, r, sigma)
        put = bs.price_put(S, K, T, r, sigma)
        
        parity = S - K * np.exp(-r * T)
        
        # Put-call parity should hold (with small numerical error)
        assert abs((call - put) - parity) < 0.01
    
    def test_greeks_call(self):
        """Test Greeks calculation for call option."""
        bs = BlackScholesModel()
        greeks = bs.calculate_greeks(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        
        # Check that all Greeks are present
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        # Delta should be between 0 and 1 for calls
        assert 0 < greeks['delta'] < 1
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Vega should be positive
        assert greeks['vega'] > 0
    
    def test_greeks_put(self):
        """Test Greeks calculation for put option."""
        bs = BlackScholesModel()
        greeks = bs.calculate_greeks(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='put'
        )
        
        # Delta should be between -1 and 0 for puts
        assert -1 < greeks['delta'] < 0
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
    
    def test_zero_time_to_maturity(self):
        """Test option pricing at expiration."""
        bs = BlackScholesModel()
        
        # ITM call at expiration
        call = bs.price_call(S=110, K=100, T=0, r=0.05, sigma=0.2)
        assert abs(call - 10) < 0.01
        
        # OTM call at expiration
        call = bs.price_call(S=90, K=100, T=0, r=0.05, sigma=0.2)
        assert abs(call) < 0.01


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation."""
    
    def test_path_generation(self):
        """Test stock price path simulation."""
        mc = MonteCarloSimulation(n_simulations=1000, n_steps=252, seed=42)
        paths = mc.simulate_paths(S0=100, T=1.0, r=0.05, sigma=0.2)
        
        # Check shape
        assert paths.shape == (1000, 253)  # n_steps + 1
        
        # All paths should start at S0
        assert np.allclose(paths[:, 0], 100)
        
        # Prices should be positive
        assert np.all(paths > 0)
    
    def test_european_call_pricing(self):
        """Test European call option pricing with Monte Carlo."""
        mc = MonteCarloSimulation(n_simulations=10000, seed=42)
        result = mc.price_european_option(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        
        # Check result structure
        assert 'price' in result
        assert 'std_error' in result
        assert 'confidence_interval' in result
        
        # Price should be positive
        assert result['price'] > 0
        
        # Compare with Black-Scholes
        bs = BlackScholesModel()
        bs_price = bs.price_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        
        # Monte Carlo should be close to Black-Scholes (within 3 std errors)
        assert abs(result['price'] - bs_price) < 3 * result['std_error']
    
    def test_european_put_pricing(self):
        """Test European put option pricing with Monte Carlo."""
        mc = MonteCarloSimulation(n_simulations=10000, seed=42)
        result = mc.price_european_option(
            S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='put'
        )
        
        assert result['price'] > 0
        
        # Compare with Black-Scholes
        bs = BlackScholesModel()
        bs_price = bs.price_put(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        
        # Should be close
        assert abs(result['price'] - bs_price) < 3 * result['std_error']
    
    def test_var_calculation(self):
        """Test VaR calculation."""
        mc = MonteCarloSimulation(n_simulations=10000, seed=42)
        var_result = mc.calculate_var(
            S0=100, T=1.0, r=0.05, sigma=0.2, confidence_level=0.95
        )
        
        # Check result structure
        assert 'var' in var_result
        assert 'cvar' in var_result
        assert 'var_absolute' in var_result
        assert 'cvar_absolute' in var_result
        
        # VaR should be negative (loss)
        assert var_result['var'] < 0
        
        # CVaR should be more negative than VaR
        assert var_result['cvar'] < var_result['var']
