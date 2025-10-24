"""
Quantitative models for option pricing and risk calculations.
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


class BlackScholesModel:
    """
    Black-Scholes model for European option pricing.
    """
    
    def __init__(self):
        self.d1 = None
        self.d2 = None
    
    def price_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(self.d1) - K * np.exp(-r * T) * norm.cdf(self.d2)
        return call_price
    
    def price_put(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-self.d2) - S * norm.cdf(-self.d1)
        return put_price
    
    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing Greeks (delta, gamma, theta, vega, rho)
        """
        if T <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Common terms
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        
        # Delta
        if option_type == 'call':
            delta = cdf_d1
        else:
            delta = cdf_d1 - 1
        
        # Gamma (same for calls and puts)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        
        # Vega (same for calls and puts)
        vega = S * pdf_d1 * np.sqrt(T) / 100  # Divided by 100 for 1% change
        
        # Theta
        if option_type == 'call':
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * cdf_d2) / 365
        else:
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * cdf_d2 / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


class MonteCarloSimulation:
    """
    Monte Carlo simulation for option pricing.
    """
    
    def __init__(self, n_simulations: int = 10000, n_steps: int = 252, seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float
    ) -> np.ndarray:
        """
        Simulate stock price paths using geometric Brownian motion.
        
        Args:
            S0: Initial stock price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Array of simulated paths (n_simulations x n_steps+1)
        """
        dt = T / self.n_steps
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            )
        
        return paths
    
    def price_european_option(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Price European option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with price and standard error
        """
        paths = self.simulate_paths(S0, T, r, sigma)
        terminal_prices = paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(terminal_prices - K, 0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0)
        
        discounted_payoffs = np.exp(-r * T) * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (
                price - 1.96 * std_error,
                price + 1.96 * std_error
            )
        }
    
    def calculate_var(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            T: Time horizon (in years)
            r: Risk-free rate
            sigma: Volatility
            confidence_level: Confidence level for VaR
            
        Returns:
            Dictionary with VaR and CVaR metrics
        """
        paths = self.simulate_paths(S0, T, r, sigma)
        terminal_prices = paths[:, -1]
        
        # Calculate returns
        returns = (terminal_prices - S0) / S0
        
        # VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar,
            'var_absolute': var * S0,
            'cvar_absolute': cvar * S0
        }
