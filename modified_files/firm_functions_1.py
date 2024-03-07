#### firm_functions_1.py

import numpy as np
from functools import partial

"""
Jieshu: introduce variables: 
- Fix_cost_ai: initial fixed investment to set up 
- omega_ai: cost of using AI to produce the equivalent amount sugar .that requires one labor hour. 
- ai_tax_rate: tax per sugar produced by AI

After decide H_M, firm evaluate whether to use AI or human

"""


def profit(omega, p, S_M, H_M, Fix_cost_ai, omega_ai, ai_tax_rate, is_ai: bool):
    return p * S_M - omega * H_M if not is_ai else p * (1-ai_tax_rate) * S_M - Fix_cost_ai - omega_ai * H_M

def determine_hours(S_D, A, gamma, S_N, n, I):
    planned = max(S_D, n * S_N)
    # use inventory, could make hours demanded = 0
    if 0 < I < planned:
        planned -= I
    elif I >= planned:
        planned = 0
    H = (planned / A) ** (1 / gamma)
    return H


def produce_supply(A, H, gamma):
    return A * H ** gamma

def determine_is_ai(omega, p, H, Fix_cost_ai, omega_ai, ai_tax_rate, A, gamma):
    imagined_supply = produce_supply(A=A, H=H, gamma=gamma)
    partial_profit = partial(profit, omega=omega, p=p, S_M=imagined_supply, H_M=H, Fix_cost_ai=Fix_cost_ai,
                             omega_ai=omega_ai, ai_tax_rate=ai_tax_rate)
    profit_with_ai, profit_without_ai = partial_profit(is_ai=True), partial_profit(is_ai=False)
    is_ai = profit_with_ai > profit_without_ai
    return is_ai

def adjust_inventory(S_P, S_M, I):
    I += S_P - S_M
    return I


#### exponential decay
def decay(t, delta_0, theta, min_pct):
    return max(delta_0 * np.exp(-1 * theta * t), min_pct)


### WAGE AND PRICE UPDATE ALGORITHMS
### percentage change algorithms
### with pct set by decay function, simulated annealing
### with constant percent, fixed change
def update_wage_price(omega, p, H_S, H_D, S_S, S_D, pct):
    omega_new = omega
    if S_S > S_D:
        if H_S > H_D:
            p = p * (1 - pct)
        else:
            omega_new = omega_new * (1 - pct)
    elif S_S < S_D:
        if H_S > H_D:
            omega_new = omega_new * (1 + pct)
        else:
            p = p * (1 + pct)
    #### sticky wage
    # omega_new = omega
    return omega_new, p


def update_expectation(value, memory, mu):
    memory = np.append(memory[1:], value)
    weights = np.arange(1, mu + 1)
    hat = np.average(memory, weights=weights)
    # only use last observed value
    # hat = value
    return hat, memory
