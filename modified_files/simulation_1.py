import numpy as np
from numpy.random import default_rng
import pandas as pd

### import functions from model files
from initialisation_functions_1 import *
from firm_functions_1 import *
from household_functions_1 import *
from market_functions_1 import *
from series_parameters_1 import *
from pathlib import Path


def simulation(r, s, t_max, n, H_max, omega_0, p_0, A, gamma, mu, S_N, delta_0, theta, min_pct,
               Fix_cost_ai, omega_ai, ai_tax_rate,
               rng):
    sim_results = []
    t = 0
    pct_change = delta_0

    households = np.arange(0, n)

    #### intitialise firm and household
    I, pi, total_pi, omega, p, H_D, S_P, S_S, is_ai = create_firm(omega_0, p_0)
    alphas, betas, H_N, H_O, H, H_W, income, m, S, S_C, U, expenditure = create_households(n, H_max, rng)
    N, S_D, H_M, S_M, H_S = 0, 0, 0, 0, 0

    #### initialise firm effort and demand expectations
    # idxs = rng.integers(0, n, size = mu)

    #### household sample S_hat
    S_hat, demand_memory = initialise_demand_expectation(n, alphas, betas, omega, p, H_max, S_N, mu)

    #### save initial results
    step_results = [s, r, t, t_max, n, H_max, A, gamma, mu, S_N, omega_0, p_0, delta_0, theta, min_pct, pct_change,
                    I, pi, total_pi, omega, p, H_D, S_S, S_P, S_hat, N, H_S, H_M, S_D, S_M,
                    np.sum(H_N), np.sum(H_O), np.sum(m), np.median(m), np.mean(m), np.max(m),
                    np.min(m), np.nanmean(U), np.mean(alphas),
                    is_ai, np.mean(H_W), Fix_cost_ai, omega_ai, ai_tax_rate, H_max-np.mean(H_W)]
    sim_results.append(step_results)

    #### the simulation steps can be run for a fixed number, t_max, 
    #### or with the stopping condition that S_S is within 1 unit of S_D.

    #### conditional steps
    # while ((abs(S_S - S_D) >= 1 or abs(pi) >= 1) and t < t_max) or t == 0:
    #     #print('supply demand delta is', abs(S_S - S_D), 'and pi is', abs(pi))
    #     t = t + 1

    #### t_max simulation steps
    for t in range(1, t_max + 1):
        #### firm determines hours
        H_D = determine_hours(S_hat, A, gamma, S_N, n, I)

        #### households determine effort and hours supplied
        H_N = tribute_hours(p, S_N, omega, m)
        for h in range(n):
            H_O[h] = optional_hours(betas[h], alphas[h], H_max, omega, m[h], p, S_N)
        H = np.minimum(H_N + H_O, H_max)

        #### market aggregates hours supplied, determines market hours and effective effort
        H_S = np.sum(H)
        H_M = min(H_D, H_S)
        N, H_W = aggregate_effort(H_S, H_D, H)

        "firm determine use AI or human labors"
        is_ai = determine_is_ai(omega=omega, p=p, H=N, Fix_cost_ai=Fix_cost_ai, omega_ai=omega_ai,
                                ai_tax_rate=ai_tax_rate, A=A, gamma=gamma)
        H_W_human = H_W if not is_ai else np.zeros(n)
        total_human_hours = sum(H_W_human)
        total_human_cost = total_human_hours * omega
        total_ai_cost = 0 if not is_ai else Fix_cost_ai + N * omega_ai

        #### firm produces sugar with effective labour
        S_P = produce_supply(A, N, gamma)
        S_S = S_P + I
        total_ai_tax = ai_tax_rate * S_P * p if is_ai else 0

        #### households plan sugar consumption
        S, income = plan_consumption(S_N, omega, p, hours=H_W_human, m=m, ai_subsidy=total_ai_tax / n)

        #### market aggregates sugar demand and determines sugar sold
        S_D = np.sum(S)
        S_M = min(S_S, S_D)
        S_C = sell_sugar(S_S, S_D, S)

        #### calculate firm profit
        pi = profit(omega=omega, p=p, S_M=S_M, H_M=H_M, Fix_cost_ai=Fix_cost_ai, omega_ai=omega_ai,
                    ai_tax_rate=ai_tax_rate, is_ai=is_ai)
        total_pi += pi

        #### calculate household utility
        U = calculate_utility(H_max=H_max, H_W=H_W_human, S_C=S_C, S_N=S_N, alpha=alphas, beta=betas)

        #### households adjust ledgers
        m, expenditure = update_ledger(income, p, S_C, m)

        #### firm adjusts inventory
        I = adjust_inventory(S_P, S_M, I)

        #### firm updates demand expectations
        S_hat, demand_memory = update_expectation(S_D, demand_memory, mu)

        #### firm raises or lowers wage and price
        # pct_change = delta_0
        ### for annealing
        pct_change = decay(t, delta_0, theta, min_pct)
        omega, p = update_wage_price(omega, p, H_S, H_D, S_S, S_D, pct_change)
        min_H_D = (n * S_N / A) ** (1 / gamma)

        #### write step state to frame
        step_results = [s, r, t, t_max, n, H_max, A, gamma, mu, S_N, omega_0, p_0, delta_0, theta, min_pct, pct_change,
                        I, pi, total_pi, omega, p, H_D, S_S, S_P, S_hat, N, H_S, H_M, S_D, S_M,
                        np.sum(H_N), np.sum(H_O), np.sum(m), np.median(m), np.mean(m), np.max(m),
                        np.min(m), np.nanmean(U), np.mean(alphas),
                        is_ai, np.mean(H_W_human), Fix_cost_ai, omega_ai, ai_tax_rate,
                        total_human_hours, total_human_cost, total_ai_cost, H_max-np.mean(H_W_human)]
        sim_results.append(step_results)

    return sim_results


def run_simulation(fix_ai_cost_list: tuple = (1_000,), ai_omega_list: tuple = (5,), ai_tax_rate_list: tuple = (.5,),
                   t_max: int = 500, reps: int = 100, n_list: tuple = (100,), omega_0_list: tuple = (10,),
                   p_0_list: tuple = (1,), H_max: int = 480, A_list: tuple = (3,), gamma_list: tuple = (1.2,),
                   mu_list: tuple = (3,), S_N_list: tuple = (1200,), delta_0_list: tuple = (.1,),
                   theta_list: tuple = (.1,), min_pct_list: tuple = (0,)):
    #### series setup
    parameters_dict = {"fix_ai_cost_list": fix_ai_cost_list,
                       "ai_omega_list": ai_omega_list,
                       "ai_tax_rate_list": ai_tax_rate_list,
                       "t_max": t_max,
                       "reps": reps,
                       "n_list": n_list,
                       "omega_0_list": omega_0_list,
                       "p_0_list": p_0_list,
                       "H_max": H_max,
                       "A_list": A_list,
                       "gamma_list": gamma_list,
                       "mu_list": mu_list,
                       "S_N_list": S_N_list,
                       "delta_0_list": delta_0_list,
                       "theta_list": theta_list,
                       "min_pct_list": min_pct_list}
    directory, series_name, seed, reps, n_sets, series_params_list = series_params(**parameters_dict)

    #### verify directory exists
    Path(directory).mkdir(parents=True, exist_ok=True)

    #### initialise rng to be used by simulation series
    rng = default_rng(seed)

    print('This experiment consists of {} simulations'.format(n_sets * reps))
    print('consisting of {} parameter sets.'.format(n_sets))

    #### intitialise storage vehicle for series results
    series_results = []

    #### main body of multi parameter set code
    for params in series_params_list:
        sim_results = simulation(*params, rng)
        series_results = series_results + sim_results

    print('\nFinished series.')

    macro_labels = ['set', 'run', 'step', 't_max', 'n', 'H_max', 'A', 'gamma', 'mu', 'S_N', 'omega_0', 'p_0', 'delta_0',
                    'theta', 'min_pct', 'pct_change',
                    'I', 'pi', 'total_pi', 'omega', 'p', 'H_D', 'S_S', 'S_P', 'S_hat', 'N', 'H_S', 'H_M', 'S_D', 'S_M',
                    'total_H_N', 'total_H_O', 'total_m', 'med_m', 'mean_m', 'max_m', 'min_m', 'mean_U', 'mean_alpha',
                    'is_ai', 'mean_hours_worked', 'fix_cost_ai', 'omega_ai', 'ai_tax_rate',
                    'total_human_hours', 'total_human_cost', 'total_ai_cost', 'mean_leisure_time']
    #### Transform list of lists into dataframes

    series_results_frame = pd.DataFrame(series_results, columns=macro_labels)

    srf = series_results_frame
    del (series_results_frame)
    srf['mean_beta'] = 1 - srf['mean_alpha']
    srf['ratio'] = srf['omega'] / srf['p']
    srf['sugar_diff'] = abs(srf['S_S'] - srf['S_D'])
    srf['hours_diff'] = abs(srf['S_S'] - srf['S_D'])
    final_srf = srf[srf['step'] == max(srf['step'])]
    return srf, final_srf
