#### series_parameters_2.py

def series_params(fix_ai_cost_list: tuple=(1_000,), ai_omega_list: tuple = (5,),ai_tax_rate_list: tuple = (.5,),
                  t_max: int = 500, reps: int =100, n_list: tuple = (100,), omega_0_list: tuple = (10,),
                  p_0_list: tuple = (1,), H_max: int = 480, A_list: tuple = (3,), gamma_list: tuple = (1.2,),
                  mu_list: tuple = (3,), S_N_list: tuple = (1200,), delta_0_list: tuple = (.1,),
                  theta_list: tuple = (.1,), min_pct_list: tuple = (0,)):
    series_name = 'ABMSS2024_1'
    directory = './results/'

    #### series variables
    seed = None
    t_max = t_max  # 500 #number of simulation steps
    reps = reps  # 100 #repetitions of each parameter set

    #### simulation variables
    n_list = list(n_list)  # number of households
    omega_0_list = list(omega_0_list)  # inital wage
    p_0_list = list(p_0_list)  # intial price
    H_max = H_max  # max household hours per month

    #### firm parameters
    A_list = list(A_list)  # firm production function coefficient
    gamma_list = list(gamma_list)  # firm production function exponent
    mu_list = list(mu_list)  # memory for firm expectation updates

    #### base consumption requirement
    S_N_list = list(S_N_list)  # [1200] #minimum household consumption

    #### simulated annealing control parameters
    delta_0_list = list(delta_0_list)  # coefficient of decay function
    theta_list = list(theta_list)  # exponent of decay function
    min_pct_list = min_pct_list  #

    "set up parameters regarding AI automation"
    fix_ai_cost_list = list(fix_ai_cost_list)
    ai_omega_list = list(ai_omega_list)
    ai_tax_rate_list = list(ai_tax_rate_list)

    param_sets = [[n, omega_0, p_0, A, gamma, mu, S_N, delta_0, theta, min_pct, fix_ai_cost, ai_omega, ai_tax_rate] for
                  n in n_list
                  for omega_0 in omega_0_list for p_0 in p_0_list for A in A_list for
                  gamma in gamma_list for mu in mu_list for S_N in S_N_list
                  for delta_0 in delta_0_list for theta in theta_list for min_pct in min_pct_list for fix_ai_cost in
                  fix_ai_cost_list for ai_omega in ai_omega_list for ai_tax_rate in ai_tax_rate_list]

    #### determine number of distinct parameter sets
    n_sets = len(param_sets)

    #### add fixed parameters
    #### resulting list is: s, t_max, n, H_max, omega_0, p_0, A, gamma, mu, S_N, delta_0, theta, min_pct
    for s in range(n_sets):
        (param_sets[s]).insert(0, t_max)
        (param_sets[s]).insert(2, H_max)
        (param_sets[s]).insert(0, s)

    #### include run number as 1st parameter for multithreading
    expanded_sets = [item.copy() for item in param_sets for i in range(reps)]
    run_list = list(range(reps)) * n_sets
    for i in range(n_sets * reps):
        (expanded_sets[i]).insert(0, run_list[i])

    return directory, series_name, seed, reps, n_sets, expanded_sets
