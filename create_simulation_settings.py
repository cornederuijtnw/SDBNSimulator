import numpy as np
import pickle as pk
import itertools
from allcode.simulator.SimulationParamContainer import SimulationParamContainer

if __name__ == "__main__":
    base_file_name = "./data/simulation_inputs/base_simulation_case.pl"
    test_file_name = "./data/simulation_inputs/test_simulation_case.pl"
    file_name_postfix = "./data/simulation_inputs/simulation_case.pl"

    # All times are assumed to be in minutes
    user_distance_sensitivity_default = 1
    users_default = 20000
    items_default = 100

    list_size = 10
    user_lifetime_default = 0.5

    # Device switch probabilities (removed the console, so re-normalize the probabilities):
    dev_init_prob = np.array([0.639, 0.112, 0.246])/sum([0.639, 0.112, 0.246])
    dev_trans_prob = np.array([[0.9874, 0.0042, 0.0084],
                               [0.00256, 0.9697, 0.0046],
                               [0.029, 0.0018, 0.9773]])
    dev_trans_prob = dev_trans_prob / np.repeat(np.sum(dev_trans_prob, 1), 3).reshape(3, 3)

    user_arrival_rate = 1/0.2  # approximately 1K users per day

    cookie_churn_rates = np.array([1, 25, 14*60, 15*24*60, 337*24*60])
    cookie_churn_rate_prob = np.array([6/100, 7/100, 7/100, 18/100, 62/100])

    inter_session_pareto_shape = 0.111

    salience_att = 5
    salience_satis = 5

    warm_up_sessions = 250
    eval_cut_off1 = 43200
    eval_cut_off2 = eval_cut_off1 * 2

    # Store base case first:
    print("Defining and storing base simulation case")
    p_cont_base = SimulationParamContainer(items_default,
                                           users_default,
                                           list_size,
                                           user_distance_sensitivity_default,
                                           user_lifetime_default,
                                           dev_init_prob,
                                           dev_trans_prob,
                                           user_arrival_rate,
                                           inter_session_pareto_shape,
                                           cookie_churn_rates,
                                           cookie_churn_rate_prob,
                                           salience_att,
                                           salience_satis,
                                           warm_up_sessions,
                                           eval_cut_off1,
                                           eval_cut_off2)

    base_file = open(base_file_name, "wb")
    pk.dump(p_cont_base, base_file)
    base_file.close()

    print("Base simulation file stored")