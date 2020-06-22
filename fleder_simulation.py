import pickle as pl
import numpy as np
import sys
from allcode.simulator.SDBNSimpleSimulator import SDBNSimpleSimulator
from allcode.util.Util import Util

if __name__ == "__main__":
   # if len(sys.argv) != 3:
   #     print("Argument error. Expected arguments: 3, got: " + str(len(sys.argv)-1))
   #     print("Expected arguments: ")
   #    print("Fraction of users used for testing")
   #    print("Random seed")

    #test_frac = float(sys.argv[1])
    #split_seed = int(sys.argv[2])

    test_frac = 0.1
    split_seed = 1992

    sim_param_filename = "./data/simulation_inputs/base_simulation_case.pl"
    sim_param_file = open(sim_param_filename, "rb")
    sim_param = pl.load(sim_param_file)

    rand_state = np.random.RandomState(split_seed)

    simulator = SDBNSimpleSimulator(sim_param)
    simulator.initialize_sim(rand_state)

    item_prop_matrix = simulator.get_item_loc()
    user_prop_matrix = simulator.get_user_loc()
    dist_prop_matrix = simulator.get_distance_mat()
    att_mat = simulator.get_attr_mat()

    item_prop_matrix.to_csv("./data/simulation_data/simulation_item_props.csv", index=False)
    user_prop_matrix.to_csv("./data/simulation_data/simulation_user_props.csv", index=False)
    dist_prop_matrix.to_csv("./data/simulation_data/simulation_dist_prop.csv", index=False)
    att_mat.to_csv("./data/simulation_data/simulation_attr_mat.csv", index=False)

    sim_result, cookie_data = simulator.simulate(warm_up_frac=.1)

    # Add ids:
    sim_result = Util.add_obs_user_id(sim_result)
    sim_result = Util.add_list_id(sim_result)

    # Otherwise annoying in feature engineering: list-id would not be informative about the number of query-sessions
    # in the training data. Now we just add a new list_id during feature engineering which re-indexes.
    sim_result = sim_result.rename(columns={'list_id': 'orig_list_id'})

    # Remove first 250 lists, otherwise all lists are first sessions (also warm-up)
    sim_result = sim_result[sim_result['orig_list_id'] > sim_param.warm_up_sessions]
    sim_result_no_rec_eval = sim_result[sim_result['start_time'] <= sim_param.eval_time_cutoff1]
    sim_result_rec_eval = \
        sim_result[(sim_param.eval_time_cutoff1 < sim_result['start_time']) &
                   (sim_result['start_time'] <= sim_param.eval_time_cutoff2)]

    print("Splitting into train, test user-wise: ")
    train, test = Util.split_dataset(sim_result_no_rec_eval, test_frac, rand_state)

    print("storing results")
    sim_result_no_rec_eval.to_csv("./data/simulation_data/full_data_set.csv", index=False)
    train.to_csv("./data/simulation_data/simulation_res_train.csv", index=False)
    # validation.to_csv("./data/simulation_data/simulation_res_valid.csv", index=False)
    test.to_csv("./data/simulation_data/simulation_res_test.csv", index=False)
    sim_result_rec_eval.to_csv("./data/simulation_data/recommender_eval_test.csv", index=False)
    cookie_data.to_csv("./data/simulation_data/simulation_cookie_data.csv", index=False)













