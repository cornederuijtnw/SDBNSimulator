import pandas as pd
import numpy as np
import time
import math
import copy
import numpy.random as rand
from deprecated import deprecated
from sklearn.metrics.pairwise import euclidean_distances
from allcode.clickmodel_fitters.SDBN import SDBN
from allcode.util.Util import Util
import multiprocessing as mp


class SDBNSimpleSimulator:
    """
    Simulation model based on SDBN, the simulation model by Fleder, and so other papers (see own paper for
    references)
    """
    MINUTES_IN_MONTH = 43200
    MIN_PROB = 10 ** (-5)  # To avoid numeric problems
    PD_RES_NAMES = ['user_id', 'item', 'item_order', 'click', 'attr', 'satis', 'obs_satis',
                    'eval', 'session_count', 'obs_session_count', 'start_time', 'absence_time', 'cookie_count',
                    'cookie_lifetime', 'cur_device', 'user_lifetime']

    SIMULATION_RES = []
    COOKIE_RES = []
    COUNTER = 0

    def __init__(self, param_container):
        self._param_container = param_container
        self._user_pref = None
        self._item_loc = None
        self._distance_mat = None
        self._null_ist = None
        self._awareness_mat = None
        self._null_dist = None
        self._attr_mat = None
        self._satis_mat = None
        self._rand_state = None
        self._has_init_sim = False

    @property
    def param_container(self):
        return self._param_container

    def get_item_loc(self):
        """
        Location of the different items, used to determine the probability of a user clicking the item
        """
        if self._item_loc is not None:
            item_prop = pd.DataFrame(self._item_loc)
            item_prop["item"] = item_prop.index.to_numpy()
            item_prop = item_prop.rename({0: 'X0', 1: "X1"}, axis=1)
            return item_prop
        else:
            raise ValueError("The item property matrix was not constructed yet. Run the simulation first")

    def get_distance_mat(self):
        """
        Distances between users and items
        """
        if self._distance_mat is not None:
            dist_mat = pd.DataFrame(self._distance_mat)
            return dist_mat
        else:
            raise ValueError("Class has not been initialized yet. Please initialize first")

    def get_user_loc(self):
        """
        Location of the different users, used to determine the probability of a user clicking the item
        """
        if self._user_pref is not None:
            user_loc = pd.DataFrame(self._user_pref)
            user_loc["user_id"] = user_loc.index.to_numpy()
            user_loc = user_loc.rename({0: 'X0', 1: "X1"}, axis=1)
            return user_loc
        else:
            raise ValueError("The user property matrix was not constructed yet. Run the simulation first")

    def get_attr_mat(self):
        """
         probabilities of a user clicking an item, given the item is observed
         """
        if self._attr_mat is not None:
            att_mat = pd.DataFrame(self._attr_mat)
            return att_mat
        else:
            raise ValueError("The user property matrix was not constructed yet. Run the simulation first")

    def initialize_sim(self, rand_state):
        """
         Computes the attraction and satisfaction matrices (which are the same), which determine whether
         a user will click on an item
         """
        self._rand_state = rand_state
        rand.set_state(self._rand_state.get_state())
        self._user_pref = self._simulate_user_preference()
        self._item_loc = self._simulate_item_loc()
        self._distance_mat = euclidean_distances(self._user_pref, self._item_loc)
        self._null_dist = np.transpose(
            np.repeat(euclidean_distances([[0, 0]], self._item_loc), self._param_container.users).
            reshape(-1, self._param_container.users))
        # self._awareness_mat = self._get_awareness_mat()
        self._attr_mat = self._get_attractiveness_mat()
        self._satis_mat = self._get_satisfaction_mat() #  With difference salience than attraction
        # self._satis_mat = self._attr_mat  # As in click-chain model, assume attractiveness = satisfaction prob
        self._has_init_sim = True

    def simulate(self, warm_up_frac):
        """
        Starts the simulation
        :param warm_up_frac: fraction of users used to determine the overall item popularity (which is used as item
        order)
        :param init_sim: True if the attraction and satisfaction matrices have not been computed yet
        :param rand_state: random seed
        :return: The result of the simulation, and a dataframe with information about the different cookies
        """
        t0 = time.time()
        print("Starting simulation")
        if not self._has_init_sim:
            raise ValueError("Initialize the simulation first before running the simulation")

        print("Running warm-up")
        warm_up_users = math.ceil(warm_up_frac * self._param_container.users)
        rand.set_state(self._rand_state.get_state())

        init_rel = np.repeat(1 / self._param_container.items, self._param_container.items)

        warm_up_res, _ = self._run_simulation(warm_up_users, init_rel)

        dur = round(time.time() - t0)
        print("Warm-up finished, time: " + str(dur) + " seconds")

        new_relevance = self._get_warmed_up_satisfaction(warm_up_res)

        print("Running simulation:")
        sim_result, cookie_data = self._run_simulation(self._param_container.users - warm_up_users, new_relevance)

        dur = round(time.time() - t0)
        print("Simulation finished, simulation time: " + str(dur) + " seconds")

        return sim_result, cookie_data

    def _get_warmed_up_satisfaction(self, warm_up_res):
        """
        Computes the overall item popularity, which is used to determine the item ordering in the simulation
        """
        unique_lists = warm_up_res.loc[:, ['user_id', 'session_count']].drop_duplicates()
        unique_lists['list_res_id'] = np.arange(unique_lists.shape[0])

        warm_up_res = warm_up_res.\
            set_index(['user_id', 'session_count']).\
            join(unique_lists.set_index(['user_id', 'session_count']), on=['user_id', 'session_count']).\
            reset_index().\
            rename(columns={'item_order': 'pos',
                            'click': 'clicked',
                            'item': 'item_id'}).\
            loc[:, ['list_res_id', 'pos', 'item_id', 'clicked']]

        est_params = SDBN.fit_model(warm_up_res)

        all_items = pd.DataFrame.from_dict({'item_id': np.arange(self._param_container.items),
                                            'dummy': np.repeat(1, self._param_container.items)}).\
            set_index('item_id').\
            join(est_params, on='item_id')

        all_items['att_est'] = all_items['att_est'].fillna(0)
        all_items.loc[all_items['att_est'] < self.MIN_PROB, 'att_est'] = self.MIN_PROB

        return all_items['att_est'].to_numpy()

    def _run_simulation(self, users, order_relevance):
        """
        Runs the simulation procedure
        :param users: number of users to simulate for
        :param order_relevance: the weights used to determine the item ordering
        :return: a pandas dataframe with the simulated clicks, and a dataframe containing cookie information
        """
        SDBNSimpleSimulator.COUNTER = 0
        SDBNSimpleSimulator.SIMULATION_RES = []
        SDBNSimpleSimulator.COOKIE_RES = []

        pool = mp.Pool(mp.cpu_count() - 1)
        cur_proc = []

        cur_time = 0

        for u in range(users):
            cur_time += rand.exponential(self._param_container.user_arrival_rate, 1)[0]

            #pd_res, cookie_dat = SDBNSimpleSimulator._single_user_simulation(self, order_relevance, u, cur_time)

            cur_proc.append(pool.apply_async(SDBNSimpleSimulator._single_user_simulation,
                                             [self._param_container,
                                              order_relevance,
                                              u,
                                              cur_time,
                                              self._attr_mat,
                                              self._satis_mat],
                                             callback=SDBNSimpleSimulator._collect_results_eval_pairs_logreg,
                                             error_callback=SDBNSimpleSimulator._error_callback))

        pool.close()
        pool.join()

        pd_all_res = pd.concat(SDBNSimpleSimulator.SIMULATION_RES, axis=0)
        cookie_all_dat = pd.concat(SDBNSimpleSimulator.COOKIE_RES, axis=0)

        return pd_all_res, cookie_all_dat

    @staticmethod
    def _error_callback(err):
        print("err " + str(err))

    @staticmethod
    def _collect_results_eval_pairs_logreg(results):
        if SDBNSimpleSimulator.COUNTER % 100 == 0:
            print("Simulated user: " + str(SDBNSimpleSimulator.COUNTER))
        SDBNSimpleSimulator.COUNTER += 1
        SDBNSimpleSimulator.SIMULATION_RES.append(results[0])
        SDBNSimpleSimulator.COOKIE_RES.append(results[1])

    @staticmethod
    def _single_user_simulation(param_container, order_pref, user, arrival_time, attr_mat, satis_mat):
        """
        Simulates clicks for one user
        :param order_pref: preferences which determines the item ordering
        :param user: the user-id to simulate for
        :param arrival_time: time at which the user starts its first session
        :return: a pandas dataframe with the simulated clicks, and a dataframe containing cookie information
        """
        # I do not correct the cookie lifetime distribution for switching devices. As the self-transition
        # probabilities are quite large, this should not have too much of an impact

        # normalize order_preference:
        order_pref = order_pref / np.sum(order_pref)
        sim_res = pd.DataFrame(columns=SDBNSimpleSimulator.PD_RES_NAMES)

        # Cookie churn rate in seconds, from paper (I use 60 seconds for the first, such that everything is in minutes):
        # (numpy.random.exponential's scale parameters is beta=1/lambda), so no need to define things in terms of lambda

        act_devices_cookie_dic = {}  # To register only the active cookies
        device_cookie_dic = {}  # To register all cookies

        # Make the initial draws:
        session_count = 1
        cookie_count = 1

        # cookie_lifetimes[0] is the primal cookie lifetime, i.e., the one that determines the user churn
        # which would make sense considering the distribution is the uncensored one
        user_lifetime = np.sum(Util.r_hyperexponential(
            rand.geometric(p=param_container.user_lifetime_phases, size=1)[0],
            param_container.cookie_churn_rates, param_container.cookie_churn_rate_probabilities))

        act_device = np.where(rand.multinomial(1, param_container.device_initial_probabilities) == 1)[0][0]
        act_devices_cookie_dic[act_device] = {'device': act_device,  # although the key, added for completeness
                                                 'cookie_id': cookie_count,
                                                 'creation_time': arrival_time,
                                                 'lifetime': Util.r_hyperexponential(1,
                                                              param_container.cookie_churn_rates,
                                                              param_container.cookie_churn_rate_probabilities)[0],
                                                 'dev_cookie_count': 1,
                                                 'obs_session_count': 1}

        # Draw a small exponential time to avoid ties in the clustering component:
        cur_time = arrival_time

        # Simulate the clicks for this session
        while cur_time < arrival_time + user_lifetime:
            ses_sim_res, time_till_next = SDBNSimpleSimulator._simulate_clicks(
                                                                param_container,
                                                                act_devices_cookie_dic[act_device]['cookie_id'],
                                                                act_devices_cookie_dic[act_device]['lifetime'],
                                                                act_device,
                                                                cur_time,
                                                                order_pref,
                                                                session_count,
                                                                act_devices_cookie_dic[act_device]['obs_session_count'],
                                                                user,
                                                                user_lifetime,
                                                                attr_mat,
                                                                satis_mat)
            # Determine the environment of the next session
            cur_time += time_till_next
            session_count += 1

            # Otherwise no use in changing devices: there will be no next session
            if cur_time < arrival_time + user_lifetime:
                act_device, act_devices_cookie_dic, device_cookie_dic, cookie_count = \
                    SDBNSimpleSimulator._simulate_cookie_churn(
                                                param_container,
                                                act_device,
                                                act_devices_cookie_dic,
                                                param_container.cookie_churn_rates,
                                                cookie_count,
                                                cur_time,
                                                device_cookie_dic,
                                                param_container.cookie_churn_rate_probabilities)

            sim_res = sim_res.append(ses_sim_res)

        # Add all activate devices to the device dictionary
        for _, device in act_devices_cookie_dic.items():
            device_cookie_dic[device['cookie_id']] = device

        device_cookie_pd = pd.DataFrame.from_dict(device_cookie_dic).T
        device_cookie_pd['user'] = user

        return sim_res, device_cookie_pd

    @staticmethod
    def _simulate_cookie_churn(param_container, act_device, act_devices_cookie_dic, cookie_churn_rates, cookie_count, cur_time,
                               device_cookie_dic, rate_prob):
        """
        Simulate whether the user will start the next session on a different device, or whether the current cookie
        will churn
        :param act_device: The current active device
        :param act_devices_cookie_dic: Dictionary last cookies for each device
        :param cookie_churn_rates: Rate at which cookies churn (for hyperexponential distribution)
        :param cookie_count: The overall user cookie-count
        :param cur_time: Time at which the next session would start
        :param device_cookie_dic: All previous cookies used (does not include active cookies)
        :param rate_prob: Probability of a cookie churn rate being picked (for hyperexponential distribution)
        :return:
        """
        new_device = np.where(rand.multinomial(1,
                                param_container.device_transition_probabilities[act_device, :]) == 1)[0][0]

        if act_device != new_device:  # device switch
            act_device = new_device  # activates the device

            # Check if the device already existed, if not add it
            if not new_device in act_devices_cookie_dic:
                cookie_count += 1
                act_devices_cookie_dic[act_device] = {'device': act_device,
                                                      'cookie_id': cookie_count,
                                                      'creation_time': cur_time,
                                                      'lifetime': Util.r_hyperexponential(1, cookie_churn_rates,
                                                                                          rate_prob)[0],
                                                      'dev_cookie_count': 1,
                                                      'obs_session_count': 1}

            # Check if the current cookie on the device has churned, if so, create a new cookie
        c_create_time = act_devices_cookie_dic[act_device]['creation_time']
        c_life_time = act_devices_cookie_dic[act_device]['lifetime']
        c_id = act_devices_cookie_dic[act_device]['cookie_id']

        # Cookie has churned, store current cookie and create a new one
        if cur_time > c_create_time + c_life_time:
            cookie_count += 1
            c_dev_cookie_count = act_devices_cookie_dic[act_device]['dev_cookie_count']

            # Store the previous cookie info
            device_cookie_dic[c_id] = copy.deepcopy(act_devices_cookie_dic[act_device])

            # Create a new cookie
            act_devices_cookie_dic[act_device] = {'device': act_device,
                                                  'cookie_id': cookie_count,
                                                  'creation_time': cur_time,
                                                  'lifetime': Util.r_hyperexponential(1, cookie_churn_rates,
                                                                                      rate_prob)[0],
                                                  'dev_cookie_count': c_dev_cookie_count + 1,
                                                  'obs_session_count': 1}

        else:  # only update the observed session count
            act_devices_cookie_dic[act_device]['obs_session_count'] += 1
        return act_device, act_devices_cookie_dic, device_cookie_dic, cookie_count

    @staticmethod
    def _simulate_clicks(param_container, cookie_count, cookie_lifetime, cur_dev_state, cur_time, order_pref, session_count,
                         obs_session_count, user, user_lifetime, attr_mat, satis_mat):
        """
        Simulates the clicks in the current session
        :param cookie_count: Current user cookie count
        :param cookie_lifetime: Current cookie lifetime
        :param cur_dev_state: Current device
        :param cur_time: Time at which the query session starts
        :param order_pref: Item weights used to determine the item ordering
        :param session_count: Overall user session count
        :param obs_session_count: Session count as observed by the analyst
        :param user: Current user-id
        :param user_lifetime: current lifetime of the user
        :return: pandas dataframe containing the clicks and several other query session data for this query session
        """
        # Draw the attraction and satisfaction parameters for all positions
        item_order = rand.choice(np.arange(param_container.items), param_container.list_size,
                                 replace=False, p=order_pref)
        real_att = np.array([rand.binomial(1, x, 1) for x in attr_mat[user, item_order]]).reshape(-1)
        real_satis = np.array([rand.binomial(1, x, 1) for x in satis_mat[user, item_order]]).reshape(-1)
        eval_vec = np.zeros(param_container.list_size + 1)
        eval_vec[0] = 1
        obs_satis = np.zeros(param_container.list_size)
        click_vec = np.zeros(param_container.list_size)
        # If the user is satisfied by the query, simulate clicks (otherwise all zero)
        for k in range(param_container.list_size):
            click_vec[k] = real_att[k] * eval_vec[k]
            obs_satis[k] = real_satis[k] * click_vec[k]
            eval_vec[k + 1] = eval_vec[k] * (1 - obs_satis[k])

        time_till_next = rand.pareto(param_container.inter_session_pareto_shape) - 1

        # Add results to dictionary
        ses_sim_res = pd.DataFrame.from_dict(dict(zip(SDBNSimpleSimulator.PD_RES_NAMES,
                                            [np.repeat(user, param_container.list_size),
                                             item_order,
                                             np.arange(param_container.list_size) + 1,
                                             click_vec,
                                             real_att,
                                             real_satis,
                                             obs_satis,
                                             eval_vec[0:param_container.list_size],
                                             np.repeat(session_count, param_container.list_size),
                                             np.repeat(obs_session_count, param_container.list_size),
                                             np.repeat(cur_time, param_container.list_size),
                                             np.repeat(time_till_next, param_container.list_size),
                                             np.repeat(cookie_count, param_container.list_size),
                                             np.repeat(cookie_lifetime, param_container.list_size),
                                             np.repeat(cur_dev_state, param_container.list_size),
                                             np.repeat(user_lifetime, param_container.list_size)])))
        return ses_sim_res, time_till_next

    def _simulate_item_loc(self):
        """
        Simulate the item locations (bivariate normal )
        """
        v1 = rand.standard_normal(self._param_container.items)
        v2 = rand.standard_normal(self._param_container.items)
        res = np.hstack([v1.reshape(-1, 1), v2.reshape(-1, 1)])

        return res

    def _simulate_user_preference(self):
        """
        Simulate the user locations (bivariate normal )
        """
        v1 = rand.standard_normal(self._param_container.users)
        v2 = rand.standard_normal(self._param_container.users)
        res = np.hstack([v1.reshape(-1, 1), v2.reshape(-1, 1)])

        return res

    @deprecated("As function was not used in the initial simulator, it has been commented out (to reduce the number"
                "of simulation parameters)")
    def _get_awareness_mat(self):
        pass
        # awareness = \
        #     self._lambda * np.exp(-self._null_dist**2/self._theta) + \
        #     (1-self._lambda)*np.exp(-self._distance_mat**2/(self._theta*self._kappa))
        #
        # return awareness

    def _get_satisfaction_mat(self):
        """
        Computed the satisfaction matrix
        """
        similarity = np.exp(-self._param_container.user_distance_sensitivity * np.log(self._distance_mat))

        row_sums = np.sum(similarity, axis=1)
        att = (similarity * np.exp(self.param_container.salience_satis))/ \
              (np.repeat(row_sums, self._param_container.items).reshape(-1, self._param_container.items) +
               similarity * (np.exp(self.param_container.salience_satis) - 1))

        # Stability
        att[np.where(att > 1 - self.MIN_PROB)] = 1 - self.MIN_PROB
        att[np.where(att < self.MIN_PROB)] = self.MIN_PROB

        return att

    def _get_attractiveness_mat(self):
        """
        Computes the attractiveness matrix
        """
        similarity = np.exp(-self._param_container.user_distance_sensitivity * np.log(self._distance_mat))

        row_sums = np.sum(similarity, axis=1)
        att = (similarity * np.exp(self.param_container.salience_att))/ \
              (np.repeat(row_sums, self._param_container.items).reshape(-1, self._param_container.items) +
               similarity * (np.exp(self.param_container.salience_att) - 1))

        # Stability
        att[np.where(att > 1 - self.MIN_PROB)] = 1 - self.MIN_PROB
        att[np.where(att < self.MIN_PROB)] = self.MIN_PROB

        return att

