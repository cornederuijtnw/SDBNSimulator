import numpy as np
import numpy.random as rand
from sklearn.model_selection import train_test_split


class Util:
    """
    Utility functions
    """
    @staticmethod
    def r_hyperexponential(n, lambda_vec, p_vec):
        """
        Draws from hyper-exponential distribution
        :param n: number of draws
        :param lambda_vec: vector of rates
        :param p_vec: vector of probabilities to draw with a certain rate
        :return: vector with n draws from a hyper-exponential distribution
        """
        res = np.zeros(n)
        for i in range(n):
            res[i] = rand.exponential(
                np.dot(
                    np.squeeze(
                        np.eye(N=len(p_vec),
                               M=1,
                               k=-rand.choice(len(p_vec), 1, replace=False, p=p_vec)[0])),
                       lambda_vec), 1)

        return res

    @staticmethod
    def add_obs_user_id(sim_data):
        """
        Adds the observed user id (i.e., different for different cookies) to the data set
        """
        sim_data_ordered = sim_data.\
            sort_values(by=['user_id', 'cookie_count'])

        obs_user_indices = sim_data_ordered.loc[:, ['user_id', 'cookie_count']].drop_duplicates()
        obs_user_indices['obs_user_id'] = np.arange(obs_user_indices.shape[0])  # Add the observed user id

        sim_data_ordered = \
            sim_data_ordered. \
                set_index(['user_id', 'cookie_count']). \
                join(obs_user_indices.set_index(['user_id', 'cookie_count']),
                     on=['user_id', 'cookie_count']). \
                reset_index()

        # Note that the max obs_user_id may be smaller than the user_id, as this only considers train/valid users
        return sim_data_ordered

    @staticmethod
    def add_list_id(sim_data):
        unique_qsessions = sim_data.\
            groupby(['obs_user_id', 'obs_session_count']).\
            size().\
            rename('freq').\
            reset_index().\
            drop(['freq'], axis=1)

        unique_qsessions['list_id'] = np.arange(unique_qsessions.shape[0])

        sim_data_with_list_id = sim_data.\
            set_index(['obs_user_id', 'obs_session_count']).\
            join(unique_qsessions.set_index(['obs_user_id', 'obs_session_count']),
                                            on=['obs_user_id', 'obs_session_count']).\
            reset_index()

        sim_data_with_list_id = sim_data_with_list_id.\
            sort_values(['obs_user_id', 'obs_session_count', 'item_order'])

        return sim_data_with_list_id

    @staticmethod
    def split_dataset(sim_data, test_frac, rand_state, valid_frac=None):
        """
        Splits dataset into train, validation and test based on users
        """
        unique_users = sim_data['user_id'].unique()
        train_users, test_users = train_test_split(unique_users, test_size=test_frac, random_state=rand_state)

        if valid_frac is not None:
            valid_of_train_frac = valid_frac/(1-test_frac)
            train_users, val_users = train_test_split(train_users, test_size=valid_of_train_frac, random_state=rand_state)
            valid_data = sim_data[sim_data['user_id'].isin(val_users)]

        train_data = sim_data[sim_data['user_id'].isin(train_users)]
        test_data = sim_data[sim_data['user_id'].isin(test_users)]

        if valid_frac is not None:
            return train_data, valid_data, test_data

        return train_data, test_data