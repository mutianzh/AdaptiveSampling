"""

Global sampling to improve regression model accuracy in general
Gradually focus more in good region

"""

from utils import *
sys.path.insert(0, '/home/mutianzh/PycharmProjects/GlobalLibrary')
from spectreIOlib import TestSpice, Netlists, send_command, prepare_socket

class Sampling(AdaptiveSampling):
    def f(self, x):
        """
            circuit simulation
            :param x: numpy array of unnormalized parameters
            :return: numpy array of all metrics
            """

        command1 = f'm = mycircuit.wholerun_normal(p, parallelid={par_id})'
        x = np.array(x).reshape(-1, self.in_dim)
        lst_p = []
        lst_m = []
        i = 1
        for params in x:
            print('Running simulation %s' % i)
            command0 = f'p = np.array({list(params)})'
            z0 = send_command(command0, s, dict2)
            z1 = send_command(command1, s, dict2, ['m'])
            lst_m.append(m)
            i += 1

        np_m = np.array(lst_m)
        if self.test_bench == 'amp2stage': # correct phase margin values
            for i in range(np_m.shape[0]):
                if np_m[i, 6] < 0:
                    np_m[i, 6] += 180
        np_m = np_m[:, self.output_selection]

        if self.test_bench == 'amp2stage':
            return -np_m.reshape(-1, self.out_dim)
        if self.test_bench == 'comp2nd':
            return np_m.reshape(-1, self.out_dim)

if __name__ == '__main__':
    dict2 = globals()

    exp = 0
    seed = exp
    par_id = 5
    port = 10005
    s = prepare_socket('mcdell0.usc.edu', port)

    log_path = f'./datasets/{version}_log_{exp}.txt'
    data_path = f'./datasets/{version}_adaptive_{exp}.csv'
    model_name = f'./datasets/{version}_adaptive_model_{exp}.h5'
    initial_dataset = None
    model_choice = 'GP'

    with s:
        adaptive_sampling = Sampling(data_path=data_path, log_path=log_path)
        all_x, all_y = adaptive_sampling = adaptive_sampling.main(exp_id=exp, seed=seed, test_bench=test_bench,
                                                                  version=version, budget=budget,
                                                                  model_choice=model_choice,
                                                                  num_per_iteration=num_per_iteration,
                                                                  initial_dataset=initial_dataset)

        # save dataset
        if test_bench == 'amp2stage':
            dataset = np.concatenate((all_x, -all_y), axis=1)
        elif test_bench == 'comp2nd':
            dataset = np.concatenate((all_x, all_y), axis=1)
        pd.DataFrame(dataset).to_csv(data_path, index=False, header=False)
