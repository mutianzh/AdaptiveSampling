"""
Implemented by Mutian Zhu

version 2_1:
Use Gaussian process regression
normalize x
use original y


version 2_2:
Global search:
correction is pred + max_val_err
Use mc sampling to find all sample with pred non inferior to existing samples
find ones with largest sparsity

Local search:
Use CMA-ES to maximize each metric
Divide into groups

v17_2
Select candidates from samples within margin



7/16/2022

Add non linear measure


"""
import numpy as np

from consts_2 import *


def normalization(x, min_x, max_x):
    x = np.array(x)
    min_x = np.array(min_x)
    max_x = np.array(max_x)
    return (x - min_x) / (max_x - min_x)

def un_normalization(nx, min_x, max_x):
    nx = np.array(nx)
    min_x = np.array(min_x)
    max_x = np.array(max_x)
    return nx * (max_x - min_x) + min_x

def discrete_normalization(x, min_x, max_x, stp):
    xrange = (max_x - min_x) / stp + 1
    nx = ((x - min_x) / stp) / xrange
    return nx

def discrete_unnormalization(nx, min_x, max_x, stp):
    xrange = (max_x - min_x) / stp + 1
    x = np.multiply(np.floor(np.multiply(nx, xrange)), stp) + min_x
    return x


def with_max_distance(x1s, cur_x):
    """
    Distance measure to existing samples
    Distance is corrected with non linearity measures
    :param x1s:
    :param cur_x:
    :return:
    """

    max_dst = 0
    max_x = np.array([])
    max_idx = None
    idx = 0
    for x1 in x1s:
        min_dst = np.min(np.linalg.norm(x1 - cur_x, axis=1))
        if min_dst > max_dst:
            max_x = x1
            max_dst = min_dst
            max_idx = idx
        idx += 1
    return max_x, max_idx


def with_max_priority(x1s, cur_x, non_lin_measure):
    """
    Distance measure to existing samples
    Distance is corrected with non linearity measures
    :param x1s:
    :param cur_x:
    :return:
    """

    max_dst = 0
    max_x = np.array([])
    max_idx = None
    idx = 0
    for i in range(x1s.shape[0]):
        x1 = x1s[i]
        # min_dst = np.min(np.linalg.norm(x1 - cur_x, axis=1)) * (1 + non_lin_measure[i])
        min_dst = np.min(np.linalg.norm(x1 - cur_x, axis=1))
        if min_dst > max_dst:
            max_x = x1
            max_dst = min_dst
            max_idx = idx
        idx += 1
    return max_x, max_idx



class AdaptiveSampling:
    def __init__(self, data_path, log_path):
        self.data_path = data_path
        self.log_path = log_path
        self.log = None
        self.model = None
        self.test_bench = 'ZDT1'
        self.model_choice = 'GP'
        self.min_y = []
        self.max_y = []
        self.in_dim = 6
        self.out_dim = 2
        self.output_selection = [0, 1]
        self.min_par = np.array([0 for _ in range(self.in_dim)])
        self.max_par = np.array([1 for _ in range(self.in_dim)])
        self.stp_par = np.array([])
        self.method_x = None
        self.method_y = None
        self.budget = 400
        self.KDTree = None

    def f(self, x):
        """
        function evaluation
        :param x: numpy array of un-normalized parameters
        :return: numpy array of all metrics
        """

        def ZT1(X):
            f1 = X[:, 0]
            g = 1 + np.multiply(9, np.divide(np.sum(X[:, 1:], axis=1), X.shape[1] - 1))
            h = 1 - np.power(np.divide(f1, g), 0.5)
            f2 = np.multiply(g, h)

            return np.concatenate((f1.reshape(-1, 1), f2.reshape(-1, 1)), axis=1)

        def ZT2(X):
            f1 = X[:, 0]
            g = 1 + np.multiply(9, np.divide(np.sum(X[:, 1:], axis=1), X.shape[1] - 1))
            h = 1 - np.power(np.divide(f1, g), 2)
            f2 = np.multiply(g, h)

            return np.concatenate((f1.reshape(-1, 1), f2.reshape(-1, 1)), axis=1)

        def F4(x):
            n = 10
            j1 = [3, 5, 7, 9]
            j2 = [2, 4, 6, 8, 10]

            def f1_helper(j):
                ret = np.square(
                    x[:, j - 1] - np.multiply(0.8 * x[:, 0], np.cos((6 * math.pi * x[:, 0] + j * math.pi / n) / 3)))
                return ret

            def f2_helper(j):
                ret = np.square(
                    x[:, j - 1] - np.multiply(0.8 * x[:, 0], np.sin(6 * math.pi * x[:, 0] + j * math.pi / n)))
                return ret

            f1 = x[:, 0] + (2 / len(j1)) * np.sum(np.array([f1_helper(j) for j in j1]), axis=0)
            f2 = 1 - np.sqrt(x[:, 0]) + (2 / len(j2)) * np.sum(np.array([f2_helper(j) for j in j2]), axis=0)

            return np.concatenate((f1.reshape(-1, 1), f2.reshape(-1, 1)), axis=1)

        if self.test_bench == 'ZDT1':
            return ZT1(x)
        elif self.test_bench == 'ZDT2':
            return ZT2(x)
        elif self.test_bench == 'F4':
            return F4(x)

    def F4_pareto_x(self, num):
        n = self.in_dim
        j1 = []
        j2 = []
        if n == 10:
            j1 = [3, 5, 7, 9]
            j2 = [2, 4, 6, 8, 10]

        x = np.random.uniform(low=0, high=1, size=(num, n))
        x[:, 1:] = x[:, 1:] * 2 - 1
        xs = [[] for _ in range(n)]
        xs[0] = x[:, 0]
        for j in j1:
            xs[j - 1] = np.multiply(0.8 * x[:, 0], np.cos((6 * math.pi * x[:, 0] + j * math.pi / n) / 3))

        for j in j2:
            xs[j - 1] = np.multiply(0.8 * x[:, 0], np.sin(6 * math.pi * x[:, 0] + j * math.pi / n))

        ret = np.array(xs).T
        return ret

    def ZDT_pareto_x(self, num):
        n = num
        stp = 1 / n
        x1 = np.arange(0, 1, stp).reshape(n, -1)
        xs = np.zeros((n, self.in_dim - 1))
        return np.concatenate((x1, xs), axis=1)

    def pareto_x(self, num=1000):
        if self.test_bench == 'F4':
            return self.F4_pareto_x(num)
        elif self.test_bench in ['ZDT1', 'ZDT2']:
            return self.ZDT_pareto_x(num)
        else:
            print('No closed form pareto front')
            return None

    def preprocess_x(self, x):
        if not self.method_x:
            return x
        elif self.method_x == 'normalization':
            if self.test_bench in ['ZDT1', 'ZDT2', 'F4']:
                return normalization(x, self.min_par, self.max_par)
            elif self.test_bench in ['amp2stage', 'comp2nd']:
                return discrete_normalization(x, self.min_par, self.max_par, self.stp_par)

    def reverse_preprocess_x(self, nx):
        if not self.method_x:
            return nx
        elif self.method_x == 'normalization':
            if self.test_bench in ['ZDT1', 'ZDT2', 'F4']:
                return un_normalization(nx, self.min_par, self.max_par)
            elif self.test_bench in ['amp2stage', 'comp2nd']:
                return discrete_unnormalization(nx, self.min_par, self.max_par, self.stp_par)

    def preprocess_y(self, y):
        if not self.method_y:
            return y
        elif self.method_y == 'normalization':
            return normalization(y, self.min_y, self.max_y)

    def reverse_preprocess_y(self, ny):
        if not self.method_y:
            return ny
        elif self.method_y == 'normalization':
            return un_normalization(ny, self.min_y, self.max_y)

    def build_model_GP(self, xs, ys):
        gp_model = []
        for i in range(self.out_dim):
            cur_y = ys[:, i].reshape(ys.shape[0], 1)
            m = GPy.models.GPRegression(xs, cur_y, GPy.kern.RBF(input_dim=xs.shape[1], ARD=True))
            gp_model.append(m)
        return gp_model

    def train_model_GP(self, model, xs, ys):
        for i in range(self.out_dim):
            cur_y = ys[:, i].reshape(ys.shape[0], 1)
            m = model[i]
            m.kern.variance = np.var(cur_y)
            m.lengthscale = np.std(xs, 0)
            m.likelihood.variance = 1e-2 * np.var(cur_y)
            m.optimize()
        return model

    def build_model(self, xs, ys):
        if self.model_choice == 'GP':
            return self.build_model_GP(xs, ys)

    def train_model(self, model, xs, ys):
        if self.model_choice == 'GP':
            return self.train_model_GP(model, xs, ys)

    def make_prediction(self, x):
        y_pred = []
        if self.model_choice == 'GP':
            ys = []
            for i in range(self.out_dim):
                yp, _ = self.model[i].predict(x)
                ys.append(np.array(yp))
            y_pred = np.concatenate(ys, axis=1)
        return y_pred

    def prediction_correction(self, y_pred, nxs, all_nx, all_ny, validation_err):
        # # Method 1: correction according to distance and Lipschitz criteria
        # # find Lipschitz criteria for each
        # Lipschitz = find_Lipschitz(all_nx, all_ny)
        # d = []
        # lpz = []
        # for i in range(nxs.shape[0]):
        #     # min distance to existing samples
        #     x0 = nxs[i]
        #     dsts = np.linalg.norm(x0 - all_nx, axis=1)
        #     d.append(np.min(dsts))
        #
        #     # Lipschitz criteria of cell where mc sample belong to
        #     cell_id = which_cell(x0, all_nx)
        #     lpz.append(Lipschitz[cell_id])
        #
        # d = np.array(d)
        # lpz = np.array(lpz)
        # nd = (d - np.min(d)) / (np.max(d) - np.min(d))
        # y_pred_c = y_pred - np.max(validation_err, axis=0) * (
        #             2 / (1 + np.exp(-2 * np.multiply(lpz, nd.reshape(-1, 1)))) - 1)

        # Method2: Just add max val error to all samples
        max_val = np.max(validation_err, axis=0)
        # y_pred_c = y_pred - max_val
        y_pred_c = np.array(y_pred)

        return y_pred_c

    def is_non_inferior(self, y1, ys):
        y1 = np.squeeze(y1)
        if self.test_bench in ['ZDT1', 'ZDT2', 'F4']:
            for y2 in ys:
                if y1[0] == y2[0] and y1[1] == y2[1]:
                    continue
                elif y1[0] >= y2[0] and y1[1] >= y2[1]:
                    return False
                else:
                    continue
            return True

        elif self.test_bench in ['amp2stage']:
            for y2 in ys:
                if y1[0] == y2[0] and y1[1] == y2[1] and y1[2] == y2[2]:
                    continue
                elif y1[0] >= y2[0] and y1[1] >= y2[1] and y1[2] >= y2[2]:
                    return False
                else:
                    continue
            return True

        elif self.test_bench in ['comp2nd']:
            for y2 in ys:
                if y1[0] == y2[0] and y1[1] == y2[1] and y1[2] == y2[2] and y1[3] == y2[3]:
                    continue
                elif y1[0] >= y2[0] and y1[1] >= y2[1] and y1[2] >= y2[2] and y1[3] >= y2[3]:
                    return False
                else:
                    continue
            return True

        # y1 = np.squeeze(y1)
        # for i in range(ys.shape[0]):
        #     y2 = ys[i]
        #     if np.array_equal(y1, y2):
        #         continue
        #     dominated = True
        #     for j in range(self.out_dim):
        #         if y1[j] < y2[j]:
        #             dominated = False
        #     if dominated:
        #         return False
        # return True

    def find_non_inferiors(self, all_y):
        all_y = np.array(all_y)
        non_inferior_index = []
        remain = []
        for i in range(all_y.shape[0]):
            y_i = all_y[i]
            if self.is_non_inferior(y_i, all_y):
                non_inferior_index.append(i)
            else:
                remain.append(i)
        return non_inferior_index


    def is_superior(self, y1, ys):
        # check if y1 is superior to at least one sample in ys
        y1 = np.squeeze(y1)
        if self.test_bench in ['ZDT1', 'ZDT2', 'F4']:
            for y2 in ys:
                if y1[0] == y2[0] and y1[1] == y2[1]:
                    continue
                elif y1[0] <= y2[0] and y1[1] <= y2[1]:
                    return True
            return False

    def estimate_dg(self, i, all_nx, all_ny):
        y = all_ny[i]
        x = all_nx[i]
        y_l = [None] * self.in_dim
        y_r = [None] * self.in_dim
        dst_l = [None] * self.in_dim
        dst_r = [None] * self.in_dim

        for j in range(all_nx.shape[0]):
            # xi and xj should be different points
            if j != i:
                xj = all_nx[j]
                yj = all_ny[j]
                v = x - xj
                d = np.linalg.norm(v)
                for which_dim in range(self.in_dim):
                    left = True if v[which_dim] > 0 else False
                    neb_dst = dst_l if left else dst_r
                    neb_y = y_l if left else y_r
                    d_old = neb_dst[which_dim]
                    if not d_old or d < d_old:
                        neb_y[which_dim] = copy.deepcopy(yj)
                        neb_dst[which_dim] = d
                    else:
                        pass
            else:
                pass

        out_indices = [0,1]
        all_dg = []
        for o in out_indices:
            dg = []
            for dim in range(self.in_dim):
                if dst_r[dim] and dst_l[dim]:
                    dl = dst_l[dim]
                    yl = y_l[dim][o]
                    dr = dst_r[dim]
                    yr = y_r[dim][o]
                    yc = y[o]
                    fx1 = (yr - yc) / dr
                    print(f'g1 for f{o} at dim{dim}: {fx1} \n')
                    fx2 = (yc - yl) / dl
                    print(f'g2 for f{o} at dim{dim}: {fx2} \n')
                    # fxx = abs((fx1 - fx2) / ((dr + dl)/2))
                    fxx = abs(fx1 - fx2)
                    dg.append(fxx)

            print(f'dg for f{o}: {dg} \n')
            self.log.write(f'dg for f{o}: {dg} \n')

            max_dg = np.max(dg) if len(dg) > 0 else 0
            all_dg.append(max_dg)

        ret = np.max(all_dg)
        return ret

    def estimate_dg_2(self, index, all_nx, all_ny):
        x0 = all_nx[index]
        y0 = all_ny[index]

        # find k nearest neighbors
        if not self.KDTree:
            self.KDTree = KDTree(all_nx)
        k = 2 * self.in_dim + 1  # Number of neighbors to find
        dd, ii = self.KDTree.query(x0, k=k)
        ii = ii[1:]
        dd = dd[1:]
        all_dgs = []
        for o in range(self.out_dim):
            dgs = []
            for i in range(len(ii)):
                x1 = all_nx[ii[i]]
                y1 = all_ny[ii[i]][o]
                d1 = dd[i]
                for j in range(i, len(ii)):
                    x2 = all_nx[ii[j]]
                    y2 = all_ny[ii[j]][o]
                    d2 = dd[j]

                    g1 = (y1 - y0[o]) / d1
                    g2 = (y2 - y0[o]) / d2
                    u1 = x1 - x0
                    u2 = x2 - x0
                    cosine = np.inner(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))

                    dg = abs(g1 - cosine * g2)
                    dgs.append(dg)
            all_dgs.append(dgs)

        # print(np.array(all_dgs))
        return np.max(all_dgs)


    def global_search(self, budget, validation_err, all_x, all_y, all_nx, all_ny):
        # Find out which cell each mc_sample belong to
        # mc_samples = qmc.LatinHypercube(d=in_dim).random(n=all_x.shape[0] * 50)
        xlimits = np.array([[0, 1] for _ in range(self.in_dim)])
        sampling = LHS(xlimits=xlimits)

        # if self.test_bench in ['amp2stage', 'comp2nd']:
        #     mc_x = discrete_unnormalization(sampling(self.budget * 10), self.min_par, self.max_par, self.stp_par)
        # else:
        #     mc_x = un_normalization(sampling(self.budget * 10), self.min_par, self.max_par)
        mc_x = un_normalization(sampling(self.budget * 10), self.min_par, self.max_par)
        mc_nx = self.preprocess_x(mc_x)

        mc_nyp = self.make_prediction(mc_nx)
        mc_yp = self.reverse_preprocess_y(mc_nyp)

        nif_index = self.find_non_inferiors(mc_yp)
        mc_nif_nx = mc_nx[nif_index]
        mc_nif_yp = mc_yp[nif_index]

        return mc_nx, mc_yp, mc_nif_nx, mc_nif_yp


        # mc_yp_c = self.prediction_correction(y_pred=mc_yp, nxs=mc_nx, all_nx=all_nx, all_ny=all_ny,
        #                                      validation_err=validation_err)
        # # pareto front of existing simulated points
        # non_inferior_index = self.find_non_inferiors(all_y)
        # nif_y = all_y[non_inferior_index]
        # print(f'Number of non inferior simulated points: {len(non_inferior_index)}')

        # # pareto front of regression model
        # nif_mc_index = self.find_non_inferiors(mc_yp_c)
        # print(f'Size of nif mc samples is {len(nif_mc_index)}')
        # nif_mc_nx = []
        # nif_mc_yc = []
        # for i in nif_mc_index:
        #     if self.is_non_inferior(mc_yp_c[i], all_y):
        #         nif_mc_nx.append(mc_nx[i])
        #         nif_mc_yc.append(mc_yp_c[i])
        # print(f'Size of filtered nif mc samples is {len(nif_mc_nx)}')
        # if len(nif_mc_nx) == 0:
        #     nif_mc_nx = np.array([])
        #     nif_mc_yc = np.array([])
        # else:
        #     nif_mc_nx = np.array(nif_mc_nx)
        #     nif_mc_yc = np.array(nif_mc_yc)

        # to_add = np.min([budget, nif_mc_nx.shape[0]])
        # if to_add == 0:
        #     return np.array([])

        # # iteratively pick ones that bring largest improvement to hyper volume
        # ref = np.max(all_y[:, output_selection], axis=0) * 2
        # cur_y = np.array(all_y)
        # cur_hv = cal_hyper(cur_y[:, output_selection], ref)
        # selected_id = []
        # while to_add > 0:
        #     max_improve = 0
        #     max_id = None
        #     for i in range(nif_x.shape[0]):
        #         temp_y = np.concatenate((cur_y, nif_y[i].reshape(1, -1)), axis=0)
        #         temp_hv = cal_hyper(temp_y[:, output_selection], ref)
        #         improve = temp_hv - cur_hv
        #         if improve > max_improve:
        #             max_improve = improve
        #             max_id = i
        #     selected_id.append(max_id)
        #     cur_y = np.concatenate((cur_y, nif_y[max_id].reshape(1, -1)), axis=0)
        #     cur_hv = cal_hyper(cur_y[:, output_selection], ref)
        #     to_add -= 1

        # # Iteratively pick ones at sparse locations
        # all_selected_nx = []
        # cur_nx = np.array(all_nx)
        # while to_add > 0:
        #     ret, _ = with_max_distance(nif_mc_nx, cur_nx)
        #     if ret.shape[0] > 0:
        #         all_selected_nx.append(ret)
        #         cur_nx = np.concatenate((cur_nx, ret.reshape(1, -1)), axis=0)
        #     to_add -= 1
        #
        # all_selected_nx = np.array(all_selected_nx)
        # all_selected_x = self.reverse_preprocess_x(all_selected_nx)
        # return all_selected_x

        # return nif_mc_nx, nif_mc_yc

    def cma_es(self, x0, y0, pop_size, sigma, search_method):
        """
        CMA-ES method
        :param x0: start point in param space. Normalized between 0 and 1
        :param y0: start point in metric space.
        :param pop_size: population size
        :param sigma: initial sigma
        :return: List of all samples created during the search
        """

        all_cma_nx = []  # preprocessed x
        all_cma_ny = []  # preprocessed y

        if search_method == 1:
            max_iteration = 10
            # method 1: optimize each metric respectively
            def cost(nxs, o):
                ny_pred = self.make_prediction(np.array(nxs))
                return ny_pred[:, o], list(nxs), list(ny_pred)

            for o in range(self.out_dim):
                es = cma.CMAEvolutionStrategy(
                    x0=x0,
                    sigma0=sigma,
                    inopts={'bounds': [0, 1], "popsize": pop_size},
                )
                cnt = 0
                while not es.stop():
                    cnt += 1
                    cma_x = es.ask()
                    if self.test_bench in ['amp2stage', 'comp2nd']:
                        xs = discrete_unnormalization(np.array(cma_x), self.min_par, self.max_par, self.stp_par)
                    else:
                        xs = un_normalization(np.array(cma_x), self.min_par, self.max_par)
                    nxs = self.preprocess_x(xs)
                    c, cma_nx, cma_ny = cost(nxs, o)
                    all_cma_nx += cma_nx
                    all_cma_ny += cma_ny

                    if cnt >= max_iteration:
                        break
                    es.tell(cma_x, c)  # return the result to the optimizer

        elif search_method == 2:
            max_iteration = 20
            # method 2: try to beat start point y0
            def soft_cost(nxs, ny_target):
                ny_pred = self.make_prediction(np.array(nxs))
                c = 1 / (1 + np.exp(-(ny_pred - ny_target)))
                return np.sum(c, axis=1), list(nxs), list(ny_pred)
            target = y0
            es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=sigma,
                inopts={'bounds': [0, 1], "popsize": pop_size},
            )
            cnt = 0
            while not es.stop():
                cnt += 1
                cma_x = es.ask()  # as for new points to evaluate
                # if self.test_bench in ['amp2stage', 'comp2nd']:
                #     xs = discrete_unnormalization(np.array(cma_x), self.min_par, self.max_par, self.stp_par)
                #     cma_x = discrete_normalization(xs, self.min_par, self.max_par, self.stp_par)
                # else:
                #     xs = un_normalization(np.array(cma_x), self.min_par, self.max_par)

                xs = un_normalization(np.array(cma_x), self.min_par, self.max_par)

                nxs = self.preprocess_x(xs)
                c, cma_nx, cma_ny = soft_cost(nxs, target)
                all_cma_nx += cma_nx
                all_cma_ny += cma_ny

                if cnt >= max_iteration:
                    break
                es.tell(cma_x, c)  # return the result to the optimizer

        return all_cma_nx, all_cma_ny

    def select_from(self, xs, K):
        """
        select K representative items from xs
        :param xs: set to select from
        :param K: nmmber of elements to select
        :return:
        """
        return


    def local_search(self, budget, validation_err, all_x, all_y, all_nx, all_ny):
        """
        :param budget:
        :param validation_err:
        :param all_x:
        :param all_y:
        :param all_nx:
        :param all_ny:
        :return: numpy array of local samples and pareto front of local samples
        """
        start_time = time.time()
        non_inferior_index = self.find_non_inferiors(all_y)
        all_nif_nx = all_nx[non_inferior_index]
        all_nif_x = all_x[non_inferior_index]
        all_nif_ny = all_ny[non_inferior_index]
        all_nif_y = all_y[non_inferior_index]
        print("Number of non inferior samples")
        print(len(non_inferior_index))
        self.log.write(f'Number of non inferior samples is {len(non_inferior_index)} \n')

        # # sort and select non-inferior sample according to the distance in metric space
        # K = 20
        # min_dsts = []
        # for i in non_inferior_index:
        #     y_i = all_ny[i][self.output_selection]
        #     min_d = np.sort(np.linalg.norm(y_i - all_nif_ny.reshape(-1, len(output_selection)), axis=1))[1]
        #     min_dsts.append(min_d)
        # selected_nif_index = np.argsort(min_dsts)[::-1]

        # select K representative ones
        K = 20
        to_add = np.min([K, len(non_inferior_index)])
        selected_nif_index = []
        starter = random.choice(list(range(all_nif_ny.shape[0])))
        selected_nif_index.append(starter)
        cur_ny = np.array([all_nif_ny[starter]])
        while to_add > 0:
            ret, idx = with_max_distance(all_nif_ny, cur_ny)
            if ret.shape[0] > 0:
                selected_nif_index.append(idx)
                cur_ny = np.concatenate((cur_ny, ret.reshape(1, -1)), axis=0)
            to_add -= 1


        pop_size = 50
        sigma0 = 0.01
        search_method = 2
        to_add = budget
        all_selected_nx = []
        all_cma_nx = []
        all_cma_ny = []
        all_cma_nif_nx = []
        all_cma_nif_ny = []
        cur_nx = np.array(all_nx)

        for index in selected_nif_index:
            # if to_add <= 0:
            #     break
            # explore around selected simulated point (x0, y0)
            x0 = all_nif_x[index]

            if self.test_bench in ['amp2stage', 'comp2nd']:
                normalized_x0 = discrete_normalization(x0, self.min_par, self.max_par, self.stp_par)
            else:
                normalized_x0 = normalization(x0, self.min_par, self.max_par)

            ny0 = all_nif_ny[index]

            cma_nx, cma_ny = self.cma_es(normalized_x0, ny0, pop_size, sigma0, search_method)
            all_cma_nx += cma_nx
            all_cma_ny += cma_ny

            temp_all_nx = np.array(all_cma_nif_nx + cma_nx)
            temp_all_ny = np.array(all_cma_nif_ny + cma_ny)

            nif_index = self.find_non_inferiors(temp_all_ny)

            all_cma_nif_nx = list(temp_all_nx[nif_index])
            all_cma_nif_ny = list(temp_all_ny[nif_index])


        all_cma_nx = np.array(all_cma_nx)
        all_cma_nif_nx = np.array(all_cma_nif_nx)
        all_cma_ny = np.array(all_cma_ny)
        all_cma_nif_ny = np.array(all_cma_nif_ny)

        all_cma_y = self.reverse_preprocess_y(all_cma_ny)
        all_cma_nif_y = self.reverse_preprocess_y(all_cma_nif_ny)
        all_cma_yc = self.prediction_correction(y_pred=all_cma_y, nxs=all_cma_nx, all_nx=all_nx, all_ny=all_ny,
                                               validation_err=validation_err)


        # nif_cma_index = self.find_non_inferiors(all_cma_y)
        # print(f'Size of nif cma samples is {len(nif_cma_index)}')
        # nif_cma_nx = []
        # nif_cma_yc = []
        # for i in nif_cma_index:
        #     if self.is_non_inferior(all_cma_yc[i], all_y):
        #         nif_cma_nx.append(all_cma_nx[i])
        #         nif_cma_yc.append(all_cma_yc[i])
        # print(f'Size of filtered nif cma samples is {len(nif_cma_nx)}')


        # to_add = np.min([len(nif_cma_nx), budget])
        # while to_add > 0:
        #     ret, _ = with_max_distance(nif_cma_nx, cur_nx)
        #     if ret.shape[0] > 0:
        #         all_selected_nx.append(ret)
        #         cur_nx = np.concatenate((cur_nx, ret.reshape(1, -1)), axis=0)
        #     to_add -= 1



        # group1 = []  # samples that can beat at least one simulated point on pareto front
        # group2 = []  # samples that are on existing pareto front
        # group3 = []  # samples that behind existing pareto front
        # # divide all local sample into groups
        # for i in range(all_cma_y.shape[0]):
        #     yi = all_cma_y[i]
        #     if self.is_superior(yi, nif_y):
        #         group1.append(all_cma_nx[i])
        #     elif self.is_non_inferior(yi, all_y):
        #         group2.append(all_cma_nx[i])
        #     else:
        #         group3.append(all_cma_nx[i])
        # group1 = np.array(group1)
        # group2 = np.array(group2)
        # group3 = np.array(group3)
        #
        # if len(group1) > 0:
        #     print('select group 1')
        #     self.log.write('select group 1 \n')
        #     ret, _ = self.with_max_distance(group1, cur_nx)
        #     all_selected_nx.append(ret)
        #     cur_x = np.concatenate((cur_x, ret.reshape(1, -1)), axis=0)
        #     to_add -= 1
        # elif len(group2) > 0:
        #     print('select group 2')
        #     self.log.write('select group 2 \n')
        #     ret, _ = self.with_max_distance(group2, cur_nx)
        #     all_selected_nx.append(ret)
        #     cur_x = np.concatenate((cur_x, ret.reshape(1, -1)), axis=0)
        # else:
        #     print('select group 3')
        #     self.log.write('select group 3 \n')
        #     # TODO: search with smaller step size next time

        # all_n_selected = np.array(all_selected_nx)
        # all_selected = self.reverse_preprocess_x(all_n_selected)

        print(f'Finish search with {time.time() - start_time} seconds')
        # return all_selected

        # return np.array(nif_cma_nx), np.array(nif_cma_yc)
        return all_cma_nx, all_cma_y, all_cma_nif_nx, all_cma_nif_y

    def main(self, exp_id, seed, test_bench, version, budget, model_choice, num_per_iteration, initial_dataset=None):
        # ===============================
        # Initialization
        # ===============================
        self.budget = budget
        self.test_bench = test_bench
        self.model_choice = model_choice
        self.min_y = []
        self.max_y = []
        if self.test_bench in ['ZDT1', 'ZDT2']:
            self.in_dim = 6
            self.out_dim = 2
            self.output_selection = [0, 1]
            self.min_par = np.array([0 for _ in range(self.in_dim)])
            self.max_par =  np.array([1 for _ in range(self.in_dim)])
            self.method_x = None
            self.method_y = None
        elif self.test_bench == 'F4':
            self.in_dim = 10
            self.out_dim = 2
            self.output_selection = [0, 1]
            self.min_par =  np.array([-1 for _ in range(self.in_dim)])
            self.max_par =  np.array([1 for _ in range(self.in_dim)])
            self.min_par[0] = 0
            self.method_x = 'normalization'
            self.method_y = None
        elif self.test_bench == 'amp2stage':
            self.in_dim = 18
            self.output_selection = [4, 5, 6]
            self.out_dim = len(self.output_selection)
            self.min_par = np.array(
                [0.2e-6, 0.2e-6, 0.4e-6, 0.2e-6, 0.2e-6, 45e-9, 45e-9, 60e-9, 45e-9, 45e-9, 10e-15, 10, 20e-15, 0.5,
                 0.2, 0.5, 1, 1])
            self.max_par = np.array(
                [4e-6, 8e-6, 24e-6, 4e-6, 8e-6, 0.2e-6, 0.2e-6, 0.4e-6, 0.2e-6, 0.2e-6, 250e-15, 1000, 100e-15, 0.8,
                 0.8, 0.7, 20, 20])
            self.stp_par = np.array(
                [20e-9, 40e-9, 40e-9, 20e-9, 20e-9, 5e-9, 5e-9, 10e-9, 5e-9, 5e-9, 5e-15, 10, 10e-15, 0.05, 0.025, 0.01,
                 1, 1])
            self.method_x = 'normalization'
            self.method_y = 'normalization'
        elif self.test_bench == 'comp2nd':
            self.in_dim = 16
            self.output_selection = [0, 1, 2, 3]
            self.out_dim = len(self.output_selection)
            self.min_par = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1e-15, 5e-15, ])
            self.max_par = np.array([24, 48, 48, 24, 64, 64, 32, 12, 64, 24, 20, 40, 300, 288, 1e-13, 100e-15, ])
            self.stp_par = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1e-15, 5e-15, ])
            self.method_x = 'normalization'
            self.method_y = 'normalization'

        np.random.seed(seed)

        start_time = time.time()
        process_start_time = time.process_time()
        start_iteration = 1
        start_run_time = 0
        start_process_time = 0
        total_sim_time = 0
        time_stamps = []

        # initial datasets
        if debug:
            initial_num = 10
        else:
            initial_num = self.in_dim * 10
        last_added = list(range(initial_num))  # samples added last iteration

        # TODO
        # if not initial_dataset:
        #     # Generate from scratch
        #     # all_nx = qmc.LatinHypercube(d=in_dim).random(n=initial_num)
        #     all_normalized_x = np.random.uniform(low=0, high=1, size=(initial_num, self.in_dim))
        #     if self.test_bench in ['amp2stage', 'comp2nd']:
        #         all_x = discrete_unnormalization(all_normalized_x, self.min_par, self.max_par, self.stp_par)
        #     else:
        #         all_x = un_normalization(all_normalized_x, self.min_par, self.max_par)
        #
        #     sim_start = time.time()
        #     all_y = self.f(all_x)
        #     sim_time = time.time() - sim_start
        #     print(f'Sim time is {sim_time}')
        #
        #     total_sim_time += sim_time
        #     process_time = time.process_time() - process_start_time + total_sim_time
        #     run_time = time.time() - start_time
        #     print(f'Total run time is {run_time}')
        #     print(f'Process time is {process_time}')
        #     time_stamps.append([initial_num, run_time, process_time])
        #
        # else:
        #     # Load from existing dataset
        #     initial_set = pd.read_csv(initial_dataset).to_numpy()[:initial_num]
        #     all_x = initial_set[:, :self.in_dim]
        #     if self.test_bench == 'amp2stage':
        #         all_y = -initial_set[:, self.in_dim:][:, self.output_selection]
        #     else:
        #         all_y = -initial_set[:, self.in_dim:][:, self.output_selection]

        s1 = 'Run time is'
        s2 = 'Process time is'
        s3 = 'Number of Samples'
        s4 = 'Iteration'
        s5 = 'mean val error is'
        s6 = 'sigma is'
        run_time_hist = []
        process_time_hist = []
        n_samples_hist = []
        with open(f'./datasets/v17_2_amp2stage_30ksec_log_{exp_id}.txt') as f:
            lines = f.readlines()
        for line in lines:
            if s1 in line:
                run_time = float(line.split()[3])
                run_time_hist.append(run_time)

            if s2 in line:
                process_time = float(line.split()[3])
                process_time_hist.append(process_time)

            if s3 in line:
                n = int(line.split()[3])
                n_samples_hist.append(n + 30)

            if s4 in line:
                end_iteration = int(line.split()[1])

        start_iteration = end_iteration + 1
        n_start_samples = n_samples_hist[-1]
        start_run_time = run_time_hist[-1]
        start_process_time = process_time_hist[-1]

        for k in range(len(n_samples_hist)):
            time_stamps.append([n_samples_hist[k] + 30, run_time_hist[k], process_time_hist[k]])


        initial_set = pd.read_csv(self.data_path).to_numpy()[:n_start_samples]
        all_x = initial_set[:, :self.in_dim]
        all_y = -initial_set[:, self.in_dim:]

        # initial model
        last_added = list(range(all_x.shape[0] - 30, all_x.shape[0]))

        self.min_y = np.min(all_y[:n_start_samples - 30], axis=0)
        self.max_y = np.max(all_y[:n_start_samples - 30], axis=0)

        all_nx = self.preprocess_x(all_x[:n_start_samples - 30])
        all_ny = self.preprocess_y(all_y[:n_start_samples - 30])
        train_data = np.concatenate((all_nx, all_ny), axis=1)
        np.random.shuffle(train_data)
        n_xtrain = train_data[:, :self.in_dim]
        n_ytrain = train_data[:, self.in_dim:]

        model = self.build_model(n_xtrain, n_ytrain)
        self.model = self.train_model(model, n_xtrain, n_ytrain)

        # TODO
        # os.system(f"rm {self.log_path}")
        self.log = open(self.log_path, 'a')


        # TODO
        # # initial random model
        # model = self.build_model(all_x, all_y)
        # self.model = model

        # start iteration
        iteration = start_iteration
        while True:
            print('==============Iteration %s ===============' % iteration)
            # print('Number of neurons: %s' % n_neurons)
            # print('Number of layers: %s' % n_layers)
            print('Number of Samples %s' % all_x.shape[0])
            print('Experiment %s' % exp_id)

            self.log.write('======================Iteration %s =================\n' % iteration)
            # log.write('Number of neurons: %s\n' % n_neurons)
            # log.write('Number of layers: %s\n' % n_layers)
            self.log.write('Number of Samples %s\n' % all_x.shape[0])
            self.log.write('Experiment %s \n' % exp_id)
            # tf.keras.backend.clear_session()

            # ===============================
            # model selection
            # ===============================

            # record validation error of newly added data
            last_x = all_x[last_added]
            last_nx = self.preprocess_x(last_x)
            ny_pred = self.make_prediction(x=last_nx)
            y_pred = self.reverse_preprocess_y(ny_pred)
            y_true = all_y[last_added]
            validation_err = np.array(abs(y_true - y_pred))

            # print('Mean validation error')
            # print(np.mean(validation_err, axis=0))
            # self.log.write('Mean validation eror \n')
            # self.log.write(f'{np.mean(validation_err, axis=0)}')
            #
            # print('Max validation error')
            # print(np.max(validation_err, axis=0))
            # self.log.write('Max validation eror \n')
            # self.log.write(f'{np.max(validation_err, axis=0)}')

            # prepare training data
            # preprocessing data
            self.min_y = np.min(all_y, axis=0)
            self.max_y = np.max(all_y, axis=0)
            all_nx = self.preprocess_x(all_x)
            all_ny = self.preprocess_y(all_y)
            train_data = np.concatenate((all_nx, all_ny), axis=1)
            np.random.shuffle(train_data)
            n_xtrain = train_data[:, :self.in_dim]
            n_ytrain = train_data[:, self.in_dim:]

            # print(self.min_y)
            # print(self.max_y)
            # print(n_xtrain)
            # print(n_ytrain)

            # train model
            if model_choice == 'GP':
                # Gaussian process
                model = self.build_model(n_xtrain, n_ytrain)
                self.model = self.train_model(model, n_xtrain, n_ytrain)
            else:
                self.model = None
            # # neural network
            # args = [n_neurons, n_layers, final_epochs]
            # n_neurons, n_layers, final_epochs = model_update(args)

            # ===============================
            # Global search
            # ===============================
            global_nx, global_y, global_nif_nx, global_nif_y = self.global_search(budget=num_per_iteration,
                                                      validation_err=validation_err, all_x=all_x, all_y=all_y,
                                                      all_nx=all_nx,
                                                      all_ny=all_ny)

            local_nx, local_y, local_nif_nx, local_nif_y = self.local_search(budget=num_per_iteration,
                                                   validation_err=validation_err, all_x=all_x, all_y=all_y,
                                                   all_nx=all_nx,
                                                   all_ny=all_ny)

            all_model_nx = np.concatenate((global_nx, local_nx), axis=0)
            all_model_ypred = np.concatenate((global_y, local_y), axis=0)
            all_model_nif_nx = np.concatenate((global_nif_nx, local_nif_nx), axis=0)
            all_model_nif_ypred = np.concatenate((global_nif_y, local_nif_y), axis=0)

            # find pareto front from all samples on model
            nif_index = self.find_non_inferiors(all_model_nif_ypred)
            all_model_nif_nx = all_model_nif_nx[nif_index]
            all_model_nif_ypred = all_model_nif_ypred[nif_index]

            print(f'Number of nif samples: {len(nif_index)} \n')
            self.log.write(f'Number of nif samples: {len(nif_index)} \n')

            # nif_nx, nif_ypred, other_nx, other_ypred, = [], [], [], []
            # for i in range(all_samples_nx.shape[0]):
            #     if i in nif_samples_index:
            #         nif_nx.append(all_samples_nx[i])
            #         nif_ypred.append(all_samples_ypred[i])
            #     else:
            #         other_nx.append(all_samples_nx[i])
            #         other_ypred.append(all_samples_ypred[i])
            #
            # nif_nx = np.array(nif_nx)
            # nif_ypred = np.array(nif_ypred)
            # other_nx = np.array(other_nx)
            # other_ypred = np.array(other_ypred)

            mean = np.mean(validation_err, axis=0)
            sigma = np.std(validation_err, axis=0)



            print(f'mean val error is {mean}')
            print(f'sigma is {sigma}')
            self.log.write(f'mean val error is {mean} \n')
            self.log.write(f'sigma is {sigma} \n')

            alpha_choice_method = 2
            new_nx = []
            new_ny = []
            if alpha_choice_method == 1:
                # method 1: try three different margins in parallel
                margins = []
                alpha = [2, 0, -2]
                # alpha = [0, -0.5, -1]

                for a in alpha:
                    margin = mean + a * sigma
                    for o in range(self.out_dim):
                        margin[o] = 0 if margin[o] < 0 else margin[o]

                    if iteration == 1:
                        for o in range(self.out_dim):
                            margin[o] = 99

                    margins.append(margin)

                print(f'margins: {np.array(margins)}')
                self.log.write(f'margins: {np.array(margins)} \n')

                all_nx_in_margin = [[] for _ in range(len(alpha))]
                all_ypred_in_margin = [[] for _ in range(len(alpha))]
                for k in range(len(alpha)):
                    all_model_ypred_c = all_model_ypred - margins[k]
                    for i in range(all_model_nx.shape[0]):
                        if self.is_non_inferior(all_model_ypred_c[i], all_model_nif_ypred):
                            all_nx_in_margin[k].append(all_model_nx[i])
                            all_ypred_in_margin[k].append(all_model_ypred[i])
                    print(f'Number of samples in margin is {len(all_nx_in_margin[k])}')
                    self.log.write(f'Number of samples in margin is {len(all_nx_in_margin[k])} \n')
                    all_nx_in_margin[k] = np.array(all_nx_in_margin[k])
                    all_ypred_in_margin[k] = np.array(all_ypred_in_margin[k])

                # prioritize samples in margin
                cur_nx = np.array(all_nx)
                for k in range(len(alpha)):
                    candidates = all_nx_in_margin[k]
                    temp_nx = []
                    to_add = np.min([10, candidates.shape[0]])
                    while to_add > 0:
                        ret, _ = with_max_distance(candidates, cur_nx)
                        if ret.shape[0] > 0:
                            temp_nx.append(ret)
                            cur_nx = np.concatenate((cur_nx, ret.reshape(1, -1)), axis=0)
                        to_add -= 1
                    new_nx += temp_nx

            elif alpha_choice_method == 2:
                # Use single alpha. Decrease it as number of budgets reduces
                min_a = -2
                max_a = 2
                cur_eval = all_x.shape[0]
                mid = int(self.budget / 2)
                rate = 0.1
                a = (max_a - min_a) / (1 + np.exp(rate * (cur_eval - mid))) + min_a
                print(f'Alpha is {a}')
                margin = mean + a * sigma
                for o in range(self.out_dim):
                    margin[o] = 0 if margin[o] < 0 else margin[o]

                if iteration == 1:
                    for o in range(self.out_dim):
                        margin[o] = 99

                print(f'margin: {np.array(margin)}')
                self.log.write(f'margin: {np.array(margin)} \n')

                # find samples in margin
                all_model_ypred_c = all_model_ypred - margin
                all_nx_in_margin = []
                all_ny_in_margin = []
                for i in range(all_model_nx.shape[0]):
                    if self.is_non_inferior(all_model_ypred_c[i], all_model_nif_ypred):
                        all_nx_in_margin.append(all_model_nx[i])
                        all_ny_in_margin.append(all_model_ypred[i])

                print(f'Number of samples in margin is {len(all_nx_in_margin)}')
                self.log.write(f'Number of samples in margin is {len(all_nx_in_margin)} \n')
                all_nx_in_margin = np.array(all_nx_in_margin)
                all_ny_in_margin = np.array(all_ny_in_margin)
                candidates = all_nx_in_margin

                # Measure non-linearity of each cell
                non_lin_of_cell = []
                for i in range(all_nx.shape[0]):
                    non_lin_of_cell.append(self.estimate_dg_2(i, all_nx, all_ny))
                non_lin_of_cell = np.array(non_lin_of_cell)

                # Normalize non-linearity measure
                norm_non_lin_of_cell = non_lin_of_cell / np.sum(non_lin_of_cell)

                print(np.max(non_lin_of_cell))
                print(np.min(non_lin_of_cell))

                non_lin_measure = []
                # # Get non linear measure for each candidate
                # for i in range(candidates.shape[0]):
                #     # find out which cell this candidate belongs to
                #     candidate = candidates[i]
                #     dsts = np.sum(np.square(all_nx - candidate), axis=1)
                #     index = np.argmin(dsts)
                #     non_lin_measure.append(norm_non_lin_of_cell[index])

                non_lin_measure = np.array(non_lin_measure)
                # prioritize samples in margin
                cur_nx = np.array(all_nx)
                temp_nx = []

                to_add = np.min([30, candidates.shape[0]])
                while to_add > 0:
                    ret, ret_id = with_max_priority(candidates, cur_nx, non_lin_measure)
                    if ret.shape[0] > 0:
                        temp_nx.append(ret)
                        new_ny.append(all_ny_in_margin[ret_id])
                        cur_nx = np.concatenate((cur_nx, ret.reshape(1, -1)), axis=0)
                    to_add -= 1
                new_nx += temp_nx

            # # Iteratively pick ones that are at sparse locations
            # all_selected_nx = []
            # cur_nx = np.array(all_nx)
            # to_add = np.min([num_per_iteration, len(nif_samples_index)])
            # if to_add <= 0:
            #     break
            # else:
            #     while to_add > 0:
            #         ret, _ = with_max_distance(nif_all_samples_nx, cur_nx)
            #         if ret.shape[0] > 0:
            #             all_selected_nx.append(ret)
            #             cur_nx = np.concatenate((cur_nx, ret.reshape(1, -1)), axis=0)
            #         to_add -= 1

            # evaluate new samples
            all_selected_ny = np.array(new_ny)
            all_selected_nx = np.array(new_nx)
            all_selected_x = self.reverse_preprocess_x(all_selected_nx)

            # find pareto front of simulated points
            nif_nx = []
            nif_ny = []
            for i in range(all_nx.shape[0]):
                if self.is_non_inferior(all_ny[i], all_ny):
                    nif_nx.append(all_nx[i])
                    nif_ny.append(all_ny[i])

            nif_nx = np.array(nif_nx)
            nif_ny = np.array(nif_ny)

            if plot:
                # plot selected sample and two neighbors
                rnd_index = np.random.randint(0, all_nx.shape[0])
                rnd_nx = all_nx[rnd_index]
                # find two closest neighbor
                dsts = np.linalg.norm(rnd_nx - all_nx, axis=1)
                sorted_idx = np.argsort(dsts)
                nb1_index = sorted_idx[1]
                nb2_index = sorted_idx[2]
                nb3_index = sorted_idx[3]
                nb1_nx = all_nx[nb1_index]
                nb1_ny = all_ny[nb1_index]
                nb2_nx = all_nx[nb2_index]
                nb2_ny = all_ny[nb2_index]
                nb3_nx = all_nx[nb3_index]
                nb3_ny = all_ny[nb3_index]

                px = self.pareto_x()
                npx = self.preprocess_x(px)
                py = self.f(px)

                # 3D Plot in parameter space
                xm1 = 0
                xm2 = 1
                xm3 = 2
                mark_size = 5
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

                # plot in parameter space
                ax.scatter(all_nx[:, xm1], all_nx[:, xm2], all_nx[:, xm3], label='Simulated points', s=mark_size, c='k', alpha=0.5)
                #ax.scatter(candidates[:, m1], candidates[:, m2], candidates[:, m3], label='Samples in the margin', s=mark_size)
                #ax.scatter(all_selected_nx[:, m1], all_selected_nx[:, m2], all_selected_nx[:, m3], label = 'Selected samples', s=mark_size)
                #ax.scatter(nif_nx[:, m1], nif_nx[:, m2], nif_nx[:, m3], label='PF of simulated points', s=mark_size, c='k', alpha=0.1)
                #ax.scatter(all_model_nif_nx[:, m1], all_model_nif_nx[:, m2], all_model_nif_nx[:, m3], label='PF of samples on model', s=mark_size)
                #ax.scatter(npx[:, m1], npx[:, m2], npx[:, m3], label='True PF', s=mark_size, c='y')

                ax.scatter(nb1_nx[xm1], nb1_nx[xm2], nb1_nx[xm2], label='point1')
                ax.scatter(nb2_nx[xm1], nb2_nx[xm2], nb2_nx[xm2], label='point2')
                ax.scatter(nb3_nx[xm1], nb3_nx[xm2], nb3_nx[xm2], label='point3')

                ax.legend()
                if not self.method_x:
                    ax.axes.set_xlim3d(self.min_par[xm1], self.max_par[xm1])
                    ax.axes.set_ylim3d(self.min_par[xm2], self.max_par[xm2])
                    ax.axes.set_zlim3d(self.min_par[xm3], self.max_par[xm3])
                elif self.method_x == 'normalization':
                    ax.axes.set_xlim3d(0, 1)
                    ax.axes.set_ylim3d(0, 1)
                    ax.axes.set_zlim3d(0, 1)
                fig_name = f'param_adaptive_{seed}_{test_bench}_iteration_{iteration}'
                fig.savefig('./figs/' + fig_name)

                # 2D plot in metric space
                ym1 = 0
                ym2 = 1
                fig = plt.figure()
                bx = fig.add_subplot()
                bx.scatter(all_ny[:, ym1], all_ny[:, ym2], label='Simulated points', s=mark_size, c='k', alpha=0.5)
                #bx.scatter(all_ny_in_margin[:, m1], all_ny_in_margin[:, m2], label='Samples in the margin', s=mark_size)
                #bx.scatter(all_selected_ny[:, m1], all_selected_ny[:, m2], label='Selected samples', s=mark_size)
                #bx.scatter(nif_ny[:, m1], nif_ny[:, m2], label='PF of simulated points', s=mark_size)
                #bx.scatter(all_model_nif_ypred[:, m1], all_model_nif_ypred[:, m2],
                #           label='PF of samples', s=mark_size)
                #bx.scatter(py[:, m1], py[:, m2], label='True PF', s=mark_size, c='y')

                bx.scatter(nb1_ny[ym1], nb1_ny[ym2], label='point1')
                bx.scatter(nb2_ny[ym1], nb2_ny[ym2], label='point2')
                bx.scatter(nb3_ny[ym1], nb3_ny[ym2], label='point3')

                bx.legend()
                fig_name = f'metric_adaptive_{seed}_{test_bench}_iteration_{iteration}'
                fig.savefig('./figs/' + fig_name)

                # plot non-linearity measure for each simulated point
                fig = plt.figure()
                cx = fig.add_subplot(projection='3d')
                cx.scatter(all_nx[:,xm1], all_nx[:,xm2], non_lin_of_cell, s=mark_size)
                fig_name = f'nonlinear_adaptive_{seed}_{test_bench}_iteration_{iteration}'
                fig.savefig('./figs/' + fig_name)


            sim_start = time.time()
            all_selected_y = self.f(all_selected_x)
            sim_time = time.time() - sim_start
            total_sim_time += sim_time

            # update existing dataset
            all_x = np.concatenate((all_x, all_selected_x), axis=0)
            all_y = np.concatenate((all_y, all_selected_y), axis=0)
            last_added = list(range(all_x.shape[0] - all_selected_x.shape[0], all_x.shape[0]))

            # save dataset
            if test_bench == 'amp2stage':
                dataset = np.concatenate((all_x, -all_y), axis=1)
            else:
                dataset = np.concatenate((all_x, all_y), axis=1)
            pd.DataFrame(dataset).to_csv(self.data_path, index=False, header=False)

            self.KDTree = None
            iteration += 1

            run_time = time.time() - start_time
            process_time = time.process_time() - process_start_time + total_sim_time
            time_stamps.append([dataset.shape[0], start_run_time + run_time, start_process_time + process_time])
            print(f'Run time is {start_run_time + run_time}')
            print(f'Process time is {start_process_time + process_time}')
            self.log.write(f'Run time is {start_run_time + run_time} \n')
            self.log.write(f'Process time is {start_process_time + process_time} \n')

            pd.DataFrame(np.array(time_stamps)).to_csv(f'./datasets/adaptive_{version}_time_{exp_id}.csv',
                                                       index=False, header=False)

            # stop criteria
            # if all_x.shape[0] >= self.budget:
            if start_run_time + run_time >= 240000:
                # update model
                # save datasets

                pd.DataFrame(np.array(time_stamps)).to_csv(f'./datasets/adaptive_{version}_time_{exp_id}.csv',
                                                           index=False, header=False)

                print('Termination')
                break

        self.log.close()
        return all_x, all_y
