from myhelper import *

# For reproducibility, Seed may need to change to obtain better result
# Still cannot identically reproduce
np.random.seed(Seed)
#from pymc3.theanof import set_tt_rng, MRG_RandomStreams
#set_tt_rng(MRG_RandomStreams(42))


file_path = file_path_mobo

try:
    os.remove(dataset_path)
except:
    pass



# print("############# important info before running#################\n")
# print("MOBO will be running, repeated time:%d, max evaluation:%d\n" %(Repeat,Max_eval))
# print("results will be stored in:",file_path," press any key to continue\n")
# input()

def mobo(cur_exp, seed):
    # randomly initialization
    np.random.seed(seed)

    # Initial samples
    # # random sampling from scratch
    # xs       = np.random.uniform(Lower, Upper, (N, testM_in))
    # ys,cur_eval = my_eval(xs,cur_eval=0)

    # load from existing dataset
    initial_set = pd.read_csv(f'./datasets/saved/mobo_{cur_exp}.csv').to_numpy()
    pd.DataFrame(initial_set).to_csv(dataset_path, index=False, header=False, mode='w+')

    xs = initial_set[:, :testM_in]
    xs = normalize_x(xs, mycircuit)
    ys = initial_set[:, testM_in:]
    ys = normalize_y(ys)
    cur_eval = initial_set.shape[0]


    #print("xs",xs,"ys",ys)
    while True:
        gp_model = ()
        for i in range(testM_out):
            cur_y = ys[:, i].reshape(ys.shape[0], 1)
            m = GPy.models.GPRegression(xs, cur_y, GPy.kern.RBF(input_dim = testM_in, ARD = True))
            m.kern.variance = np.var(cur_y)
            m.lengthscale = np.std(xs,0)
            m.likelihood.variance = 1e-2 * np.var(cur_y)
            m.optimize()
            gp_model = gp_model + (m,)
        def lcb(x):
            res = list (range(testM_out))
            for i in range(testM_out):
                py, ps = gp_model[i].predict(np.array([x]))
                res[i] = py[0,0] - LW_k * np.sqrt(ps[0,0])
            return res
        problem = Problem(testM_in,testM_out)
        for i in range(testM_in):
            problem.types[i] = Real(Lower[i], Upper[i])
        problem.function = lcb
        algorithm        = NSGAII(problem, population = LW_size)
        algorithm.run(LW_times)
        optimized = algorithm.result
        idx = np.random.permutation(len(optimized))[0]
        new_x = np.array(optimized[idx].variables)
        new_y,cur_eval = my_eval(np.array([new_x]),cur_eval)
        print("experiment:%d\n" %cur_exp)
        print ("new generated point:", new_x, "cur_eval:%d\n" %cur_eval)
        # no need for next iteration, maximum simulation reached
        if cur_eval >= Max_eval:
            # calculate metric
            ref_point = np.max(ys,axis=0)
            hv = cal_hyper(ys,ref_point)
            # # test model accuracy here, generate random testing points first(if PS is know, generate random PS points)
            # if Test_bench == 1:
            #     test_ps = np.random.uniform(low=0, high=1, size=(N_pof,1)).astype(floatX)
            #     com  = np.zeros((N_pof, testM_in-1))
            #     test_ps = np.concatenate((test_ps,com),axis=1)
            #     test_ps = np.multiply(test_ps, (Upper-Lower).reshape(1,testM_in).repeat(N_pof,axis=0)) + Lower.reshape(1,testM_in).repeat(N_pof,axis=0)
            # else:
            #     test_ps = np.random.uniform(low=0, high=1, size=(N_pof,testM_in)).astype(floatX)
            #     test_ps = np.multiply(test_ps, (Upper-Lower).reshape(1,testM_in).repeat(N_pof,axis=0)) + Lower.reshape(1,testM_in).repeat(N_pof,axis=0)
            # esti_ps = np.zeros((N_pof,testM_out))
            # for i in range(testM_out):
            #       y,s = gp_model[i].predict(test_ps)
            #       esti_ps[:,i] = y.flatten()
            #
            # golden_ps,tmp = my_eval(test_ps,0)
            # delta = np.divide(golden_ps - esti_ps, golden_ps)
            # acc_list = np.linalg.norm(delta,axis=1)
            # acc_mean, acc_std = acc_list.mean(),acc_list.std()
            # ind = pg.non_dominated_front_2d(points = ys)
            # pof = ys[ind,:]

            print("\n####Summary for %d-th experiment#####\n Simulation evaluation number is : %d\n MOBO's hypervolume is: %f\n" %(cur_exp,cur_eval,hv))
            # print ('Model average error: %f, with std: %f\n' %(acc_mean, acc_std))
            print("ref_point:",ref_point,"\n")
            if  Test_bench == 1:
                # ZT1 function golden POF
                X_gold = np.random.uniform(low=0, high=1, size=(N_gold,1)).astype(floatX)
                Y_gold = 1 - np.power(X_gold,0.5)
                fig, ax = plt.subplots()
                ax.scatter(X_gold,Y_gold,color='b',label='Golden POF')
                ax.scatter(pof[:,0], pof[:,1],color='c',label = 'POF by MOBO')
                ax.scatter(ref_point[0],ref_point[1],color='r',label='Ref point')
                sns.despine()
                ax.legend()
                ax.set(title='POF', xlabel='f1', ylabel='f2')
                if Drawfig == True:
                    plt.show()
                return (fig,xs,ys,pof,acc_mean,acc_std)
            elif Test_bench == 2:
                pass

            break

        # update training set
        xs    = np.concatenate((xs, new_x.reshape(1, new_x.size)), axis=0)
        dataset = pd.read_csv(dataset_path, header=None).to_numpy()
        y = dataset[:, testM_in:]
        min_y = np.min(y, axis=0)
        max_y = np.max(y, axis=0)
        ys = (y - min_y) / (max_y - min_y)
        ys = -ys
        # ys    = np.concatenate((ys, new_y.reshape(1, testM_out)), axis=0)

if __name__ == '__main__':

    mobo(0, 0)

    # if os.path.exists(file_path):
    #     print(file_path," alreay exists, please check store dir !\n")
    #     input() # manually delete files in dir outside this script or stop this program here.
    # else:
    #     os.mkdir(file_path)
    # fd = open(file_path+"/info.txt","w")
    # fd.write("Max_eval=%d\nLW_k=%d\nLW_size=%d\nLW_times=%d\nN=%d\ntestM_in=%d\nSeed=%d\n" %(Max_eval,LW_k,LW_size,LW_times,N,testM_in,Seed))
    # fd.close()
    # for cur_exp in range(Repeat):
    #     cur_file = file_path + "/res_" + str(cur_exp)
    #     if not os.path.exists(cur_file):
    #         os.makedirs(cur_file)
    #     try:
    #         if Test_bench == 1:
    #             fig,xs,ys,pof,acc_mean,acc_std = mobo(cur_exp)
    #             plt.savefig(cur_file)
    #             np.savetxt(cur_file+"/x_train.csv",xs,delimiter=",")
    #             np.savetxt(cur_file+"/y_train.csv",ys,delimiter=",")
    #             np.savetxt(cur_file+"/pof.csv",pof,delimiter=",")
    #             fd = open(cur_file+"/model_acc.txt","w")
    #             fd.write("acc_mean=%f\nacc_std=%f\n" %(acc_mean,acc_std))
    #             fd.close()
    #         elif Test_bench == 2:
    #             pass
    #     except:
    #         os.remove(cur_file)
