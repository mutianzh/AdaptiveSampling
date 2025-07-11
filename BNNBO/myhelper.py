from myconst import *

import sys
import pandas as pd
sys.path.insert(0, '/home/mutianzh/PycharmProjects/AMPSE2/GlobalLibrary')
from spectreIOlib import TestSpice, Netlists
from netlist_database import Amp_twostage3
mycircuit = Amp_twostage3(paralleling=True, max_parallel_sim=16)


def normalize_x(x, mycircuit):
    x = np.array(x)
    xrange = (mycircuit.maxpar - mycircuit.minpar) / mycircuit.stppar + 1
    return ((x - mycircuit.minpar) / mycircuit.stppar) / xrange

def normalize_y(y):
    y = np.array(y)
    return (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

# interface for outside algorithms
def my_eval(X,cur_eval):
    if Test_bench == 1:
        return ZT1(X,cur_eval)
    elif Test_bench == 2:
        return opamp(X,cur_eval)
    elif Test_bench == 3:
        return charge_pump(X,cur_eval)
    elif Test_bench == 4:
        return two_stage_amp(X, cur_eval)

def two_stage_amp(X, cur_eval):
    """
    :param X: np array of normalzied circuit parameters
    :param cur_eval: total number of simulations has been run so far
    :return: np array of normalized metrics

    """
    print(dataset_path)
    all_y = []
    xrange = (mycircuit.maxpar - mycircuit.minpar) / mycircuit.stppar + 1
    all_x = np.multiply(np.floor(np.multiply(X, xrange)), mycircuit.stppar) + mycircuit.minpar
    in_dim = len(mycircuit.parname)
    out_put_selection = [4,5,6]

    # True simulation
    for x in all_x:
        y = mycircuit.wholerun_normal(x, parallelid=parallel_index)
        all_y.append(y)
    all_y = np.array(all_y)[:,out_put_selection]

    # # dummy simulation
    # all_y = np.random.uniform(size=(all_x.shape[0],3))


    if not os.path.exists(dataset_path):
        dataset = np.concatenate((all_x, all_y), axis=1)

    else:
        dataset = pd.read_csv(dataset_path, header=None).to_numpy()
        new_data = np.concatenate((all_x, all_y), axis=1)
        dataset = np.concatenate((dataset, new_data), axis=0)

    pd.DataFrame(dataset).to_csv(dataset_path, index=False, header=False, mode='w+')

    min_y = np.min(dataset[:, in_dim:], axis=0)
    max_y = np.max(dataset[:, in_dim:], axis=0)

    all_ny = (all_y - min_y) / (max_y - min_y)
    new_cur_eval = X.shape[0] + cur_eval

    return -all_ny, new_cur_eval

def opamp(X,cur_eval):
    corner = ["head_tt.sp","head_ss.sp","head_ff.sp","head_fs.sp","head_sf.sp"] # five corners worst case
    out = testM_out # or out=3 (srr,srf are discarded)
    res = np.zeros((X.shape[0],5))
    for i in range(X.shape[0]):
        fd=open("param.txt","w")
        content=".param w1=" + str(X[i,0]) +"\n.param w3=" + str(X[i,1]) +"\n.param w5=" + str(X[i,2]) +"\n.param w7=" + str(X[i,3])+ "\n.param w9=" + str(X[i,4])
        fd.write(content)
        fd.close()
        gain_list,ugf_list,pm_list,srr_list,srf_list=[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]
        for j in range(len(corner)):
            os.system("cat " + corner[j] + " opamp_tail.sp > cur_sim.sp")
            os.system("hspice64 cur_sim.sp -o ./tmp_res")
            gain_list[j],ugf_list[j],pm_list[j],srr_list[j],srf_list[j]= my_reg(["gain","ugf","pm","srr","srf"],"./tmp_res.lis")
            os.system("rm ./tmp_res.* ./cur_sim.sp")
        res[i,0], res[i,1], res[i,2], res[i,3], res[i,4]= min(gain_list),min(ugf_list),min(pm_list),min(srr_list),min(srf_list)
    return (-1) * res[:,0:out],cur_eval+X.shape[0] # since our formulation support minimization

def charge_pump(X,cur_eval):
    corner = ["head_tt.sp","head_ss.sp","head_ff.sp","head_fs.sp","head_sf.sp"] # five corners worst case
    out = testM_out # or out=3 (maximum is 5)
    res = np.zeros((X.shape[0],5))
    for i in range(X.shape[0]):
        fd=open("param.txt","w")
        content=".param vdd33=" + str(X[i,0]) +"\n.param lp4=" + str(X[i,1]) +"\n.param wp4=" + str(X[i,2]) +"\n.param ln4=" + str(X[i,3])+ "\n.param wn4=" + str(X[i,4])
        fd.write(content)
        fd.close()
        diff1,diff2,diff3,diff4,div1,div2=[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]
        for j in range(len(corner)):
            os.system("cat " + corner[j] + " chargepump_tail.sp > cur_sim.sp")
            os.system("hspice64 cur_sim.sp -o ./tmp_res")
            diff1[j],diff2[j],diff3[j],diff4[j],div1[j],div2[j] = my_reg(["diff_upup","diff_updn","diff_loup","diff_lodn","deviation1","deviation2"],"./tmp_res.lis")
            os.system("rm ./tmp_res.* ./cur_sim.sp")
        res[i,0], res[i,1], res[i,2], res[i,3], res[i,4] = max(diff1),max(diff2),max(diff3),max(diff4),max(div1)+max(div2)
    return res[:,0:out],cur_eval+X.shape[0] # since our formulation support minimization


def my_reg(pat_list,file_name):
    res = list (range(len(pat_list)))
    with open(file_name,"r") as fd:
        for line in fd.readlines():
            for j in range(len(pat_list)):
                cur_line = line.strip()
                if cur_line.startswith(pat_list[j]+"="):
                    try:
                        res[j]= eval(cur_line.lstrip(pat_list[j]+"=").strip())
                    except:
                        print("Error in simulation, refer to "+file_name+" for details\n")
                        input()
    return res

# test bench function from ref MOEAD, not used in experiment
def F2(X,cur_eval):
    abs_j1 = int ((X.shape[1]-3)/2)+1
    abs_j2 = int ((X.shape[1]-2)/2)+1
    wrk1 = 6*np.pi*X[:,0].reshape(X.shape[0],1).repeat(abs_j1,axis=1) + np.divide(np.pi,X.shape[1]) * np.array(range(3,X.shape[1]+1,2)).reshape(1,abs_j1).repeat(X.shape[0],axis=0)
    wrk2 = np.power(X[:,2:X.shape[1]:2]-np.sin(wrk1),2)
    f1 = X[:,0].reshape(X.shape[0],1) + np.divide(2,abs_j1) * np.sum(wrk2,axis=1).reshape(X.shape[0],1)
    wrk3 = 6*np.pi*X[:,0].reshape(X.shape[0],1).repeat(abs_j1,axis=1) + np.divide(np.pi,X.shape[1]) * np.array(range(2,X.shape[1]+1,2)).reshape(1,abs_j2).repeat(X.shape[0],axis=0)
    wrk4 = np.power(X[:,1:X.shape[1]:2]-np.sin(wrk1),2)
    f2 = 1 - np.power(X[:,0],0.5).reshape(X.shape[0],1) + np.divide(2,abs_j2) * np.sum(wrk4,axis=1).reshape(X.shape[0],1)
    return np.concatenate((f1.reshape(X.shape[0],1),f2.reshape(X.shape[0],1)),axis=1),cur_eval+f1.shape[0]


# Test bench function ZT1: [X.shape[1]*1 vector]-> [2*1 vector]
#                          input variable in [0,1], POF f1=(1-f2)^2
#                          f1<=g
# X [ N * X.shape[1] matrix]: each row is a sample
# res [N * 2 matrix]: each row is an output. 
def ZT1(X,cur_eval):
    f1 = X[:,0]
    g = 1 + np.multiply(9,np.divide(np.sum(X[:,1:],axis=1),X.shape[1]-1))
    h = 1 - np.power(np.divide(f1,g),0.5)
    f2 = np.multiply(g,h)
    return np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1)),axis=1),cur_eval+f1.shape[0]

def gen_mylambda(H,m):
    res = list (combinations(range(0,H+m-1),m-1)) # no need to sort
    res2 = np.zeros((len(res),m-1))
    com = np.zeros((len(res),1))
    for i in range(len(res)):
        for j in range(len(res[0])-1,0,-1):
            res2[i,j] = ( res[i][j] - res[i][j-1] - 1 ) / H
            com[i,0] += res2[i,j]
        res2[i,0] = res[i][0] / H
        com[i,0] += res2[i,0]
        com[i,0] = 1 - com[i,0]
    res3 = np.concatenate((res2,com),axis=1)
    return res3

# change design variable to input features of nueral network
# des [N*1 vector]: input design variables
# method [string]: define expanding method,'x','x^2','x^3'
# res [return variable]:  if method=='x', [N*1 vector]. 
#                         if method=='x+x^2', [N*1+(C_N^2+N^2) vector]. 
#                         if method=='x+x^3', [N*1+(C_N^2+N^2)+(C_N^3) vector]. 
def des_to_fea(des,method='x'):
    if des.shape[0]==1 and des.shape[1]!=1:
        print('warning in [des_to_fea]: des should be a column vector'
            ', autaomatically reshape\n')
        des = des.reshape(-1,1)
    if method=='x':
        return des
    elif method=='x+x^2':
        res = des
        for i1 in range(des.size):
            for i2 in range(i1,des.size):
                    tmp = des[i1]*des[i2]
                    res = np.append(res,[tmp],axis=0)
        return res
    elif method =='x+x^2+x^3':
        res = des_to_fea(des,'x+x^2')
        for i1 in range(des.size):
            for i2 in range(i1,des.size):
                for i3 in range(i2,des.size):
                    tmp = des[i1]*des[i2]*des[i3]
                    res = np.append(res,[tmp],axis=0)
        return res
    elif method == 'testN':
        if (des.size>1):
            print('warning in [des_to_fea]: in test case, des should have only one element\n')
        res = np.power(des[0],range(testN+1)).reshape(-1,1)
        return res

def construct_nn(bnn_input, bnn_output, input_dim, Y_train, init_1, init_b_1, init_2, init_b_2, init_out, init_b_out):
    # Initialize random weights between each layer
    # init_1 = np.random.randn(input_dim, N_hidden).astype(floatX) * sigma_w
    # init_out = np.random.randn(N_hidden,testM_out).astype(floatX) * sigma_w
    # init_b_1 = np.random.randn(N_hidden).astype(floatX) * sigma_w
    # init_b_out = np.random.randn(testM_out).astype(floatX) * sigma_w

    with pm.Model() as bnn:
        # Weights from input to 1st hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=sigma_w,
                                 shape=(input_dim, n_hidden_1),
                                 testval=init_1)
        b_in_1 = pm.Normal('b_in_1', 0, sd=sigma_w,
                                 shape=(n_hidden_1,),
                                 testval=init_b_1)

        # Weights from 1st to 2nd hidden layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=sigma_w,
                                 shape=(n_hidden_1, n_hidden_2),
                                 testval=init_2)
        b_1_2 = pm.Normal('b_1_2', 0, sd=sigma_w,
                           shape=(n_hidden_2,),
                           testval=init_b_2)

        # Weights from 2nd hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=sigma_w,
                                  shape=(n_hidden_2,testM_out),
                                  testval=init_out)
        b_2_out = pm.Normal('b_2_out', 0, sd=sigma_w,
                                  shape=(testM_out,),
                                  testval=init_b_out)

        # Build bayesian neural network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(bnn_input,
                                         weights_in_1) + b_in_1)
        act_2 = pm.math.tanh(pm.math.dot(act_1,
                                         weights_1_2) + b_1_2)

        act_out = pm.math.dot(act_2, weights_2_out) + b_2_out

        # # With relu activation function
        # act_1 = tt.nnet.relu(pm.math.dot(bnn_input,
        #                                  weights_in_1) + b_in_1)
        # act_2 = tt.nnet.relu(pm.math.dot(act_1,
        #                                  weights_1_2) + b_1_2)
        #
        # act_out = pm.math.dot(act_2, weights_2_out) + b_2_out

        out = pm.Normal('out',act_out,sd=sigma_out,
                           observed=bnn_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return bnn

# calculate hyper-volume with respect to a given ref_point
# X [N*testM_out matrix]: objective values 
# ref_point [testM_out*1 vector]: reference point, required for calculating hyper-volume
def cal_hyper(X,ref_point):
    hv = pg.hypervolume(X)
    return hv.compute(ref_point)

# for testing
if __name__ == '__main__':
    # np.random.seed(0)
    # testing_N = 10
    # X_train = np.random.uniform(low=0, high=1, size=(testing_N,testM_in))#.astype(floatX)
    # # X_train = np.multiply(X_train, (Upper-Lower).reshape(1,testM_in).repeat(testing_N,axis=0))+Lower.reshape(1,testM_in).repeat(testing_N,axis=0)
    # Y_train,cur_eval = my_eval(X_train,cur_eval=0)
    # print('X_train',X_train)
    # print('Y_train', Y_train)
    # print('min:',np.min(Y_train,axis=0))
    # print('max:',np.max(Y_train,axis=0))

    mylambda = gen_mylambda(MO_H, testM_out)
    print(mylambda)
