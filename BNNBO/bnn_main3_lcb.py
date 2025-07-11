"""
Implemented by Zhenqi Gao
"""

from myhelper import *

# For reproducibility, Seed may need to change to obtain better result
# Still cannot identically reproduce
np.random.seed(Seed)
#from pymc3.theanof import set_tt_rng, MRG_RandomStreams
#set_tt_rng(MRG_RandomStreams(42))


# BNNBO requires a lonr running time, manually open multiple servers.
cur_parallel = 1 # 0, 1, 2, ..., Repeat/Repeat_bnnbo - 1
idx_list = list(range(cur_parallel*Repeat_bnnbo,(cur_parallel+1) * Repeat_bnnbo))
file_path = file_path_bnnbo


try:
	os.remove(dataset_path)
except:
	pass

# print("############# important info before running#################\n")
# print("bnnbo will be running, repeated time/cur_parallel:%d/%d in this terminal, repeated time in total:%d, max evaluation:%d\n" %(Repeat_bnnbo,cur_parallel,Repeat,Max_eval))
# print("results will be stored in:",file_path," press any key to continue\n")
# input()


def bnnbo(cnt_exp, seed):

	np.random.seed(seed)

	mylambda = gen_mylambda(H_moead,testM_out)

	if mylambda.shape[0]!= N_moead:
		print("error in main: the shape of weight vector is wrong!")
		input()

	# initialize dataset, bnn weight and bias

	# generate initial samples from scratch
	X_train = np.random.uniform(low=0, high=1, size=(N,testM_in)).astype(floatX)
	X_train = np.multiply(X_train, (Upper-Lower).reshape(1,testM_in).repeat(N,axis=0)) + Lower.reshape(1,testM_in).repeat(N,axis=0)
	Y_train,cur_eval = my_eval(X_train,cur_eval=0)

	# # load from existing dataset
	# initial_set = pd.read_csv(f'./datasets/saved/bnnbo_{cnt_exp}.csv', header=None).to_numpy()[:N]
	# pd.DataFrame(initial_set).to_csv(dataset_path, index=False, header=False, mode='w+')
	# X_train = initial_set[:, :testM_in]
	# Y_train = initial_set[:, testM_in:]
	#
	# X_train = normalize_x(X_train, mycircuit)
	# Y_train = normalize_y(Y_train)
	# Y_train = -Y_train
	# cur_eval = X_train.shape[0]

	init_1 = np.random.randn(testM_in, n_hidden_1).astype(floatX) * sigma_w
	init_b_1 = np.random.randn(n_hidden_1).astype(floatX) * sigma_w

	init_2 = np.random.randn(n_hidden_1, n_hidden_2).astype(floatX) * sigma_w
	init_b_2 = np.random.randn(n_hidden_2).astype(floatX) * sigma_w

	init_out = np.random.randn(n_hidden_2, testM_out).astype(floatX) * sigma_w
	init_b_out = np.random.randn(testM_out).astype(floatX) * sigma_w

	bnn_input = theano.shared(X_train)
	bnn_output = theano.shared(Y_train)

	while True:
		### Bayesian Optimization phase 1: build surrogate model using bnn
		neural_network = construct_nn(bnn_input, bnn_output,testM_in,Y_train,init_1=init_1,init_b_1=init_b_1, init_2=init_2, init_b_2=init_b_2, init_out=init_out, init_b_out=init_b_out)
		with neural_network:
			approx = pm.fit(n=Fit_iter, method=pm.ADVI())
			trace = approx.sample(draws=W_sample)

			if cur_eval >= Max_eval:  # no need to adopt phase 2, stop BNNBO

				# # test model accuracy
				# # generate random testing points first(if PS is know, generate random PS points)
				# if Test_bench == 1:
				# 	test_ps = np.random.uniform(low=0, high=1, size=(N_pof,1)).astype(floatX)
				# 	com  = np.zeros((N_pof, testM_in-1))
				# 	test_ps = np.concatenate((test_ps,com),axis=1)
				# 	test_ps = np.multiply(test_ps, (Upper-Lower).reshape(1,testM_in).repeat(N_pof,axis=0)) + Lower.reshape(1,testM_in).repeat(N_pof,axis=0)
				# else:
				# 	test_ps = np.random.uniform(low=0, high=1, size=(N_pof,testM_in)).astype(floatX)
				# 	test_ps = np.multiply(test_ps, (Upper-Lower).reshape(1,testM_in).repeat(N_pof,axis=0)) + Lower.reshape(1,testM_in).repeat(N_pof,axis=0)
				# bnn_input.set_value(test_ps)
				# ppc = pm.sample_posterior_predictive(trace, samples=Pos_sample,progressbar=False)
				# esti_ps = ppc['out'].mean(axis=0)
				# golden_ps,tmp = my_eval(test_ps,0)
				# delta = np.divide(golden_ps - esti_ps, golden_ps)
				# acc_list = np.linalg.norm(delta,axis=1)
				# acc_mean, acc_std = acc_list.mean(),acc_list.std()

				# calculate metric
				ref_point = np.max(Y_train,axis=0)
				hv = cal_hyper(Y_train,ref_point)
				# ind = pg.non_dominated_front_2d(points = Y_train)
				# pof = Y_train[ind,:]

				# display results
				print("\n####Summary for %d-th experiment#####\n Simulation evaluation number is : %d\n BNNBO's hypervolume is: %f\n" %(cnt_exp,cur_eval,hv))
				print("ref_point:",ref_point,"\n")
				if  Test_bench == 1:
					# ZT1 function golden POF
					X_gold = np.random.uniform(low=0, high=1, size=(N_gold,1)).astype(floatX)
					Y_gold = 1 - np.power(X_gold,0.5)
					# figure
					fig, ax = plt.subplots()
					ax.scatter(X_gold,Y_gold,color='b',label='Golden POF')
					ax.scatter(pof[:,0], pof[:,1],color='c',label = 'Trainset true POF by BNNBO')
					ax.scatter(ref_point[0],ref_point[1],color='r',label='Ref point')
					sns.despine()
					ax.legend()
					ax.set(title='POF', xlabel='f1', ylabel='f2')
					return (fig,X_train,Y_train,pof,acc_mean,acc_std)
				elif Test_bench == 2:
					pass
				elif Test_bench == 4:
					return

			### Bayesian Optimization phase 2: define acquisition function and generate new data points
			# initialization lambda is initialized in the beginning
			#mylambda = np.random.uniform(low=0, high=1, size=(N_moead,testM_out)).astype(floatX)
			#mylambda = np.divide(mylambda,np.sum(mylambda,axis=1).reshape(N_moead,1).repeat(testM_out,axis=1))


			# Find neighbors
			dis = euclidean_distances(mylambda,mylambda)
			B = {}
			for i in range(N_moead):
				dis[i,i] = My_big # for fast computation of T_moead neighbors using np.argpartition
				idx = np.argpartition(dis[i,:], T_moead)
				B[str(i)] = idx[:T_moead] 

			# Initialization of parents
			new_gen = floor(N_moead * Ratio_moead)
			inheri = N_moead - new_gen 
			if inheri < X_train.shape[0]:
				X_moead	 = X_train[X_train.shape[0]-inheri:X_train.shape[0],:]
				X_new = np.random.uniform(low=0, high=1, size=(new_gen,testM_in)).astype(floatX)
				X_new = np.multiply(X_new, (Upper-Lower).reshape(1,testM_in).repeat(new_gen,axis=0)) + Lower.reshape(1,testM_in).repeat(new_gen,axis=0)
				X_moead = np.concatenate((X_new,X_moead),axis=0)
				bnn_input.set_value(X_moead)
				ppc = pm.sample_posterior_predictive(trace, samples=Pos_sample,progressbar=False)

				mc_fv = ppc['out']

				fv = ppc['out'].mean(axis=0)
				std = ppc['out'].std(axis=0)
				fv = fv - k_lcb * std

			else:
				X_new = np.random.uniform(low=0, high=1, size=(N_moead-X_train.shape[0],testM_in)).astype(floatX)
				X_new = np.multiply(X_new, (Upper-Lower).reshape(1,testM_in).repeat(N_moead-X_train.shape[0],axis=0)) + Lower.reshape(1,testM_in).repeat(N_moead-X_train.shape[0],axis=0)
				X_moead = np.concatenate((X_new,X_train),axis=0)
				bnn_input.set_value(X_moead)
				ppc = pm.sample_posterior_predictive(trace, samples=Pos_sample,progressbar=False)

				fv = ppc['out'].mean(axis=0)
				std = ppc['out'].std(axis=0)				
				fv = fv - k_lcb * std

			z0 = np.min(fv,axis=0).reshape(1,testM_out)

			# Find the pareto set in parents
			ps, ep ,cur_count = None, None, 0
			for j in range(X_moead.shape[0]):
				if cur_count == 0:
					# initialize pareto set
					ps, ep, cur_count = X_moead[j,:].reshape(1,testM_in),fv[j,:].reshape(1,testM_out),1
				else:
					# update pareto set given new sample
					remain_ind,wrk_domi = [], False
					for i in range(ep.shape[0]):
						#print("fv",fv.shape)
						#print("ep",ep.shape)
						if not pg.pareto_dominance(fv[j,:].flatten(), ep[i,:].flatten()):
							remain_ind.append(i)
						if pg.pareto_dominance(ep[i,:].flatten(),fv[j,:].flatten()):
							wrk_domi = True
					ps, ep , cur_count = ps[remain_ind,:], ep[remain_ind,:], len(remain_ind)
					if wrk_domi == False:
						#print(fv[j,:].shape,ps.shape,ep.shape,)
						ps = np.concatenate((ps,X_moead[j,:].reshape(1,testM_in)),axis=0)
						ep = np.concatenate((ep,fv[j,:].reshape(1,testM_out)),axis=0)
						cur_count += 1

			cnt = 0
			
			while True:	
				flag = False
				for i in range(N_moead):
					# mutation pool
					if np.random.uniform(low=0, high=1, size=(1,1)).astype(floatX)<=Delta_moead:
						P = B[str(i)]
					else:
						P = np.arange(N_moead)
					
					# mutation used neighbor
					muta = np.random.choice(P,size=(1,2))
					y_hat = np.zeros((1,testM_in))
					for j in range(testM_in):
						if np.random.uniform(low=0,high=1,size=(1,1)).astype(floatX)<=CR_moead:
							y_hat[0,j] = X_moead[i,j] + F_moead * (X_moead[muta[0,0],j] - X_moead[muta[0,1],j])
						else:
							y_hat[0,j] = X_moead[i,j]
					child = np.zeros((1,testM_in))
					for j in range(testM_in):
						if np.random.uniform(low=0,high=1,size=(1,1)).astype(floatX)<=pm_moead:
							rand_wrk = np.random.uniform(low=0,high=1,size=(1,1)).astype(floatX)
							if rand_wrk <= 0.5:
								sigma_wrk = np.power(2*rand_wrk,np.divide(1,enta_moead+1)) - 1
							else:
								sigma_wrk = 1 - np.power(2 - 2*rand_wrk,np.divide(1,enta_moead+1))
							child[0,j] = y_hat[0,j] + sigma_wrk * (Upper[j] - Lower[j]) 
						else :
							child[0,j] = y_hat[0,j]
						# repair
						#if child[0,j]>Upper[j] or child[0,j]<Lower[j]:
							#child[0,j] = (Upper[j] - Lower[j]) * np.random.uniform(low=0, high=1, size=(1,1)).astype(floatX) + Lower[j]
						if child[0,j]>Upper[j]:						
							child[0,j] = Upper[j] - myeps[j]
						if child[0,j]<Lower[j]:
							child[0,j] = Lower[j] + myeps[j]				

					# LCB of child
					bnn_input.set_value(child)
					ppc = pm.sample_posterior_predictive(trace, samples=Pos_sample,progressbar=False)
					ppc_out = ppc['out'].mean(axis=1)
					y_child = np.array([ppc_out.mean(axis=0)])
					std = np.array([ppc_out.std(axis=0)])
					y_child = y_child - k_lcb * std

					# update z0
					for j in range(testM_out):
						if z0[0,j] > y_child[0,j]:
							z0[0,j] = y_child[0,j]

					# calculate g(child) and update population
					np.random.shuffle(P)
					for j in P[:nr_moead]:
						tmp1 = np.max(np.multiply(mylambda[j,:],np.absolute(y_child-z0)))
						tmp2 = np.max(np.multiply(mylambda[j,:],np.absolute(fv[j,:]-z0)))
						if tmp1 <= tmp2:
							X_moead[j,:] = child
							fv[j,:] = y_child

					# update pareto set given child
					if cur_count == 0:
						ps, ep,cur_count = child , y_child, 1
					else:
						# remover all points dominated by child, add child, y_child?
						remain_ind,wrk_domi = [], False

						# if any([(y_child.flatten()==ep[ii,:].flatten()).all() for ii in range(ep.shape[0])])==True:
						if np.array([y_child.flatten() == ep[ii, :].flatten() for ii in range(ep.shape[0])]).any():
							continue
						for ii in range(ep.shape[0]):
							if not pg.pareto_dominance(y_child.flatten(), ep[ii,:].flatten()):
								remain_ind.append(ii)
							if pg.pareto_dominance(ep[ii,:].flatten(),y_child.flatten()):
								wrk_domi = True
						ps, ep, cur_count = ps[remain_ind,:], ep[remain_ind,:], len(remain_ind)
						if not wrk_domi:
							ps = np.concatenate((ps,child),axis=0)
							ep = np.concatenate((ep,y_child),axis=0)
							cur_count += 1
					# print("ps",ps)
					# print("ep",ep) #no need print ep, since in lcb case, ep will not stand for pf because std is introduced.

					cnt += 1
					print("cur eval:%d, cur iter:%d, size of ps:%d,ep:%d,cnt:%d\n" % (
					cur_eval, i, ps.shape[0], ep.shape[0], cnt))
					if cnt >= Inner_moead:
						flag = True
						# make sure the total number of simulations is right
						if cur_count + cur_eval >= Max_eval:
							Xp_train = ps[:Max_eval - cur_eval, :]
						else:
							Xp_train = ps

						break


				if flag:
					break

			# prepare for next training
			Yp_train,cur_eval = my_eval(Xp_train,cur_eval)
			X_train = np.concatenate((Xp_train,X_train),axis=0)

			dataset = pd.read_csv(dataset_path, header = None).to_numpy()
			y = dataset[:, testM_in:]
			print(y.shape)
			min_y = np.min(y, axis=0)
			max_y = np.max(y, axis=0)
			Y_train = (y - min_y) / (max_y - min_y)
			Y_train = -Y_train
			# Y_train = np.concatenate((Yp_train,Y_train),axis=0)

			bnn_input.set_value(X_train)
			bnn_output.set_value(Y_train)
			init_1 =  trace['w_in_1'].mean(axis=0)
			init_b_1 =  trace['b_in_1'].mean(axis=0)
			init_2 = trace['w_1_2'].mean(axis=0)
			init_b_2 = trace['b_1_2'].mean(axis=0)
			init_out =  trace['w_2_out'].mean(axis=0)
			init_b_out =  trace['b_2_out'].mean(axis=0)

if __name__ == '__main__':
	# if os.path.exists(file_path):
	# 	print(file_path," alreay exists, please check store dir !\n")
	# 	input() # manually delete the file in dir outside this script or stop this program here.
	# else:
	# 	os.makedirs(file_path)
	# fd = open(file_path+"/info.txt","w")
	# fd.write("Max_eval=%d\nH_moead=%d\nN_moead=%d\nF_moead=%f\nCR_moead=%f\nenta_moead=%f\npm_moead=%f\nnr_moead=%d\nN=%d\ntestM_in=%d\ntestM_out=%d\nSeed=%d\n" %(Max_eval,H_moead,N_moead,F_moead,CR_moead,enta_moead,pm_moead,nr_moead,N,testM_in,testM_out,Seed))
	# fd.close()
	# for cur_exp in range(len(idx_list)):
	# 	cur_file = file_path + "/res_" + str(idx_list[cur_exp])
	# 	if not os.path.exists(cur_file):
	# 		os.makedirs(cur_file)
	# 	try:
	# 		if Test_bench == 1:
	# 			fig,xs,ys,pof,acc_mean,acc_std = bnnbo(cur_exp)
	# 			plt.savefig(cur_file)
	# 			np.savetxt(cur_file+"/x_train.csv",xs,delimiter=",")
	# 			np.savetxt(cur_file+"/y_train.csv",ys,delimiter=",")
	# 			np.savetxt(cur_file+"/pof.csv",pof,delimiter=",")
	# 			fd = open(cur_file+"/model_acc.txt","w")
	# 			fd.write("acc_mean=%f\nacc_std=%f\n" %(acc_mean,acc_std))
	# 			fd.close()
	# 		elif Test_bench == 2:
	# 			pass
	# 	except:
	# 		os.remove(cur_file)

	# fig, xs, ys, pof, acc_mean, acc_std = bnnbo(0)
	bnnbo(0, 10)




	plt.show()
