from myhelper import *

# For reproducibility, Seed may need to change to obtain better result
# Still cannot identically reproduce
np.random.seed(Seed)
#from pymc3.theanof import set_tt_rng, MRG_RandomStreams
#set_tt_rng(MRG_RandomStreams(42))


# Because using the same myconst.py, the following lines may need uncomment when debugging
#Max_eval = 300

file_path = file_path_moead

try:
	os.remove(dataset_path)
except:
	pass




# print("############# important info before running#################\n")
# print("MOEAD will be running, repeated time:%d, max evaluation:%d\n" %(Repeat,Max_eval))
# print("results will be stored in:",file_path," press any key to continue\n")
# input()

def moead(cnt_exp, seed):
	np.random.seed(seed)

	mylambda = gen_mylambda(MO_H,testM_out)
	if mylambda.shape[0]!= MO_N:
		print("error in main: the shape of weight vector is wrong!")
		input()
	dis = euclidean_distances(mylambda,mylambda)
	B,cur_eval = {},0
	for i in range(MO_N):
		dis[i,i] = My_big # for fast computation of T_moead neighbors using np.argpartition
		idx = np.argpartition(dis[i,:], T_moead)
		B[str(i)] = idx[:T_moead]

	X_moead = np.random.uniform(low=0, high=1, size=(MO_N,testM_in)).astype(floatX)
	X_moead = np.multiply(X_moead, (Upper-Lower).reshape(1,testM_in).repeat(MO_N,axis=0)) + Lower.reshape(1,testM_in).repeat(MO_N,axis=0)
	fv,cur_eval = my_eval(X_moead,cur_eval=0)
	z0 = np.min(fv,axis=0).reshape(1,testM_out)
	while True:
		flag = False 
		for i in range(MO_N):
				# mutation pool
				if np.random.uniform(low=0, high=1, size=(1,1)).astype(floatX)<=Delta_moead:
					P = B[str(i)]
				else:
					P = np.arange(MO_N)
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
					# if child[0,j]>Upper[j] or child[0,j]<Lower[j]:
					# 	child[0,j] = (Upper[j] - Lower[j]) * np.random.uniform(low=0, high=1, size=(1,1)).astype(floatX) + Lower[j]
					if child[0,j]>Upper[j]:						
						child[0,j] = Upper[j] - myeps[j]
					if child[0,j]<Lower[j]:
						child[0,j] = Lower[j] + myeps[j]				
				# update reference point z

				y_child,cur_eval = my_eval(child,cur_eval)
				# maximum simulation is reached and thus break
				if cur_eval>=Max_eval:
					flag = True
					break
				for j in range(testM_out):
					if z0[0,j] > y_child[0,j]:
						z0[0,j] = y_child[0,j]
				# update neighbor
				np.random.shuffle(P)
				for j in P[:nr_moead]:
					tmp1 = np.max(np.multiply(mylambda[j,:],np.absolute(y_child-z0)))
					tmp2 = np.max(np.multiply(mylambda[j,:],np.absolute(fv[j,:]-z0)))
					if tmp1 <= tmp2:
						X_moead[j,:] = child
						fv[j,:] = y_child
				print("cur eval:%d, cur iter/MO_N:%d/%d\n" %(cur_eval,i,MO_N))
		if flag==True:
			# calculate metric
			ref_point = np.max(fv,axis=0)
			hv = cal_hyper(fv,ref_point)
			# display results
			print("\n####Summary for %d-th experiment#####\n Simulation evaluation number is : %d\n MOEAD's hypervolume is: %f\n" %(cnt_exp,cur_eval,hv))
			print ("ref_point:", ref_point,"\n")

			if  Test_bench == 1:
				ind = pg.non_dominated_front_2d(points=fv)
				pof = fv[ind, :]
				# ZT1 function golden POF
				X_gold = np.random.uniform(low=0, high=1, size=(N_gold,1)).astype(floatX)
				Y_gold = 1 - np.power(X_gold,0.5)
				fig, ax = plt.subplots()
				ax.scatter(X_gold,Y_gold,color='b',label='Golden POF')
				ax.scatter(ref_point[0],ref_point[1],color='r',label='Ref point')
				ax.scatter(pof[:,0], pof[:,1],color='c',label = 'POF by MOEAD')
				sns.despine()
				ax.legend()
				ax.set(title='POF', xlabel='f1', ylabel='f2')
				if Drawfig==True: 
					plt.show()
				return (fig,X_moead,fv,pof)
			elif Test_bench == 2:
				pass
			else:
				return

if __name__ == '__main__':

	moead(0, 10)

	# if os.path.exists(file_path):
	# 	print(file_path," alreay exists, please check store dir !\n")
	# 	input() # manually delete files in dir outside this script or stop this program here.
	# else:
	# 	os.makedirs(file_path)
	# fd = open(file_path+"/info.txt","w")
	# fd.write("Max_eval=%d\nMO_H=%d\nMO_N=%d\nF_moead=%f\nCR_moead=%f\nenta_moead=%f\npm_moead=%f\nnr_moead=%d\nN=%d\ntestM_in=%d\ntestM_out=%d\nSeed=%d\n" %(Max_eval,MO_H,MO_N,F_moead,CR_moead,enta_moead,pm_moead,nr_moead,N,testM_in,testM_out,Seed))
	# fd.close()
	# for cur_exp in range(Repeat):
	# 	cur_file = file_path + "/res_" + str(cur_exp)
	# 	if not os.path.exists(cur_file):
	# 		os.makedirs(cur_file)
	# 	try:
	# 		if Test_bench == 1:
	# 			fig,xs,ys,pof = moead(cur_exp)# moead no model. thus no acc.
	# 			plt.savefig(cur_file)
	# 			np.savetxt(cur_file+"/x_train.csv",xs,delimiter=",")
	# 			np.savetxt(cur_file+"/y_train.csv",ys,delimiter=",")
	# 			np.savetxt(cur_file+"/pof.csv",pof,delimiter=",")
	# 		elif Test_bench == 2:
	# 			pass
	# 	except:
	# 		os.remove(cur_file)
