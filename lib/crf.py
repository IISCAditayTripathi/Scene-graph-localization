import torch
import numpy as np

def max_product_belief_propogation(factors, num_vars, num_states, num_iters=30):
	msg_fv = {} # f->v messages (dictionary)
	msg_vf = {} # v->f messages (dictionary)
	ne_var = [[] for i in range(num_vars)] # neighboring factors of variables (list of list)

	# set messages to zero; determine factors neighboring each variable
	for [f_idx,f] in enumerate(factors):
	    # print(f['vars'])
	    for v_idx in f['vars']:
	        msg_fv[(f_idx,v_idx)] = torch.zeros(num_states).cuda() # factor->variable message
	        msg_vf[(v_idx,f_idx)] = torch.zeros(num_states).cuda() # variable->factor message
	        ne_var[v_idx].append(f_idx) # factors neighboring variable v_idx

	# status message
	# print("Messages initialized!")

	# run inference
	for it in range(num_iters):
	  
	    # for all factor-to-variable messages do
	    for [key,msg] in msg_fv.items():
	        
	        # shortcuts to variables
	        f_idx = key[0] # factor (source)
	        v_idx = key[1] # variable (target)
	        f_vars = factors[f_idx]['vars'] # variables connected to factor
	        f_vals = factors[f_idx]['vals'] # vector/matrix of factor values 
	        # unary factor-to-variable message
	        if len(f_vars)==1:
	            msg_fv[(f_idx,v_idx)] = f_vals

	        # pairwise factor-to-variable-message
	        else:

	            # if target variable is first variable of factor
	            if v_idx==f_vars[0]:
	                a = msg_vf[(f_vars[1],f_idx)]
	                
	                # msg_in = np.tile(msg_vf[(f_vars[1],f_idx)],(num_states,1))
	                msg_in = msg_vf[(f_vars[1],f_idx)].unsqueeze(0)
	                msg_fv[(f_idx,v_idx)], _ = (f_vals+msg_in).max(1) # max over columns

	            # if target variable is second variable of factor
	            else:
	                # msg_in = np.tile(msg_vf[(f_vars[0],f_idx)],(num_states,1))
	                msg_in = msg_vf[(f_vars[0],f_idx)].unsqueeze(0)
	                msg_fv[(f_idx,v_idx)], _ = (f_vals+msg_in.permute(1,0)).max(0) # max over rows
	                
	        # normalize
	        msg_fv[(f_idx,v_idx)] = msg_fv[(f_idx,v_idx)] - torch.mean(msg_fv[(f_idx,v_idx)])

	    # for all variable-to-factor messages do
	    for [key,msg] in msg_vf.items():
	        
	        # shortcuts to variables
	        v_idx = key[0] # variable (source)
	        f_idx = key[1] # factor (target)

	        # add messages from all factors send to this variable (except target factor)
	        # and send the result to the target factor
	        msg_vf[(v_idx,f_idx)] = torch.zeros(num_states).cuda()
	        for f_idx2 in ne_var[v_idx]:
	            if f_idx2 != f_idx:
	                msg_vf[(v_idx,f_idx)] += msg_fv[(f_idx2,v_idx)]
	                
	        # normalize
	        msg_vf[(v_idx,f_idx)] = msg_vf[(v_idx,f_idx)] - torch.mean(msg_vf[(v_idx,f_idx)])
	        
	# calculate max-marginals (num_vars x num_states matrix)
	max_marginals = torch.zeros([num_vars,num_states]).cuda()
	for v_idx in range(num_vars):
	    
	    # add messages from all factors sent to this variable
	    max_marginals[v_idx] = torch.zeros(num_states).cuda()
	    for f_idx in ne_var[v_idx]:
	        max_marginals[v_idx] += msg_fv[(f_idx,v_idx)]
	    #print max_marginals[v_idx]

	# get MAP solution
	values, map_est = torch.max(max_marginals,axis=1)

	return values, map_est, max_marginals



def sum_product_belief_propogation(factors, num_vars, num_states, num_iters=10):
	msg_fv = {} # f->v messages (dictionary)
	msg_vf = {} # v->f messages (dictionary)
	ne_var = [[] for i in range(num_vars)] # neighboring factors of variables (list of list)

	# set messages to zero; determine factors neighboring each variable
	for [f_idx,f] in enumerate(factors):
	    # print(f['vars'])
	    for v_idx in f['vars']:
	        msg_fv[(f_idx,v_idx)] = torch.zeros(num_states).cuda() # factor->variable message
	        msg_vf[(v_idx,f_idx)] = torch.zeros(num_states).cuda() # variable->factor message
	        ne_var[v_idx].append(f_idx) # factors neighboring variable v_idx

	# status message
	# print("Messages initialized!")

	# run inference
	for it in range(num_iters):
	  
	    # for all factor-to-variable messages do
	    for [key,msg] in msg_fv.items():
	        
	        # shortcuts to variables
	        f_idx = key[0] # factor (source)
	        v_idx = key[1] # variable (target)
	        f_vars = factors[f_idx]['vars'] # variables connected to factor
	        f_vals = factors[f_idx]['vals'] # vector/matrix of factor values 
	        # unary factor-to-variable message
	        if len(f_vars)==1:
	            msg_fv[(f_idx,v_idx)] = f_vals

	        # pairwise factor-to-variable-message
	        else:

	            # if target variable is first variable of factor
	            if v_idx==f_vars[0]:
	                a = msg_vf[(f_vars[1],f_idx)]
	                
	                # msg_in = np.tile(msg_vf[(f_vars[1],f_idx)],(num_states,1))
	                msg_in = msg_vf[(f_vars[1],f_idx)].unsqueeze(0)
	                # msg_fv[(f_idx,v_idx)], _ = (f_vals+msg_in).max(1) # max over columns
	                msg_fv[(f_idx,v_idx)] = torch.log((torch.exp(f_vals+msg_in)).sum(1)) # max over columns

	            # if target variable is second variable of factor
	            else:
	                # msg_in = np.tile(msg_vf[(f_vars[0],f_idx)],(num_states,1))
	                msg_in = msg_vf[(f_vars[0],f_idx)].unsqueeze(0)
	                # msg_fv[(f_idx,v_idx)], _ = (f_vals+msg_in.permute(1,0)).max(0) # max over rows
	                msg_fv[(f_idx,v_idx)] = torch.log((torch.exp(f_vals+msg_in.permute(1,0))).sum(0)) # max over rows
	                
	        # normalize
	        msg_fv[(f_idx,v_idx)] = msg_fv[(f_idx,v_idx)] - torch.mean(msg_fv[(f_idx,v_idx)])

	    # for all variable-to-factor messages do
	    for [key,msg] in msg_vf.items():
	        
	        # shortcuts to variables
	        v_idx = key[0] # variable (source)
	        f_idx = key[1] # factor (target)

	        # add messages from all factors send to this variable (except target factor)
	        # and send the result to the target factor
	        msg_vf[(v_idx,f_idx)] = torch.zeros(num_states).cuda()
	        for f_idx2 in ne_var[v_idx]:
	            if f_idx2 != f_idx:
	                msg_vf[(v_idx,f_idx)] += msg_fv[(f_idx2,v_idx)]
	                
	        # normalize
	        msg_vf[(v_idx,f_idx)] = msg_vf[(v_idx,f_idx)] - torch.mean(msg_vf[(v_idx,f_idx)])
	        
	# calculate max-marginals (num_vars x num_states matrix)
	max_marginals = torch.zeros([num_vars,num_states]).cuda()
	for v_idx in range(num_vars):
	    
	    # add messages from all factors sent to this variable
	    max_marginals[v_idx] = torch.zeros(num_states).cuda()
	    for f_idx in ne_var[v_idx]:
	        max_marginals[v_idx] += msg_fv[(f_idx,v_idx)]
	    #print max_marginals[v_idx]

	# get Posterioir marginals
	return max_marginals