#!/usr/bin/env python
# coding: utf-8

# ## Define the plain vanilla gradient descent function

# In[ ]:


def gd_fun(n_iter,fn_grad1,fn_grad2,fn_loss,eta,tol,x1,x2):
    loss_path = [fn_loss(x1,x2)]
    x1_path = [x1]
    x2_path =[x2]
    y_change = fn_loss(x1,x2)
    iter_num = 0
    
    
    g1 = fn_grad1(x1,x2)
    g2 = fn_grad2(x1,x2)
    
    while (y_change > tol and iter_num < n_iter) :
        tmp_x1 = x1 - eta * fn_grad1(x1,x2)
        tmp_x2 = x2 - eta * fn_grad1(x1,x2)
        tmp_y = fn_loss(tmp_x1,tmp_x2)
        y_change = np.absolute(tmp_y - fn_loss(x1,x2))
        x1 = tmp_x1
        x2 = tmp_x2
        
        loss_path.append(tmp_y)
        x1_path.append(x1)
        x2_path.append(x2)
        iter_num += 1
    return loss_path,x1_path,x2_path 


# ## Define the momentum descent function

# In[1]:


def momentum_fun(n_iter,fn_grad1,fn_grad2,fn_loss,eta,tol,x1,x2):
    loss_path = [fn_loss(x1,x2)]
    x1_path = [x1]
    x2_path =[x2]
    y_change = fn_loss(x1,x2)
    iter_num = 0
    moment = 0.9
    g1 = fn_grad1(x1,x2)
    g2 = fn_grad2(x1,x2)
    v1 = 0
    v2 = 0
    while (y_change > tol and iter_num < n_iter) :
        
        v1 = eta * fn_grad1(x1,x2) + moment*v1
        v2 = eta * fn_grad1(x1,x2) + moment*v2
        tmp_x1 = x1 - v1 
        tmp_x2 = x2 - v2 
        tmp_y = fn_loss(tmp_x1,tmp_x2)
        y_change = np.absolute(tmp_y - fn_loss(x1,x2))
        x1 = tmp_x1
        x2 = tmp_x2
        
        loss_path.append(tmp_y)
        x1_path.append(x1)
        x2_path.append(x2)
        iter_num += 1
    return loss_path,x1_path,x2_path 


# ## Define the Nesterov’s Accelerated Gradient (NAG) 

# In[2]:


def NAG_fun(n_iter,fn_grad1,fn_grad2,fn_loss,eta,tol,x1,x2):
    loss_path = [fn_loss(x1,x2)]
    x1_path = [x1]
    x2_path =[x2]
    y_change = fn_loss(x1,x2)
    iter_num = 0
    moment = 0.9
    g1 = fn_grad1(x1,x2)
    g2 = fn_grad2(x1,x2)
    v1 = 0
    v2 = 0
    while (y_change > tol and iter_num < n_iter) :
        
        v1 = moment*v1 + eta* fn_grad1(x1-moment*v1,x2)
        v2 = moment*v2 + eta* fn_grad2(x1,x2-moment*v2)
        tmp_x1 = x1 - v1 
        tmp_x2 = x2 - v2 
        tmp_y = fn_loss(tmp_x1,tmp_x2)
        y_change = np.absolute(tmp_y - fn_loss(x1,x2))
        x1 = tmp_x1
        x2 = tmp_x2
        
        loss_path.append(tmp_y)
        x1_path.append(x1)
        x2_path.append(x2)
        iter_num += 1
    return loss_path,x1_path,x2_path 

