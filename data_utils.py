import os
import torch
import numpy as np
from pc_encoders import *

# Directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))


def mapfunc(x):
    """
    Identity mapping (placeholder for transformations).
    """
    return x


def minmax(data, min_val=0.0, max_val=1.0):
    """
    Normalize data to [0, 1] range.
    """
    return (data - min_val) / (max_val - min_val)


def inv_minmax(normalized, min_val=0.0, max_val=1.0):
    """
    Inverse of minmax normalization.
    """
    return normalized * (max_val - min_val) + min_val


def sample_unique_indices(n_train, N, M):
    """
    Generate a tensor of unique random indices for each sample in n_train.
    """
    indices = torch.stack([torch.randperm(N)[:M] for _ in range(n_train)])  # Shape: (n_train, M)
    return indices
def sample_tensors(tensor_A, tensor_B, M):
    """
    Args:
        tensor_A: Tensor of shape (n_train, N, 2)
        tensor_B: Tensor of shape (n_train, N)
        M: Number of points to sample from N dimension.

    Returns:
        sampled_A: Tensor of shape (n_train, M, 2)
        sampled_B: Tensor of shape (n_train, M)
    """
    n_train, N, _ = tensor_A.shape
    
    # Get unique random indices for each sample
    indices = sample_unique_indices(n_train, N, M).unsqueeze(-1)  # Shape: (n_train, M, 1)

    # Expand indices for advanced indexing
    expanded_indices_A = indices.expand(-1, -1, tensor_A.shape[-1])  # Shape: (n_train, M, 2)

    # Sample from tensors using advanced indexing
    sampled_A = torch.gather(tensor_A, dim=1, index=expanded_indices_A)  # Shape: (n_train, M, 2)
    sampled_B = torch.gather(tensor_B, dim=1, index=indices.squeeze(-1))  # Shape: (n_train, M)

    return sampled_A, sampled_B


def sample_tensors_variable_M(tensor_A, tensor_B, min_frac=0.5, max_frac=0.8):
    """
    Args:
        tensor_A: Tensor of shape (n_train, N, 2)
        tensor_B: Tensor of shape (n_train, N)
        min_frac, max_frac: floats defining the fraction-interval [min_frac, max_frac]
            from which each sample's retained-point-count fraction is drawn.

    Returns:
        sampled_A: list of length n_train of Tensors with shapes (M_i, 2)
        sampled_B: list of length n_train of Tensors with shapes (M_i,)
    """
    n_train, N, _ = tensor_A.shape
    # draw fractions and compute per-sample M_i
    fracs = torch.rand(n_train) * (max_frac - min_frac) + min_frac    # shape: (n_train,)
    Ms = (fracs * N).floor().long()                                    # shape: (n_train,)

    sampled_A = []
    sampled_B = []
    for i, M_i in enumerate(Ms):
        # random unique indices for the i-th sample
        idx = torch.randperm(N)[:M_i]          # shape: (M_i,)
        sampled_A.append(tensor_A[i, idx])
        sampled_B.append(tensor_B[i, idx])

    return sampled_A, sampled_B




def get_data_naca(ntrain=200, s=400, dropout=1.0, test_do=1.0):
    """
    Load and encode NACA data for training and testing.

    Args:
        ntrain (int): Number of training samples.
        s (int): Grid resolution (mesh size).

    Returns:
        Tuple of tensors: (X_train, A_train, U_train,
                            X_test, A_test, U_test,
                            input, ref)
    """
    # Paths and raw data
    data_path = os.path.join(script_dir, "data", "naca_data.pt")
    raw = torch.load(data_path)

    # ---------------------
    # Training set
    # ---------------------
    X_pc = raw["X"][:ntrain].float()  # (ntrain, N, 2)
    U_pc = raw["U"][:ntrain].float()  # (ntrain, N)

    # Random subsampling
    N = X_pc.shape[1]
    k = int(dropout * N)
    idx = torch.randperm(N)[:k]
    X_pc = X_pc[:, idx, :]
    U_pc = U_pc[:, idx]

    # Normalize coordinates to [-1, 1]
    x_min, x_max, y_min, y_max = -0.4, 1.26, -0.45, 0.45
    x = (X_pc[..., 0] - x_min) / (x_max - x_min)
    y = (X_pc[..., 1] - y_min) / (y_max - y_min)
    x = x.unsqueeze(-1) * 2 - 1
    y = y.unsqueeze(-1) * 2 - 1
    input_pc = torch.cat([x, y], dim=-1).unsqueeze(1)
    ref_pc = U_pc.unsqueeze(1).unsqueeze(-1)

    # Encode to grids
    grid_i = torch.zeros(ntrain, 1, s, s)
    grid_o = geometry_encoder(grid_i.clone(), input_pc, step='bilinear')
    grid_u = response_encoder(grid_i.clone(), input_pc, ref_pc, step='bilinear')

    A_train = grid_o[:, 0].unsqueeze(-1)
    U_train = grid_u[:, 0].unsqueeze(-1)

    # Create mesh coordinates
    x_mesh = torch.linspace(x_min, x_max, s)
    y_mesh = torch.linspace(y_min, y_max, s)
    Xm, Ym = torch.meshgrid(x_mesh, y_mesh, indexing='xy')
    X_train = torch.stack((Xm, Ym), dim=-1)
    X_train = X_train.unsqueeze(0).repeat(ntrain, 1, 1, 1)

    # ---------------------
    # Test set
    # ---------------------
    X_pc = raw["X"][-100:].float()
    U_pc = raw["U"][-100:].float()

    N_test = X_pc.shape[1]
    k_test = int(test_do * N_test)
    idx_test = torch.randperm(N_test)[:k_test]
    X_pc = X_pc[:, idx_test, :]
    U_pc = U_pc[:, idx_test]

    x = (X_pc[..., 0] - x_min) / (x_max - x_min)
    y = (X_pc[..., 1] - y_min) / (y_max - y_min)
    x = x.unsqueeze(-1) * 2 - 1
    y = y.unsqueeze(-1) * 2 - 1
    input_pc = torch.cat([x, y], dim=-1).unsqueeze(1)
    ref_pc = U_pc.unsqueeze(1).unsqueeze(-1)

    grid_i = torch.zeros(100, 1, s, s)
    grid_o = geometry_encoder(grid_i.clone(), input_pc, step='bilinear')
    grid_u = response_encoder(grid_i.clone(), input_pc, ref_pc, step='bilinear')

    A_test = grid_o[:, 0].unsqueeze(-1)
    U_test = grid_u[:, 0].unsqueeze(-1)

    X_test = X_train[:100]

    return X_train, A_train, U_train, X_test, A_test, U_test, input_pc, ref_pc








def get_data_elas(ntrain = 200 , s = 400 , dropout = 1.0 ,test_do = 1.0,noisestd =0.):

 
    ########## Training data set

    n_train = ntrain
    n_test = 100

 
    data_X_path = script_dir + "/data/elas_data.pt"

    X_pointcloud = torch.load(data_X_path)["X"][:n_train , ...].to(torch.float32)#.numpy().astype(np.float64)  #(n_train , 972 , 2)
    U_pointcloud = torch.load(data_X_path)["U"][:n_train , ...].to(torch.float32)*1e-2#.numpy().astype(np.float64) #(n_train , 972)
    T = torch.load(data_X_path)["T"][:n_train , ...].to(torch.float32) #(ntrain,42)
    T = T.unsqueeze(-1).repeat(1 , 1 ,42).unsqueeze(1) #(n_train , 42 , 42)
    T_train = torch.nn.functional.interpolate(T.clone() , [s , s]).permute(0 , 2 , 3 , 1) #(n_train , 1 , s , s) ->(ntrain , s , s , 1)
    
    num_samples = int(dropout*X_pointcloud.shape[1])
    indices = torch.randperm(X_pointcloud.shape[1])[:num_samples]
    
    X_pointcloud = X_pointcloud[: , indices , :]
    U_pointcloud = U_pointcloud[: , indices]

    noise = torch.randn_like(X_pointcloud).to(torch.float32)*noisestd
    X_pointcloud = X_pointcloud+noise

    


    x_min , x_max , y_min , y_max = [0., 1.0 , 0 , 1.]
    grid_i = torch.zeros([n_train , 1 , s , s])
    x = (X_pointcloud[... , 0]-x_min)/(x_max - x_min)
    y = (X_pointcloud[... , 1]-y_min)/(y_max - y_min)
    x = x.unsqueeze(-1)*2 -1
    y = y.unsqueeze(-1)*2 -1
    input = torch.cat([x,y] , -1).unsqueeze(1) 
    input = torch.clamp(input , min =-1. , max = 1.)
    ref = U_pointcloud.unsqueeze(1).unsqueeze(-1)# (n_train ,1, 2820,1)


    grid_o = geometry_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , step = 'bilinear') #torch.Size([n_train, 1, 50, 50])
    grid_u = response_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , ref.to(torch.float32) , step = 'bilinear')
 
    A_train = torch.cat([grid_o[: , 0 , ...].unsqueeze(-1) , T_train] , -1)#(ntrain , s , s , 2)
    U_train = grid_u[: , 0 , ...].unsqueeze(-1)  #(ntrain , s , s ,1)
    


    x_mesh =torch.linspace(x_min, x_max, s, dtype=torch.float32)
    y_mesh = torch.linspace(y_min, y_max, s, dtype=torch.float32)
    Xmesh , Ymesh = torch.meshgrid([x_mesh , y_mesh])
    X_train = torch.cat([Xmesh.unsqueeze(-1) , Ymesh.unsqueeze(-1)] , dim = -1).unsqueeze(0).repeat(ntrain , 1 , 1 , 1).permute(0 , 2 , 1 , 3) #(ntrain , s , s , 2)
 

 
    ########## Test data set

    X_pointcloud = torch.load(data_X_path)["X"][-n_test: , ...].to(torch.float32)
    U_pointcloud = torch.load(data_X_path)["U"][-n_test: , ...].to(torch.float32)*1e-2
    T = torch.load(data_X_path)["T"][-n_test: , ...].to(torch.float32)
    T = T.unsqueeze(-1).repeat(1 , 1 ,42).unsqueeze(1) #(n_train , 42 , 42)
    T_test = torch.nn.functional.interpolate(T.clone() , [s , s]).permute(0 , 2 , 3 , 1)
    
   
    num_samples = int(test_do*X_pointcloud.shape[1])
    indices = torch.randperm(X_pointcloud.shape[1])[:num_samples]
    
    X_pointcloud = X_pointcloud[: , indices , :]
    U_pointcloud = U_pointcloud[: , indices]


    grid_i = torch.zeros([n_test , 1 ,s , s])
    x = (X_pointcloud[... , 0]-x_min)/(x_max - x_min)
    y = (X_pointcloud[... , 1]-y_min)/(y_max - y_min)
    x = x.unsqueeze(-1)*2 -1
    y = y.unsqueeze(-1)*2 -1
    input = torch.cat([x,y] , -1).unsqueeze(1) 
    ref = U_pointcloud.unsqueeze(1).unsqueeze(-1)

    grid_o = geometry_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , step = 'bilinear') #torch.Size([n_test, 1, 50, 50])
    grid_u = response_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , ref.to(torch.float32) , step = 'bilinear')
 
    A_test = torch.cat([grid_o[: , 0 , ...].unsqueeze(-1) , T_test] , -1)
    U_test = grid_u[: , 0 , ...].unsqueeze(-1)


    x_mesh =torch.linspace(x_min, x_max, s, dtype=torch.float32)
    y_mesh = torch.linspace(y_min, y_max, s, dtype=torch.float32)
    Xmesh , Ymesh = torch.meshgrid([x_mesh , y_mesh])
    X_test = torch.cat([Xmesh.unsqueeze(-1) , Ymesh.unsqueeze(-1)] , dim = -1).unsqueeze(0).repeat(n_test , 1 , 1 , 1).permute(0 , 2 , 1 , 3) #(ntrain , s , s , 2)
 
    A_train = mapfunc(A_train)
    U_train = mapfunc(U_train)

    A_test = mapfunc(A_test)
    U_test = mapfunc(U_test) 
   

    return X_train , A_train , U_train  ,\
    X_test , A_test , U_test , input , ref 



def get_data_darcy(ntrain = 200 , s = 120 ):

    
 
    
    data_X_path = os.path.join(script_dir, "data/darcy_data2.pt") 
    n_train = ntrain
    n_test = 100
    X_pointclouds = torch.load(data_X_path)["X"]
    U_pointclouds = torch.load(data_X_path)["U"]*10
    Ks = torch.load(data_X_path)["K"]

    x_min , x_max , y_min , y_max = [0., 1.0 , 0. , 1.]

 
 



    ########## Training data set
    X_pointcloud = X_pointclouds[:n_train , ...].to(torch.float32)#.numpy().astype(np.float64)  #(n_train , 972 , 2)
    U_pointcloud =U_pointclouds [:n_train , ...].to(torch.float32)#.numpy().astype(np.float64) #(n_train , 972)
    K = Ks[:n_train , ...].unsqueeze(1).to(torch.float32) #(ntrain,241 , 241)
    
    K_train = torch.nn.functional.interpolate(K.clone() , [s , s]).permute(0 , 2 , 3 , 1) #(n_train , 1 , s , s) ->(ntrain , s , s , 1)
    



    grid_i = torch.zeros([n_train , 1 , s , s])
    x = (X_pointcloud[... , 0]-x_min)/(x_max - x_min)
    y = (X_pointcloud[... , 1]-y_min)/(y_max - y_min)
    x = x.unsqueeze(-1)*2 -1
    y = y.unsqueeze(-1)*2 -1
    input = torch.cat([x,y] , -1).unsqueeze(1) 
    ref = U_pointcloud.unsqueeze(1).unsqueeze(-1)# (n_train ,1, 2820,1)


    grid_o = geometry_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , step = 'bilinear') #torch.Size([n_train, 1, 50, 50])
    grid_u = response_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , ref.to(torch.float32) , step = 'bilinear')
 
    A_train = torch.cat([grid_o[: , 0 , ...].unsqueeze(-1) , K_train] , -1)
    U_train = grid_u[: , 0 , ...].unsqueeze(-1) 
    


    x_mesh =torch.linspace(x_min, x_max, s, dtype=torch.float32)
    y_mesh = torch.linspace(y_min, y_max, s, dtype=torch.float32)
    Xmesh , Ymesh = torch.meshgrid([x_mesh , y_mesh])
    X_train = torch.cat([Xmesh.unsqueeze(-1) , Ymesh.unsqueeze(-1)] , dim = -1).unsqueeze(0).repeat(ntrain , 1 , 1 , 1).permute(0 , 2 , 1 , 3) #(ntrain , s , s , 2)
 

 
    ########## Test data set

    X_pointcloud = X_pointclouds[-n_test: , ...].to(torch.float32)
    U_pointcloud = U_pointclouds[-n_test: , ...].to(torch.float32)
    K = Ks[-n_test:, ...].unsqueeze(1).to(torch.float32) 
    
    K_test = torch.nn.functional.interpolate(K.clone() , [s , s]).permute(0 , 2 , 3 , 1) #(n_train , 1 , s , s) ->(ntrain , s , s , 1)
    
   
    #X_pointcloud = X_pointcloud+noise

    #x_min , x_max , y_min , y_max = [0., 1.0 , 0. , 1.]
    grid_i = torch.zeros([n_test , 1 ,s , s])
    x = (X_pointcloud[... , 0]-x_min)/(x_max - x_min)
    y = (X_pointcloud[... , 1]-y_min)/(y_max - y_min)
    x = x.unsqueeze(-1)*2 -1
    y = y.unsqueeze(-1)*2 -1
    input = torch.cat([x,y] , -1).unsqueeze(1) #(n_test , 1 , N , 2)
    ref = U_pointcloud.unsqueeze(1).unsqueeze(-1)

    grid_o = geometry_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , step = 'bilinear') #torch.Size([n_test, 1, 50, 50])
    grid_u = response_encoder(grid_i.clone().to(torch.float32) , input.to(torch.float32) , ref.to(torch.float32) , step = 'bilinear')
 
    A_test = torch.cat([grid_o[: , 0 , ...].unsqueeze(-1) , K_test] , -1)
    U_test = grid_u[: , 0 , ...].unsqueeze(-1)



    x_mesh =torch.linspace(x_min, x_max, s, dtype=torch.float32)
    y_mesh = torch.linspace(y_min, y_max, s, dtype=torch.float32)
    Xmesh , Ymesh = torch.meshgrid([x_mesh , y_mesh])
    X_test = torch.cat([Xmesh.unsqueeze(-1) , Ymesh.unsqueeze(-1)] , dim = -1).unsqueeze(0).repeat(n_test , 1 , 1 , 1).permute(0 , 2 , 1 , 3) #(ntrain , s , s , 2)
 

   

    return X_train , A_train , U_train ,\
    X_test , A_test , U_test , input , ref 



    
def get_data_circles(ntrain = 200 , s = 100 ):

    n_train = ntrain
    n_test = 100

    res = s
    x_min , x_max , y_min , y_max = [0. , 8. , 0 , 8.] 

    ########## Training data set
 
 
    data_X_path = script_dir + "/data/ns2d_data.pt"

    DATA = torch.load(data_X_path)

    u_data = DATA["U"]



    Us = []
    for u in u_data:
        u[:,0] = minmax(u[:,0] , -0.05, 3.21)
        u[:,1] =  minmax(u[:,1] , -1.90, 1.62)
        u[:,2] = minmax(u[:,2] , -4.93, 73.33)
        Us.append(torch.from_numpy(u))
    
    x_data =  DATA["X"] 

    A_trains = []
    U_trains = []

    for i in range (n_train):

        X = torch.from_numpy(x_data[i]) 
        U = Us[i] 

        grid_i = torch.zeros([1 , 1 , res , res])

        x = (X[... , 0]-x_min)/(x_max - x_min)
        y = (X[... , 1]-y_min)/(y_max - y_min)

        x = x*2-1
        y = y*2-1
        input = torch.from_numpy(np.concatenate([x.unsqueeze(-1) , y.unsqueeze(-1)] , -1)).unsqueeze(0).unsqueeze(0) #(1 , 1 . N_i , 2)
        ref = U.unsqueeze(1)

        
        grid_o = geometry_encoder(grid_i.clone() , input.to(torch.float32) , step = 'bilinear') #(1 , 1 , 100 , 100)
  

        grid_us = []
        for j in range(ref.shape[-1]):
            grid_us.append(response_encoder(grid_i.clone() , input.to(torch.float32) , ref[... , j:j+1].to(torch.float32) , step = 'bilinear'))
        grid_u = torch.cat(grid_us , 1)         


    
        A_train = grid_o.permute(0 , 2 , 3  , 1)  #(1 , s  ,s,1)
        U_train = grid_u.permute(0 , 2 , 3 , 1)  #(1 , s  ,s,3)

        A_trains.append(A_train)
        U_trains.append(U_train)
    
    A_train = torch.cat(A_trains , 0)
    U_train = torch.cat(U_trains , 0)


    # ########## Test data set

    data_X_path = script_dir + "/data/ns2d_data_test.pt"


    DATA = torch.load(data_X_path)

    u_data = DATA["U"]
    c_data = [torch.from_numpy(c[0]) for c in DATA["C"]]




    Us = []
    for u in u_data:
        u[:,0] = minmax(u[:,0] , -0.05, 3.21)
        u[:,1] =  minmax(u[:,1] , -1.90, 1.62)
        u[:,2] = minmax(u[:,2] ,   -4.93, 73.33)
        Us.append(torch.from_numpy(u))
    
    x_data =  DATA["X"]  

    A_tests = []
    U_tests = []
    input_list = []
    ref_list = []
    REC = []

    for i in range (n_test):

        X = torch.from_numpy(x_data[i]) 
        U = Us[i] 


        grid_i = torch.zeros([1 , 1 , res , res])

        x = (X[... , 0]-x_min)/(x_max - x_min)
        y = (X[... , 1]-y_min)/(y_max - y_min)

        x = x*2-1
        y = y*2-1
        input = torch.from_numpy(np.concatenate([x.unsqueeze(-1) , y.unsqueeze(-1)] , -1)).unsqueeze(0).unsqueeze(0)
        ref = U.unsqueeze(1)


        grid_o = geometry_encoder(grid_i.clone() , input.to(torch.float32) , step = 'bilinear') #(1 , 1 , 100 , 100)

        grid_us = []
        for j in range(ref.shape[-1]):
            grid_us.append(response_encoder(grid_i.clone() , input.to(torch.float32) , ref[... , j:j+1].to(torch.float32) , step = 'bilinear'))
        grid_u = torch.cat(grid_us , 1)      

    
        A_test = grid_o.permute(0 , 2 , 3  , 1) 
        U_test = grid_u.permute(0 , 2 , 3 , 1)  

        A_tests.append(A_test)
        U_tests.append(U_test)


        ref[... ,0]  = inv_minmax(ref[...,0] ,  -0.05, 3.21)
        ref[...,1] = inv_minmax(ref[...,1] , -1.90, 1.62)
        ref[...,2] = inv_minmax(ref[...,2] ,  -4.93, 73.33)




        input_list.append(input)
        ref_list.append(ref)
    
    A_test = torch.cat(A_tests , 0)
    U_test = torch.cat(U_tests , 0)


    return A_train , U_train , A_test , U_test , input_list , ref_list , c_data


def get_data_maze(ntrain = 1600 , s = 120):


    n_train = ntrain
    n_test = 200

    res = s
    x_min , x_max , y_min , y_max = [-1 ,1 , -1 , 1] 

    ########## Training data set
 
    data_X_path = script_dir + "/data/maze_data.pt"

    DATA = torch.load(data_X_path)

    au_data = DATA["AU"] 
    
    Us = []
    As = []
    for au in au_data:
        au[: , 0:1] = minmax(au[: , 0:1] ,0 , 11.5)
        au[: , 10:11] = minmax(au[: , 10:11] ,-4.0 , 4.0)

        a = au[: , 0:1]
        u = au[: , [10]]


        As.append(a)
        Us.append(u)
    
    x_data =  DATA["X"]  #(1000 , n_i , 2)

    A_trains = []
    U_trains = []

    for i in range (n_train):

        X = x_data[i] #(n_i , 2)
        U = Us[i] ##(n_i , 3)
        A = As[i]

        grid_i = torch.zeros([1 , 1 , res , res])

        x = (X[... , 0]-x_min)/(x_max - x_min)
        y = (X[... , 1]-y_min)/(y_max - y_min)

        x = x*2-1
        y = y*2-1
        input = torch.cat([x.unsqueeze(-1) , y.unsqueeze(-1)] , -1).unsqueeze(0).unsqueeze(0) #(1 , 1 . N_i , 2)
        ref = torch.cat([A , U] , -1).unsqueeze(1)

        
        grid_o = geometry_encoder(grid_i.clone() , input.to(torch.float32) , step = 'bilinear') 


        grid_us = []
        for j in range(ref.shape[-1]):
            grid_us.append(response_encoder(grid_i.clone() , input.to(torch.float32) , ref[... , j:j+1].to(torch.float32) , step = 'bilinear'))
        grid_u = torch.cat(grid_us , 1)      


    
        A_train = torch.cat([grid_o , grid_u[: , 0:1 , ...]] , 1).permute(0 , 2 , 3  , 1)  #(1 , s  ,s,2)
        U_train = grid_u[: , 1: , ...].permute(0 , 2 , 3 , 1)  #(1 , s  ,s,3)

        A_trains.append(A_train)
        U_trains.append(U_train)
    
    A_train = torch.cat(A_trains , 0)
    U_train = torch.cat(U_trains , 0)


    # ########## Test data set

    A_tests = []
    U_tests = []
    ref_list = []
    input_list = []

    for i in range(1600,1600+n_test):

        X = x_data[i] 
        U = Us[i] 
        A = As[i]


        grid_i = torch.zeros([1 , 1 , res , res])

        x = (X[... , 0]-x_min)/(x_max - x_min)
        y = (X[... , 1]-y_min)/(y_max - y_min)

        x = x*2-1
        y = y*2-1
        input = torch.cat([x.unsqueeze(-1) , y.unsqueeze(-1)] , -1).unsqueeze(0).unsqueeze(0) 
        ref = torch.cat([A , U] , -1).unsqueeze(1)

        
        grid_o = geometry_encoder(grid_i.clone() , input.to(torch.float32) , step = 'bilinear') 


        grid_us = []
        for j in range(ref.shape[-1]):
            grid_us.append(response_encoder(grid_i.clone() , input.to(torch.float32) , ref[... , j:j+1].to(torch.float32) , step = 'bilinear'))
        grid_u = torch.cat(grid_us , 1)    


    
        A_test = torch.cat([grid_o , grid_u[: , 0:1 , ...]] , 1).permute(0 , 2 , 3  , 1)  #(1 , s  ,s,2)
        U_test = grid_u[: , 1: , ...].permute(0 , 2 , 3 , 1)  #(1 , s  ,s,3)


        A_tests.append(A_test)
        U_tests.append(U_test)
        input_list.append(input)


        ref[... ,1:]  = inv_minmax(ref[...,1:] ,  -4.0, 4.0)

        ref_list.append(ref[... ,1:] )


    
    A_test = torch.cat(A_tests , 0)
    U_test = torch.cat(U_tests , 0)


    return A_train , U_train , A_test , U_test , input_list , ref_list


def get_data_solid2(ntrain = 200 , s = 400 ):


 
    ########## Training data set

    n_train = ntrain
    n_test = 500

 
 
    data_X_path = script_dir + "/data/solid_data5000.pt"
    #data_U_path = r"D:\PMACS\naca\U.pt"

    DATA = torch.load(data_X_path)

    u_data = DATA["U"][:2000+n_test]

    # Threshold/clip the values to specific ranges
    u_data[:,:,0] = np.clip(u_data[:,:,0], 0., 310.)      # vM stress
    u_data[:,:,1] = np.clip(u_data[:,:,1], -0.008, 0.005) # UX
    u_data[:,:,2] = np.clip(u_data[:,:,2], -0.0005, 0.02) # UY
    u_data[:,:,3] = np.clip(u_data[:,:,3], -0.005, 0.01)  # UZ

    u_data[:,:,0] = minmax(u_data[:,:,0] , 0., 310.)
    u_data[:,:,1] = minmax(u_data[:,:,1] , -0.008, 0.005)
    u_data[:,:,2] = minmax(u_data[:,:,2] , -0.0005, 0.02)
    u_data[:,:,3] = minmax(u_data[:,:,3] , -0.005, 0.01)
    



    X = DATA["X"][:n_train , ...]     #(ntrain , N , 3) 

    U = u_data[:n_train , ... ]



    print(np.max(u_data[...,0]) , np.max(u_data[...,1]) , np.max(u_data[...,2]) , np.max(u_data[...,3]))
    print(np.min(u_data[...,0]) , np.min(u_data[...,1]) , np.min(u_data[...,2]) , np.min(u_data[...,3]))
    #U[...,1:] +=5
    e_train = DATA["ey"][:n_train] #(ntrain)

    X_max_train = torch.from_numpy(X.max(axis = 1)) /10




    B = X.shape[0]
    res = s
    x_min , x_max , y_min , y_max , z_min , z_max = np.min(X[... , 0] , axis = 1) , np.max(X[... , 0], axis = 1) ,\
        np.min(X[... , 1], axis = 1) , np.max(X[... , 1], axis = 1) , np.min(X[... , 2], axis = 1) , np.max(X[... , 2], axis = 1) 
    grid_i = torch.zeros([B , 1 , res , res , res] , dtype = torch.float32)

    x = (X[... , 0]-x_min[:,np.newaxis])/(x_max[:,np.newaxis] - x_min[:,np.newaxis])
    y = (X[... , 1]-x_min[:,np.newaxis])/(y_max[:,np.newaxis] - y_min[:,np.newaxis])
    z = (X[... , 2]-x_min[:,np.newaxis])/(z_max[:,np.newaxis] - z_min[:,np.newaxis])

    x = x*2-1
    y = y*2-1
    z = z*2-1
    input = torch.from_numpy(np.concatenate([x[...,np.newaxis] , y[...,np.newaxis] , z[...,np.newaxis]] , -1)).unsqueeze(1).unsqueeze(1)
    ref = torch.from_numpy(U).unsqueeze(1).unsqueeze(1)

    
    grid_o = geometry_encoder3d(grid_i.clone() , input.to(torch.float32) , step = 'trilinear')

    grid_us = []
    for i in range(ref.shape[-1]):
        grid_us.append(response_encoder3d(grid_i.clone() , input.to(torch.float32) , ref[... , i:i+1].to(torch.float32) , step = 'trilinear'))
    grid_u = torch.cat(grid_us , 1)

    E_train =  torch.from_numpy(e_train).view(ntrain , 1 , 1 , 1 , 1).repeat(1 , s , s , s ,1)
 
    A_train = grid_o.permute(0 , 2 , 3 , 4 , 1)  #(ntrain , s , s ,s,1)
    U_train = grid_u.permute(0 , 2 , 3 , 4 , 1)  #(ntrain , s , s ,s,1)

    A_train = torch.cat([A_train , E_train] , dim = -1) #(ntrain , s , s ,s,2)

    ref[... , 0] = inv_minmax(ref[...,0] , 0. , 310.)
    ref[...,1] = inv_minmax(ref[...,1] , -0.008, 0.005)
    ref[...,2] = inv_minmax(ref[...,2] , -0.0005, 0.02)
    ref[...,3] = inv_minmax(ref[...,3] , -0.005, 0.01)

    input_train = input
    ref_train = ref



 
    ########## Test data set

    X = DATA["X"][2000:2000+n_test, ...]
    U = u_data[2000:2000+n_test: , ... ]
    
    e_test = DATA["ey"][2000:2000+n_test:] #(ntest)

    X_max_test = torch.from_numpy(X.max(axis = 1))/10

    x_min , x_max , y_min , y_max , z_min , z_max = np.min(X[... , 0] , axis = 1) , np.max(X[... , 0], axis = 1) ,\
    np.min(X[... , 1], axis = 1) , np.max(X[... , 1], axis = 1) , np.min(X[... , 2], axis = 1) , np.max(X[... , 2], axis = 1) 


    E_test =  torch.from_numpy(e_test).view(n_test , 1 , 1 , 1 , 1).repeat(1 , s , s , s ,1)

    B = X.shape[0]
    res = s
    grid_i = torch.zeros([B , 1 , res , res , res] , dtype = torch.float32)

    x = (X[... , 0]-x_min[:,np.newaxis])/(x_max[:,np.newaxis] - x_min[:,np.newaxis])
    y = (X[... , 1]-x_min[:,np.newaxis])/(y_max[:,np.newaxis] - y_min[:,np.newaxis])
    z = (X[... , 2]-x_min[:,np.newaxis])/(z_max[:,np.newaxis] - z_min[:,np.newaxis])

    x = x*2-1
    y = y*2-1
    z = z*2-1
    input = torch.from_numpy(np.concatenate([x[...,np.newaxis] , y[...,np.newaxis] , z[...,np.newaxis]] , -1)).unsqueeze(1).unsqueeze(1)
    ref = torch.from_numpy(U).unsqueeze(1).unsqueeze(1)

    grid_o = geometry_encoder3d(grid_i.clone() , input.to(torch.float32) , step = 'trilinear')
    grid_us = []
    for i in range(ref.shape[-1]):
        grid_us.append(response_encoder3d(grid_i.clone() , input.to(torch.float32) , ref[... , i:i+1].to(torch.float32) , step = 'trilinear'))
    grid_u = torch.cat(grid_us , 1)#torch.Size([500, 4, 70, 70, 70])
 
    A_test= grid_o.permute(0 , 2 , 3 , 4 , 1)  #(ntrain , s , s ,s,1)
    U_test = grid_u.permute(0 , 2 , 3 , 4 , 1)  #(ntrain , s , s ,s,1)

    A_test = torch.cat([A_test , E_test] , -1) #(ntrain , s , s ,s,2)

   
    ref[... , 0] = inv_minmax(ref[...,0] , 0. , 310.)
    ref[...,1] = inv_minmax(ref[...,1] , -0.008, 0.005)
    ref[...,2] = inv_minmax(ref[...,2] , -0.0005, 0.02)
    ref[...,3] = inv_minmax(ref[...,3] , -0.005, 0.01)

    return A_train , U_train  , A_test , U_test , input , ref , X_max_train , X_max_test  , input_train , ref_train
    


    
    

    



