import torch

def geometry_encoder(input, grid, step='bilinear', offset=False):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW) representing the grid (vertices).
        grid: A torch.Tensor of dimension (N, H, W, 2) representing scattered points in normalized coordinates.
    Return:
        torch.Tensor: The updated input grid after accumulating contributions from scattered points.
    '''
    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    # Step function definition (bilinear or cosine)
    if step == 'bilinear':
        step_f = lambda x: x
    elif step == 'cosine':
        step_f = lambda x: 0.5 * (1 - torch.cos(torch.pi * x))
    else:
        raise NotImplementedError

    # Normalize coordinates to match the input grid size
    ix = grid[..., 0]  # x-coordinates
    iy = grid[..., 1]  # y-coordinates

    if offset:
        offset_tensor = torch.linspace(0, 1 - (1 / N), N, device=grid.device).reshape(N, 1, 1)
        iy = ((iy + 1) / 2) * (IH - 2) + offset_tensor
        ix = ((ix + 1) / 2) * (IW - 2) + offset_tensor
    else:
        iy = (iy + 1) / 2 * (IH - 1)
        ix = (ix + 1) / 2 * (IW - 1)

    # Compute corner indices for the four neighboring grid vertices
    ix_left = torch.floor(ix).clamp(0, IW - 1).long()
    ix_right = (ix_left + 1).clamp(0, IW - 1).long()
    iy_top = torch.floor(iy).clamp(0, IH - 1).long()
    iy_bottom = (iy_top + 1).clamp(0, IH - 1).long()

    # Compute interpolation weights for each corner
    dx_right = step_f(ix_right - ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom - iy)
    dy_top = 1 - dy_bottom

    nw = dx_right * dy_bottom
    ne = dx_left * dy_bottom
    sw = dx_right * dy_top
    se = dx_left * dy_top

    # Reshape input for easier access to vertex values
    input_flat = input.view(N, C, IH * IW)

    # Helper function to scatter-add values to the grid
    def scatter_add_value(ix, iy, weight, value):
        # Convert the 2D indices into flat indices
        flat_indices = (iy * IW + ix).view(N, 1, H * W).expand(-1, C, -1)
        input_flat.scatter_add_(2, flat_indices, weight.view(N, 1, H * W) * value.view(N, C, H * W))

    # Contributions from scattered points to the grid (no gathering needed, values already in `grid`)
    value = torch.ones_like(ix)  # Assuming the contribution is 1 for each point

    # Accumulate the contributions for each corner
    scatter_add_value(ix_left, iy_top, nw, value)
    scatter_add_value(ix_right, iy_top, ne, value)
    scatter_add_value(ix_left, iy_bottom, sw, value)
    scatter_add_value(ix_right, iy_bottom, se, value)

    # Reshape the updated grid back to its original shape
    input_updated = input_flat.view(N, C, IH, IW)

    return input_updated  # (N, C, IH, IW) - the updated grid with vertex contributions


def response_encoder(input, grid, reference_values, step='bilinear', offset=False):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW) representing the grid (vertices).
        grid: A torch.Tensor of dimension (N, H, W, 2) representing scattered points in normalized coordinates.
        reference_values: A torch.Tensor of dimension (N, H, W, 1) representing the values to be scattered to the input grid.
    Return:
        torch.Tensor: The updated input grid after accumulating contributions from scattered points.
    '''
    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    # Step function definition (bilinear or cosine)
    if step == 'bilinear':
        step_f = lambda x: x
    elif step == 'cosine':
        step_f = lambda x: 0.5 * (1 - torch.cos(torch.pi * x))
    else:
        raise NotImplementedError

    # Normalize coordinates to match the input grid size
    ix = grid[..., 0]  # x-coordinates
    iy = grid[..., 1]  # y-coordinates

    if offset:
        offset_tensor = torch.linspace(0, 1 - (1 / N), N, device=grid.device).reshape(N, 1, 1)
        iy = ((iy + 1) / 2) * (IH - 2) + offset_tensor
        ix = ((ix + 1) / 2) * (IW - 2) + offset_tensor
    else:
        iy = (iy + 1) / 2 * (IH - 1)
        ix = (ix + 1) / 2 * (IW - 1)

    # Compute corner indices for the four neighboring grid vertices
    ix_left = torch.floor(ix).clamp(0, IW - 1).long()
    ix_right = (ix_left + 1).clamp(0, IW - 1).long()
    iy_top = torch.floor(iy).clamp(0, IH - 1).long()
    iy_bottom = (iy_top + 1).clamp(0, IH - 1).long()

    # Compute interpolation weights for each corner
    dx_right = step_f(ix_right - ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom - iy)
    dy_top = 1 - dy_bottom

    nw = dx_right * dy_bottom
    ne = dx_left * dy_bottom
    sw = dx_right * dy_top
    se = dx_left * dy_top


    input_flat = input.view(N, C, IH * IW)

    # Reshape reference values for use in scatter_add (values from the reference grid)
    reference_values_flat = reference_values.view(N, 1, H * W)  # (N, 1, H * W)

    # Helper function to scatter-add values to the grid
    def scatter_add_value(ix, iy, weight, value):
        # Convert the 2D indices into flat indices
        flat_indices = (iy * IW + ix).view(N, 1, H * W).expand(-1, C, -1)
        input_flat.scatter_add_(2, flat_indices, weight.view(N, 1, H * W).expand(-1, C, -1) * value.view(N, 1, H * W))

    # Accumulate the contributions from the reference grid to the four neighboring vertices
    scatter_add_value(ix_left, iy_top, nw, reference_values_flat)
    scatter_add_value(ix_right, iy_top, ne, reference_values_flat)
    scatter_add_value(ix_left, iy_bottom, sw, reference_values_flat)
    scatter_add_value(ix_right, iy_bottom, se, reference_values_flat)

    # Reshape the updated grid back to its original shape
    input_updated = input_flat.view(N, C, IH, IW)

    return input_updated  # (N, C, IH, IW) - the updated grid with vertex contributions


    
def reconstruct_reference_values(input_updated, grid, step='bilinear', offset=False):
    '''
    Args:
        input_updated : A torch.Tensor of dimension (N, C, IH, IW) representing the updated grid.
        grid: A torch.Tensor of dimension (N, H, W, 2) representing scattered points in normalized coordinates.
    Return:
        torch.Tensor: The reconstructed reference_values of dimension (N, H, W, 1) from the updated grid.
    '''
    N, C, IH, IW = input_updated.shape
    _, H, W, _ = grid.shape

    # Step function definition (bilinear or cosine)
    if step == 'bilinear':
        step_f = lambda x: x
    elif step == 'cosine':
        step_f = lambda x: 0.5 * (1 - torch.cos(torch.pi * x))
    else:
        raise NotImplementedError

    # Normalize coordinates to match the input grid size
    ix = grid[..., 0]  # x-coordinates
    iy = grid[..., 1]  # y-coordinates

    if offset:
        offset_tensor = torch.linspace(0, 1 - (1 / N), N, device=grid.device).reshape(N, 1, 1)
        iy = ((iy + 1) / 2) * (IH - 2) + offset_tensor
        ix = ((ix + 1) / 2) * (IW - 2) + offset_tensor
    else:
        iy = (iy + 1) / 2 * (IH - 1)
        ix = (ix + 1) / 2 * (IW - 1)

    # Compute corner indices for the four neighboring grid vertices
    ix_left = torch.floor(ix).clamp(0, IW - 1).long()
    ix_right = (ix_left + 1).clamp(0, IW - 1).long()
    iy_top = torch.floor(iy).clamp(0, IH - 1).long()
    iy_bottom = (iy_top + 1).clamp(0, IH - 1).long()

    # Compute interpolation weights for each corner
    dx_right = step_f(ix_right - ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom - iy)
    dy_top = 1 - dy_bottom

    nw = dx_right * dy_bottom
    ne = dx_left * dy_bottom
    sw = dx_right * dy_top
    se = dx_left * dy_top


    # Reshape input for easier access to vertex values
    input_flat = input_updated.view(N, C, IH * IW)

    # Helper function to gather the values from the grid
    def gather_value(ix, iy, weight):
        flat_indices = (iy * IW + ix).view(N, 1, H * W).expand(-1, C, -1)
        gathered_values = torch.gather(input_flat, 2, flat_indices)
        return gathered_values * weight.view(N, 1, H * W).expand(-1, C, -1)

    # Gather contributions from the four neighboring vertices
    val_nw = gather_value(ix_left, iy_top, nw)
    val_ne = gather_value(ix_right, iy_top, ne)
    val_sw = gather_value(ix_left, iy_bottom, sw)
    val_se = gather_value(ix_right, iy_bottom, se)

    # Sum the contributions from the four corners to get the reconstructed reference values
    reconstructed_values = val_nw + val_ne + val_sw + val_se

    # Sum across the channels to reconstruct the reference values correctly
    reconstructed_values_sum = reconstructed_values.sum(dim=1, keepdim=True)

    # Reshape the reconstructed values to the original shape (N, H, W, 1)
    reconstructed_values_reshaped = reconstructed_values_sum.view(N, H, W, 1)

    return reconstructed_values_reshaped





def geometry_encoder3d(input, grid, step='trilinear', offset=False):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW, ID) representing the grid (vertices).
        grid: A torch.Tensor of dimension (N, H, W, D, 3) representing scattered points in normalized coordinates.
    Return:
        torch.Tensor: The updated input grid after accumulating contributions from scattered points.
    '''
    N, C, IH, IW, ID = input.shape
    _, H, W, D, _ = grid.shape

    # Step function definition (bilinear or cosine)
    if step == 'trilinear':
        step_f = lambda x: x
    elif step == 'cosine':
        step_f = lambda x: 0.5 * (1 - torch.cos(torch.pi * x))
    else:
        raise NotImplementedError

    # Normalize coordinates to match the input grid size
    ix, iy, iz = grid[..., 0], grid[..., 1], grid[..., 2]  # x, y, and z coordinates

    if offset:
        offset_tensor = torch.linspace(0, 1 - (1 / N), N, device=grid.device).reshape(N, 1, 1, 1)
        ix = ((ix + 1) / 2) * (IW - 2) + offset_tensor
        iy = ((iy + 1) / 2) * (IH - 2) + offset_tensor
        iz = ((iz + 1) / 2) * (ID - 2) + offset_tensor
    else:
        ix = (ix + 1) / 2 * (IW - 1)
        iy = (iy + 1) / 2 * (IH - 1)
        iz = (iz + 1) / 2 * (ID - 1)

    # Compute corner indices for the eight neighboring grid vertices
    ix_left = torch.floor(ix).clamp(0, IW - 1).long()
    ix_right = (ix_left + 1).clamp(0, IW - 1).long()
    iy_top = torch.floor(iy).clamp(0, IH - 1).long()
    iy_bottom = (iy_top + 1).clamp(0, IH - 1).long()
    iz_front = torch.floor(iz).clamp(0, ID - 1).long()
    iz_back = (iz_front + 1).clamp(0, ID - 1).long()

    # Compute interpolation weights for each corner
    dx_right = step_f(ix_right - ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom - iy)
    dy_top = 1 - dy_bottom
    dz_back = step_f(iz_back - iz)
    dz_front = 1 - dz_back

    # Calculate weights for the eight corners
    nwf = dx_left * dy_top * dz_front
    nef = dx_right * dy_top * dz_front
    swf = dx_left * dy_bottom * dz_front
    sef = dx_right * dy_bottom * dz_front
    nwb = dx_left * dy_top * dz_back
    neb = dx_right * dy_top * dz_back
    swb = dx_left * dy_bottom * dz_back
    seb = dx_right * dy_bottom * dz_back

    # Flatten input for easier manipulation
    input_flat = input.view(N, C, IH * IW * ID)

    # Helper function to scatter-add values to the grid
    def scatter_add_value(ix, iy, iz, weight, value):
        flat_indices = (iy * IW * ID + ix * ID + iz).view(N, 1, H * W * D).expand(-1, C, -1)
        input_flat.scatter_add_(2, flat_indices, weight.view(N, 1, H * W * D) * value.view(N, C, H * W * D))

    # Contributions from scattered points to the grid (assuming value=1 for each point)
    value = torch.ones_like(ix)

    # Accumulate contributions for each corner
    scatter_add_value(ix_left, iy_top, iz_front, nwf, value)
    scatter_add_value(ix_right, iy_top, iz_front, nef, value)
    scatter_add_value(ix_left, iy_bottom, iz_front, swf, value)
    scatter_add_value(ix_right, iy_bottom, iz_front, sef, value)
    scatter_add_value(ix_left, iy_top, iz_back, nwb, value)
    scatter_add_value(ix_right, iy_top, iz_back, neb, value)
    scatter_add_value(ix_left, iy_bottom, iz_back, swb, value)
    scatter_add_value(ix_right, iy_bottom, iz_back, seb, value)

    # Reshape the updated grid back to its original shape
    input_updated = input_flat.view(N, C, IH, IW, ID)

    return input_updated  # (N, C, IH, IW, ID) - the updated grid with vertex contributions



def response_encoder3d(input, grid, reference_values, step='trilinear', offset=False):
    N, C, IH, IW, ID = input.shape
    _, H, W, D, _ = grid.shape

    # Step function definition (bilinear or cosine)
    if step == 'trilinear':
        step_f = lambda x: x
    elif step == 'cosine':
        step_f = lambda x: 0.5 * (1 - torch.cos(torch.pi * x))
    else:
        raise NotImplementedError

    # Normalize coordinates to match the input grid size
    ix, iy, iz = grid[..., 0], grid[..., 1], grid[..., 2]

    if offset:
        offset_tensor = torch.linspace(0, 1 - (1 / N), N, device=grid.device).reshape(N, 1, 1, 1)
        ix = ((ix + 1) / 2) * (IW - 2) + offset_tensor
        iy = ((iy + 1) / 2) * (IH - 2) + offset_tensor
        iz = ((iz + 1) / 2) * (ID - 2) + offset_tensor
    else:
        ix = (ix + 1) / 2 * (IW - 1)
        iy = (iy + 1) / 2 * (IH - 1)
        iz = (iz + 1) / 2 * (ID - 1)

    # Compute corner indices
    ix_left, ix_right = torch.floor(ix).clamp(0, IW - 1).long(), (torch.floor(ix) + 1).clamp(0, IW - 1).long()
    iy_top, iy_bottom = torch.floor(iy).clamp(0, IH - 1).long(), (torch.floor(iy) + 1).clamp(0, IH - 1).long()
    iz_front, iz_back = torch.floor(iz).clamp(0, ID - 1).long(), (torch.floor(iz) + 1).clamp(0, ID - 1).long()

    # Weights
    dx_right, dx_left = step_f(ix_right - ix), 1 - step_f(ix_right - ix)
    dy_bottom, dy_top = step_f(iy_bottom - iy), 1 - step_f(iy_bottom - iy)
    dz_back, dz_front = step_f(iz_back - iz), 1 - step_f(iz_back - iz)

    nwf, nef = dx_left * dy_top * dz_front, dx_right * dy_top * dz_front
    swf, sef = dx_left * dy_bottom * dz_front, dx_right * dy_bottom * dz_front
    nwb, neb = dx_left * dy_top * dz_back, dx_right * dy_top * dz_back
    swb, seb = dx_left * dy_bottom * dz_back, dx_right * dy_bottom * dz_back

    # Flatten for easier indexing
    input_flat = input.view(N, C, IH * IW * ID)
    reference_values_flat = reference_values.view(N, 1, H * W * D)

    # Scatter-add with reference values
    def scatter_add_value(ix, iy, iz, weight, value):
        flat_indices = (iy * IW * ID + ix * ID + iz).view(N, 1, H * W * D).expand(-1, C, -1)
        input_flat.scatter_add_(2, flat_indices, weight.view(N, 1, H * W * D).expand(-1, C, -1) * value.view(N, 1, H * W * D))

    scatter_add_value(ix_left, iy_top, iz_front, nwf, reference_values_flat)
    scatter_add_value(ix_right, iy_top, iz_front, nef, reference_values_flat)
    scatter_add_value(ix_left, iy_bottom, iz_front, swf, reference_values_flat)
    scatter_add_value(ix_right, iy_bottom, iz_front, sef, reference_values_flat)
    scatter_add_value(ix_left, iy_top, iz_back, nwb, reference_values_flat)
    scatter_add_value(ix_right, iy_top, iz_back, neb, reference_values_flat)
    scatter_add_value(ix_left, iy_bottom, iz_back, swb, reference_values_flat)
    scatter_add_value(ix_right, iy_bottom, iz_back, seb, reference_values_flat)

    return input_flat.view(N, C, IH, IW, ID)



def reconstruct_reference_values3d(input_updated, grid, step='trilinear', offset=False):
    N, C, IH, IW, ID = input_updated.shape
    _, H, W, D, _ = grid.shape

    if step == 'trilinear':
        step_f = lambda x: x
    elif step == 'cosine':
        step_f = lambda x: 0.5 * (1 - torch.cos(torch.pi * x))
    else:
        raise NotImplementedError

    ix, iy, iz = grid[..., 0], grid[..., 1], grid[..., 2]
    if offset:
        offset_tensor = torch.linspace(0, 1 - (1 / N), N, device=grid.device).reshape(N, 1, 1, 1)
        ix = ((ix + 1) / 2) * (IW - 2) + offset_tensor
        iy = ((iy + 1) / 2) * (IH - 2) + offset_tensor
        iz = ((iz + 1) / 2) * (ID - 2) + offset_tensor
    else:
        ix, iy, iz = (ix + 1) / 2 * (IW - 1), (iy + 1) / 2 * (IH - 1), (iz + 1) / 2 * (ID - 1)

    ix_left, ix_right = torch.floor(ix).clamp(0, IW - 1).long(), (torch.floor(ix) + 1).clamp(0, IW - 1).long()
    iy_top, iy_bottom = torch.floor(iy).clamp(0, IH - 1).long(), (torch.floor(iy) + 1).clamp(0, IH - 1).long()
    iz_front, iz_back = torch.floor(iz).clamp(0, ID - 1).long(), (torch.floor(iz) + 1).clamp(0, ID - 1).long()

    dx_right, dx_left = step_f(ix_right - ix), 1 - step_f(ix_right - ix)
    dy_bottom, dy_top = step_f(iy_bottom - iy), 1 - step_f(iy_bottom - iy)
    dz_back, dz_front = step_f(iz_back - iz), 1 - step_f(iz_back - iz)

    nwf, nef = dx_left * dy_top * dz_front, dx_right * dy_top * dz_front
    swf, sef = dx_left * dy_bottom * dz_front, dx_right * dy_bottom * dz_front
    nwb, neb = dx_left * dy_top * dz_back, dx_right * dy_top * dz_back
    swb, seb = dx_left * dy_bottom * dz_back, dx_right * dy_bottom * dz_back

    input_flat = input_updated.view(N, C, IH * IW * ID)
    def gather_value(ix, iy, iz, weight):
        flat_indices = (iy * IW * ID + ix * ID + iz).view(N, 1, H * W * D).expand(-1, C, -1)
        gathered_values = torch.gather(input_flat, 2, flat_indices)
        return gathered_values * weight.view(N, 1, H * W * D).expand(-1, C, -1)

    val_nwf = gather_value(ix_left, iy_top, iz_front, nwf)
    val_nef = gather_value(ix_right, iy_top, iz_front, nef)
    val_swf = gather_value(ix_left, iy_bottom, iz_front, swf)
    val_sef = gather_value(ix_right, iy_bottom, iz_front, sef)
    val_nwb = gather_value(ix_left, iy_top, iz_back, nwb)
    val_neb = gather_value(ix_right, iy_top, iz_back, neb)
    val_swb = gather_value(ix_left, iy_bottom, iz_back, swb)
    val_seb = gather_value(ix_right, iy_bottom, iz_back, seb)

    reconstructed_values = val_nwf + val_nef + val_swf + val_sef + val_nwb + val_neb + val_swb + val_seb
    reconstructed_values_sum = reconstructed_values.sum(dim=1, keepdim=True)
    return reconstructed_values_sum.view(N, H, W, D, 1)


# B = 100
# grid_i50 = torch.zeros([B , 1 ,50 , 50 , 50])

# input = torch.rand([B , 1 , 1,1000 , 3])#torch.cat([x,y] , -1).unsqueeze(1) 
# ref =  torch.rand([B , 1 , 1,1000 , 1])


# grid_o50 = geometry_encoder3d( grid_i50.clone().to(torch.float32) , input.to(torch.float32) , step = 'trilinear')
# grid_u50 = response_encoder3d( grid_i50.clone().to(torch.float32) , input.to(torch.float32) , ref.to(torch.float32) ,  step = 'trilinear')

# print(grid_o50.shape , grid_u50.shape)



# data = torch.load(r"D:\PMACS\naca\X.pt") #torch.Size([2490, 2820, 2])
# response = torch.load(r"D:\PMACS\naca\U.pt") #torch.Size([2490, 2820])
# B = data.shape[0]
# x_min , x_max , y_min , y_max = [-0.4 , 1.26 , -0.45 , 0.45]
# grid_i200 = torch.zeros([B , 1 ,400 , 400])
# x = (data[... , 0]-x_min)/(x_max - x_min)
# y = (data[... , 1]-y_min)/(y_max - y_min)
# x = x.unsqueeze(-1)*2 -1
# y = y.unsqueeze(-1)*2 -1
# input = torch.cat([x,y] , -1).unsqueeze(1) 
# ref = response.unsqueeze(1).unsqueeze(-1)

# grid_o200 = geometry_encoder( grid_i200.clone().to(torch.float32) , input.to(torch.float32) , step = 'bilinear')
# grid_u200 = response_encoder( grid_i200.clone().to(torch.float32) , input.to(torch.float32) , ref.to(torch.float32) ,  step = 'bilinear')

# print(grid_o200.shape , grid_u200.shape)