import meshio
import os.path as osp
import os
from pathlib import Path
import torch
# import trimesh
import numpy as np
import pickle
from torch_geometric.data import Data

# from hammer import GeoReader


# data_dir = os.path.join(Path.home(), 'data','hammer') 

def preprocess_graph_data(data_dir, model_name,bjorn=True):
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    if bjorn:
        vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    else:    
        vtk_dir = osp.join(data_dir,"vtk",problem_name, model_name)

    ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    os.makedirs(ml_data_dir, exist_ok=True)

    fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 

    voxel_inds, deposit_sequence, toolpath = fem_file["voxel_inds"], fem_file["deposit_sequence"], fem_file["toolpath"]
    dx = fem_file["dx"]
    nx = fem_file["nx"]
    Add_base = fem_file["Add_base"]
    path_dx = fem_file["dx"]
    if bjorn:
        whole_cells = fem_file["hexahedra"]
        whole_points = fem_file["vertices"]
        num_base = whole_cells.shape[0]-toolpath.shape[0]

    edges = []

    for i in range(whole_cells.shape[0]):
        for j in range(whole_cells.shape[0]):
            if i==j:
                    continue

            arr_1 = whole_cells[i]
            arr_2 = whole_cells[j]
            matching_elements = np.array([element for element in arr_1 if element in arr_2])
            if len(matching_elements)>=3:
                edges.append(np.array([i,j]))

        edges = np.array(edges)
        edge_index = torch.tensor(edges).long().t().contiguous()

    centeroids = np.mean(whole_points[whole_cells], axis=1)
    num_points = centeroids.shape[0]

    #base_depth = fem_file["base_depth"]
    base_depth = 50e-3
    ratio = np.abs(np.min(fem_file["vertices"][:,2]))/base_depth

    deposit_pairs = fem_file["deposit_pairs"]
    
    for i_sample in range(deposit_pairs.shape[0]):
        ### load vtu files
        if bjorn:
            i_time_step = num_base + deposit_pairs[i_sample,0] + 1
            vtk_path_00 = osp.join(vtk_dir, f"T{(i_time_step):07}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"T{(i_time_step+1):07d}.vtu")
        else:
            i_time_step = deposit_pairs[i_sample,0]
            vtk_path_00 = osp.join(vtk_dir, f"u_{(i_time_step):05d}_active_{0:05d}.vtu")       
            vtk_path_01 = osp.join(vtk_dir, f"u_{(i_time_step+1):05d}_active_{0:05d}.vtu")
        if os.path.isfile(vtk_path_00) and os.path.isfile(vtk_path_01):
            print(i_time_step)
            mesh_00 = meshio.read(vtk_path_00)
            mesh_01 = meshio.read(vtk_path_01)

            cells_00 = mesh_00.cells_dict['hexahedron'] ### input cells
            points_00 = mesh_00.points 
            if bjorn:
                points_00 /= 1e3
            points_00[points_00[:,2]<0,2] = points_00[points_00[:,2]<0,2]*ratio ### input point coordinates

            T_input = np.zeros(num_points)
            T_output = np.zeros(num_points)
            # get matching global index
            global_index_00 = match_global_and_local_cell_index(whole_cells, whole_points, cells_00, points_00)

            if bjorn:
                sol_00_center = np.expand_dims(mesh_00.cell_data['T'][0].astype(np.float32), axis=1) ### input temperature for each cells
                sol_00_center = torch.tensor(sol_00_center)
                # store the input temperature in T_input by global index
                T_input[global_index_00] = sol_00_center
            else:
                sol_00 = mesh_00.point_data['sol']
                sol_00_center = np.mean(sol_00[cells_00],axis=1)
 
            cells_01 = mesh_01.cells_dict['hexahedron'] ### output cells, note that there is a new activated element in the output mesh
                                                        ### build graph based on the input mesh
            points_01 = mesh_01.points
            if bjorn:
                points_01 /= 1e3
            points_01[points_01[:,2]<0,2] = points_01[points_01[:,2]<0,2]*ratio ### output point coordinates
            
            global_index_01 = match_global_and_local_cell_index(whole_cells, whole_points, cells_01, points_01)

            if bjorn:
                sol_01_center = np.expand_dims(mesh_01.cell_data['T'][0].astype(np.float32), axis=1) ### output temperature for each cells
                sol_01_center = torch.tensor(sol_01_center)
                # store the output temperature in T_output by global index
                T_output[global_index_01] = sol_01_center

            else:
                sol_01 = mesh_01.point_data['sol']
                sol_01_center = np.mean(sol_01[cells_01],axis=1)

            T_input = torch.tensor(T_input)
            T_output = torch.tensor(T_output)
            
            #######################
            ### build graph
            i_deposit = deposit_pairs[i_sample,0]
            lag = 1
            i_deposit -= lag
            laser_center = np.array([toolpath[i_deposit,1], toolpath[i_deposit,2], toolpath[i_deposit,3]])
            centeroids_00 = np.mean(points_00[cells_00],axis=1)
            
            heat_info = laser_center-centeroids_00 ###  input laser position   
            heat_info_global = np.zeros(num_points)
            heat_info_global[global_index_00] = heat_info
            heat_info_global = torch.tensor(heat_info_global)

            pairwise_data = Data(x=torch.cat([T_input, heat_info], dim=1), y=T_output, edge_index=edge_index, pos=centeroids)
            torch.save(pairwise_data, osp.join(ml_data_dir, f"model_{model_name}_problem_{problem_name}_{i_time_step}.pt"))
            
            ###### summary
            ###### input node attribude: input temperature, point coordinates, heat_info
            ###### output node attribude: output temperature 
            ###### build graph based on input cells 
            

          
def match_global_and_local_cell_index(whole_cells, whole_points, cells_00, points_00):
    # problem_name = "small_10_base_20"
    # femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    # if bjorn:
    #     vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    # else:    
    #     vtk_dir = osp.join(data_dir,"vtk",problem_name, model_name)

    # ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    # os.makedirs(ml_data_dir, exist_ok=True)

    # fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 

    # global cell index
    #base_depth = fem_file["base_depth"]
    base_depth = 50e-3

    # output global index of local cells by matching the coordinate of points
    cells_00_global = np.zeros(cells_00.shape, dtype=np.int32)
    for i in range(cells_00.shape[0]):
        cells_00_global[i] = np.where(np.all(whole_points[whole_cells]==points_00[cells_00[i]], axis=1))[0]

    # return cells_00_global
    return cells_00_global


if __name__ == "__main__":
    data_dir = "D:/Work/research/data/hammer"
    model_name = "hollow_1"
    problem_name = "small_10_base_20"
    femfile_dir = osp.join(data_dir,"meshes","extend_base_bjorn",problem_name)
    vtk_dir = osp.join(data_dir,"vtk",'bjorn_fem',problem_name, model_name,'vtk')
    # ml_data_dir = osp.join(data_dir,"ml_data",model_name)
    # os.makedirs(ml_data_dir, exist_ok=True)

    fem_file = pickle.load( open( osp.join(femfile_dir, model_name+".p"), "rb" ) ) 
    vtk_path_00 = osp.join(vtk_dir, f"vtk_{0:05d}.vtk")
    vtk_path_01 = osp.join(vtk_dir, f"vtk_{1:05d}.vtk")
    match_global_and_local_cell_index(fem_file, vtk_path_00, vtk_path_01, bjorn=True)