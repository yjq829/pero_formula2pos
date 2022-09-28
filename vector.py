# ******************************************************************************************************
# python 3.6
# version 1.0
# calculate gesture vectors for input ligand, 2 vectors are introduced
#
# Be advised, due to the property of python, some function will change the input data when it is called.
# Be advised, due to the property of python, some function will change the input data when it is called.
# Be advised, due to the property of python, some function will change the input data when it is called.
#
# ******************************************************************************************************

# dependency environment

import numpy as np
import math
from io_lib import *
import copy


# building gesture vectors
def end_atom_locator(atom_chain_in, atom_h_in):
    top_N_index = NH3_locator(atom_chain_in, atom_h_in)
    for x in atom_chain_in:
        if x.index == top_N_index:
            v_start = np.array(x.coordination)
            break
    dis_max = 0
    index_max = 0
    for a in range(len(atom_chain_in)):
        if a != top_N_index:
            v_end = np.array(atom_chain_in[a].coordination)
            dis = np.linalg.norm(v_end - v_start)
            if abs(dis) >= dis_max:
                dis_max = abs(dis)
                index_max = atom_chain_in[a].index

    return index_max


def ligand_vector_z(atom_chain_ligand, NH3_index, end_index):
    for y in atom_chain_ligand:
        if y.index == NH3_index:
            p_start = np.array(y.coordination)
        elif y.index == end_index:
            p_end = np.array(y.coordination)

    vec_z = p_end - p_start
    return vec_z


# rotation calculation to align chain vector // z-axis
# which requires 2 rotation, along x & y


def rotation_angle_x(vec_z_x, vec_nor):
    vec_tem = copy.copy(vec_z_x)
    vec_tem[0] = 0
    cos_beta = np.dot(vec_tem, vec_nor) / np.linalg.norm(vec_tem)
    x_angle = math.acos(cos_beta)
    return x_angle


def rotation_angle_y(vec_z_y, vec_nor):
    vec_tem = copy.copy(vec_z_y)
    vec_tem[1] = 0
    cos_beta = np.dot(vec_tem, vec_nor) / np.linalg.norm(vec_tem)
    y_angle = math.acos(cos_beta)
    return y_angle


# calculate 'vec_z' is left to [001] or right to [001]
def rotation_symbol_x(vec_z_x, vec_nor):
    vec_tem = copy.copy(vec_z_x)
    vec_tem[0] = 0
    cross = np.cross(vec_nor, vec_tem)
    if cross[0] >= 0:
        x_symbol = -1
    else:
        x_symbol = 1

    return x_symbol


def rotation_symbol_y(vec_z_y, vec_nor):
    vec_tem = copy.copy(vec_z_y)
    vec_tem[1] = 0
    cross = np.cross(vec_nor, vec_tem)
    if cross[1] >= 0:
        y_symbol = -1
    else:
        y_symbol = 1
    return y_symbol


def rotation_matrix_x(x_angle):
    x_M_tem = np.zeros((3, 3))
    x_M_tem[0, 0] = 1
    x_M_tem[1, 1] = math.cos(x_angle)
    x_M_tem[1, 2] = -math.sin(x_angle)
    x_M_tem[2, 1] = math.sin(x_angle)
    x_M_tem[2, 2] = math.cos(x_angle)
    return x_M_tem


def rotation_matrix_y(y_angle):
    y_M_tem = np.zeros((3, 3))
    y_M_tem[1, 1] = 1
    y_M_tem[0, 0] = math.cos(y_angle)
    y_M_tem[0, 2] = math.sin(y_angle)
    y_M_tem[2, 0] = -math.sin(y_angle)
    y_M_tem[2, 2] = math.cos(y_angle)
    return y_M_tem


def total_rotation_align_z(x_M_input, y_M_input):
    tot_align_z = np.dot(x_M_input, y_M_input)
    return tot_align_z


def tilt_rotation_matrix(theta):
    tilt_M = np.zeros((3, 3))
    tilt_M[0, 0] = 1
    tilt_M[1, 1] = math.cos(theta)
    tilt_M[1, 2] = -math.sin(theta)
    tilt_M[2, 1] = -math.sin(theta)
    tilt_M[2, 2] = math.cos(theta)
    return tilt_M


def tilt_processs(t_flag, theta, z_align_matrix):
    z_align_matrix_tem = copy.deepcopy(z_align_matrix)
    if t_flag == 1:
        tilt_rot_matrix = tilt_rotation_matrix(theta)
        new_align_matrix = np.dot(z_align_matrix_tem, tilt_rot_matrix)
    elif t_flag == 0:
        new_align_matrix = z_align_matrix_tem
    return new_align_matrix


# rotation to align chain // z is completed
# now begin to apply the rotation matrix to all atom

def atom_rotation(atom_list, rot_M):
    atom_roted = copy.deepcopy(atom_list)
    for atom in atom_roted:
        co = np.array(atom.coordination)
        co = np.dot(co, rot_M)
        atom.coordination = co
    return atom_roted


def rotation_apply(atom_chain_rot, atom_h_rot, cell_ligand_rot, tot_rot_app):
    atom_chain_rot_tem = copy.deepcopy(atom_chain_rot)
    atom_h_rot_tem = copy.deepcopy(atom_h_rot)
    cell_new = copy.deepcopy(cell_ligand_rot)
    tot_rot_tem = copy.copy(tot_rot_app)
    atom_chain_rot_tem = atom_rotation(atom_chain_rot_tem, tot_rot_tem)
    atom_h_rot_tem = atom_rotation(atom_h_rot_tem, tot_rot_tem)
    p_new = cell_new.get_positions()
    for i in range(len(atom_chain_rot_tem)):
        index_t = atom_chain_rot_tem[i].index
        p_new[index_t] = atom_chain_rot_tem[i].coordination
    for i in range(len(atom_h_rot_tem)):
        index_h = atom_h_rot_tem[i].index
        p_new[index_h] = atom_h_rot_tem[i].coordination
    cell_new.set_positions(p_new)
    return cell_new


#######################################################################################################################

# after rotation, we need to adsorb the ligand on the surface

def vec_adsorption(position_coordination, cell_rot_input):
    cell_rot_input_tem = copy.deepcopy(cell_rot_input)
    atom_chain_vec_ads, atom_h_vec_ads = chain_input(cell_rot_input_tem)
    top_N_index = NH3_locator(atom_chain_vec_ads, atom_h_vec_ads)
    for x in atom_chain_vec_ads:
        if x.index == top_N_index:
            vec_ini = np.array(x.coordination)
            break
    vec_ads = np.array(position_coordination)
    vec_move = vec_ads - vec_ini
    return vec_move


# move all ligand atoms as vec_move shows
def apply_ads_movement(cell_ligand_mov, vec_move):
    cell_ligand_ads = copy.deepcopy(cell_ligand_mov)
    atom_chain_ads, atom_h_ads = chain_input(cell_ligand_ads)
    p_ads = cell_ligand_ads.get_positions()
    for i in range(len(atom_chain_ads)):
        index_t = atom_chain_ads[i].index
        p_ads[index_t] = atom_chain_ads[i].coordination + vec_move
    for i in range(len(atom_h_ads)):
        index_h = atom_h_ads[i].index
        p_ads[index_h] = atom_h_ads[i].coordination + vec_move
    cell_ligand_ads.set_positions(p_ads)
    return cell_ligand_ads


######################################################################################################################

# automatically detect the adosption position
# Mark atom: B
# test version: 1 mark per unit cell
def find_mark(cell_slab):
    cell_slab_tem = copy.deepcopy(cell_slab)
    slab_element = cell_slab_tem.get_chemical_symbols()
    mark_index = []
    for e in range(len(slab_element)):
        if slab_element[e] == 'B':
            mark_index.append(e)
    if len(mark_index) == 1:
        print("There is ONLY 1 mark. OK to go!")
        return mark_index
    else:
        print("There are more than 1 marks. The number of marks is", len(mark_index))
        return mark_index


# use mark atom position to get adsorption position
# [x,y,z]=[xB,yB,zB]+[0,0,1.5] 1.5A here is adjustable
def mark_to_vec_up(mark_index, num_mark_u, cell_slab):
    cell_slab_tem = copy.deepcopy(cell_slab)
    goal_position_u = cell_slab_tem.get_positions()[mark_index[num_mark_u]]
    goal_position_u = np.array(goal_position_u + [0.4, 0.4, 1.5])
    return goal_position_u


def mark_to_vec_down(mark_index, num_mark_d, cell_slab):
    cell_slab_tem_d = copy.deepcopy(cell_slab)
    goal_position_d = cell_slab_tem_d.get_positions()[mark_index[num_mark_d]]
    goal_position_d = np.array(goal_position_d - [0, 0, 1.5])
    return goal_position_d


# Here, we need a function to generate a cell_slab without mark atom
def slab_del_B(cell_slab):
    cell_slab_new = copy.deepcopy(cell_slab)
    for num_del in sorted(range(len(cell_slab_new)), reverse=True):
        if cell_slab_new.get_chemical_symbols()[num_del] == 'B':
            del (cell_slab_new[num_del])
    return cell_slab_new


# test run
if __name__ == '__main__':
    cell_ligand = ligand_input('ligand/POSCAR')
    atom_chain, atom_h = chain_input(cell_ligand)
    # for i in range(len(atom_chain)):
    # print(atom_chain[i].coordination)
    # print('*******')
    # for i in range(len(atom_h)):
    # print(atom_h[i].coordination)
    # print('*******')
    # print(find_neighbor_H(atom_chain[0], atom_h))
    p1 = NH3_locator(atom_chain, atom_h)
    p2 = end_atom_locator(atom_chain, atom_h)
    vec_z = ligand_vector_z(atom_chain, p1, p2)
    x_ang = rotation_angle_x(vec_z)
    y_ang = rotation_angle_y(vec_z)
    x_ang = rotation_symbol_x(vec_z) * x_ang
    y_ang = rotation_symbol_y(vec_z) * y_ang
    x_M = rotation_matrix_x(x_ang)
    y_M = rotation_matrix_y(y_ang)
    print(x_M)
    print(y_M)
    result_rot = np.dot(vec_z, x_M)
    result_rot = np.dot(result_rot, y_M)
    tot_rot = total_rotation_align_z(x_M, y_M)
    print(np.dot(vec_z, tot_rot))
    print('**********')
    cell_roted = rotation_apply(atom_chain, atom_h, cell_ligand, tot_rot)
    from ase.io import write

    # 2 type of output, vasp version written by ASE is old version, which needs to copy and paste the element line
    # cif is easier to use.
    write('CONTCAR', cell_roted, format='vasp')
    write('result.cif', cell_roted, format='cif')
