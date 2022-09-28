# ******************************************************************************************************
# python 3.6
# version 1.0
#
# main running script
#
# Be advised, due to the property of python, some function will change the input data when it is called.
# Be advised, due to the property of python, some function will change the input data when it is called.
# Be advised, due to the property of python, some function will change the input data when it is called.
#
#
# v1.4 update:
# (1) tilt function added
#       (a) tilt_flag = 1 -> tilt activated
#           tilt_theta = degree *pi/180
#       (b) function changed: vector -> tilt rotation matrix
#                             ligand_ads -> assemble process change
#
#
# ******************************************************************************************************

# dependency environment
from ase.io import read, write
import numpy as np
import math
from io_lib import *
import copy
from vector import *
import time

# import ligand
cell_ligand = ligand_input('ligand/POSCAR')
atom_chain, atom_h = chain_input(cell_ligand)

# import slab
cell_slab = slab_input('slab/POSCAR')

# ******************************************************************************************************
# calculate gesture vectors
p1 = NH3_locator(atom_chain, atom_h)
p2 = end_atom_locator(atom_chain, atom_h)
vec_z_ligand = ligand_vector_z(atom_chain, p1, p2)

# calculate ligand rotation angle
vec_up = np.array([0, 0, -1])
vec_down = np.array([0, 0, 1])

# tilt activated
tilt_flag = 1
tilt_theta = (10 * math.pi / 180)


def run_ligand_rotation(vec_dir_input, vec_z_input, atom_chain_run, atom_h_run, cell_ligand_run, t_flag, theta):
    atom_chain_run_tem = copy.copy(atom_chain_run)
    atom_h_run_tem = copy.copy(atom_h_run)
    cell_ligand_run_tem = copy.deepcopy(cell_ligand_run)
    vec_z_tem = copy.copy(vec_z_input)
    vec_dir = copy.copy(vec_dir_input)
    x_ang_tem = rotation_angle_x(vec_z_tem, vec_dir)
    y_ang_tem = rotation_angle_y(vec_z_tem, vec_dir)
    x_ang_tem = rotation_symbol_x(vec_z_tem, vec_dir) * x_ang_tem
    y_ang_tem = rotation_symbol_y(vec_z_tem, vec_dir) * y_ang_tem
    x_M_tem = rotation_matrix_x(x_ang_tem)
    y_M_tem = rotation_matrix_y(y_ang_tem)
    # result_rot = np.dot(vec_z, x_M)
    # result_rot = np.dot(result_rot, y_M)
    tot_rot_tem = total_rotation_align_z(x_M_tem, y_M_tem)
    tot_rot_tilt = tilt_processs(t_flag, theta, tot_rot_tem)
    cell_roted_tem = rotation_apply(atom_chain_run_tem, atom_h_run_tem, cell_ligand_run_tem, tot_rot_tilt)
    return cell_roted_tem


cell_roted_up = run_ligand_rotation(vec_up, vec_z_ligand, atom_chain, atom_h, cell_ligand, tilt_flag, tilt_theta)

cell_roted_down = run_ligand_rotation(vec_down, vec_z_ligand, atom_chain, atom_h, cell_ligand, tilt_flag, tilt_theta)

# 2 type of output, vasp version written by ASE is old version, which needs to copy and paste the element line
# cif is easier to use.
# write('CONTCAR', cell_roted, format='vasp')
write('result_up.cif', cell_roted_up, format='cif')
write('result_down.cif', cell_roted_down, format='cif')

# ******************************************************************************************************
# try adsorption
# detect mark and generate position vector
mark = find_mark(cell_slab)
print(mark)
cell_slab_start = copy.deepcopy(cell_slab)
for mark_num in range(len(mark)):
    p_up = mark_to_vec_up(mark, mark_num, cell_slab)
    p_down = mark_to_vec_down(mark, mark_num, cell_slab)

    # generate a slab cell (Atoms class) without mark atoms
    # move ligand atoms
    vec_move_up = vec_adsorption(p_up, cell_roted_up)
    cell_ligand_ads_up = apply_ads_movement(cell_roted_up, vec_move_up)
    write('ads_lig_up.cif', cell_ligand_ads_up, format='cif')
    vec_move_down = vec_adsorption(p_down, cell_roted_down)
    cell_ligand_ads_down = apply_ads_movement(cell_roted_down, vec_move_down)

    # combine slab and ligand, output
    cell_ads_up = ligand_ads(cell_slab_start, cell_ligand_ads_up)
    cell_ads_down = ligand_ads(cell_ads_up, cell_ligand_ads_down)
    cell_slab_start = copy.deepcopy(cell_ads_down)

cell_slab_final = slab_del_B(cell_ads_down)
write('ligand_ads.cif', cell_slab_final, format='cif')

# ******************************************************************************************************
print("*************************")
print("The process is completed!")
print(time.asctime(time.localtime(time.time())))
