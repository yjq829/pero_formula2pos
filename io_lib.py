# ******************************************************************************************************
# python 3.6
# version 1.0
# organic ligand and surface input
#
# Be advised, due to the property of python, some function will change the input data when it is called.
# Be advised, due to the property of python, some function will change the input data when it is called.
# Be advised, due to the property of python, some function will change the input data when it is called.
#
# ******************************************************************************************************

# dependency environment

from ase.io import read
import math
import numpy as np
import copy


# data class, describing the properties of every atom
class Chain:
    def __init__(self, index, coordination, previous, next_atom, element):
        self.index = index
        self.coordination = coordination
        self.previous = previous
        self.next = next_atom
        self.element = element


class H_group:
    def __init__(self, index, coordination):
        self.index = index
        self.coordination = coordination


# read ligand 'vasp' files
def ligand_input(dir_ligand):
    cell_ligand = read(dir_ligand, format='vasp')
    return cell_ligand


# grab and organize chain atom data
def chain_input(cell):
    a = cell.get_positions(cell)
    element = cell.get_chemical_symbols()
    atom_chain = []
    atom_h = []
    for i in range(len(a)):
        if element[i] != 'H':
            atom_chain.append(Chain(i, a[i], 0, 0, element[i]))
        if element[i] == 'H':
            atom_h.append(H_group(i, a[i]))
    return atom_chain, atom_h


# find the N atom INDEX(!!!) with NH3 acceptor
#
# NOTICE: the return in INDEX!!!!
#

def NH3_locator(atom_chain_input, atom_h_input):
    N_chain = []
    for i in range(len(atom_chain_input)):
        if atom_chain_input[i].element == 'N':
            N_chain.append(atom_chain_input[i])
    if len(N_chain) == 1:
        return N_chain[0].index
    else:
        for n in range(len(N_chain)):
            if find_neighbor_H(N_chain[n], atom_h_input) == 3:
                return N_chain[n].index


#
# calculate the distance between 1 N atom and the list of H atoms.
# GOAL: find number of H neighbor within 1.2A
# RETURN: the number of H less than 1.2A
# this function can distinguish N atom type in [NH2,NH3,NH]
#

def find_neighbor_H(atom_N, atom_h):
    num_neighbor = 0
    v1 = np.array(atom_N.coordination)
    for h in range(len(atom_h)):
        v2 = np.array(atom_h[h].coordination)
        dis = np.linalg.norm(v1 - v2)
        if abs(dis) <= 1.2:
            num_neighbor += 1
    return num_neighbor


#
#
# input slab information
#
def slab_input(dir_slab):
    cell_slab = read(dir_slab, format='vasp')
    return cell_slab


def ligand_ads(cell_slab, cell_ligand):
    cell_ads = copy.deepcopy(cell_slab)
    cell_ads.extend(cell_ligand)
    return cell_ads


# test run
if __name__ == '__main__':
    cell_ligand = ligand_input('ligand/POSCAR')
    atom_chain, atom_h = chain_input(cell_ligand)
    for i in range(len(atom_chain)):
        print(atom_chain[i].index)
    print('*******')
    for i in range(len(atom_h)):
        print(atom_h[i].index)
    print('*******')
    print(find_neighbor_H(atom_chain[0], atom_h))
    print(NH3_locator(atom_chain, atom_h))
