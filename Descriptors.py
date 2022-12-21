import os
import argparse
import pickle

import pandas as pd
import numpy as np

from rdkit import Chem

from morfeus.conformer import ConformerEnsemble
from morfeus import XTB as morf_xtb
from morfeus import SASA, Dispersion as morf_all
from morfeus import Sterimol, BuriedVolume, Pyramidalization, LocalForce


# def compute(smiles, n_confs=None, program='xtb', method='GFN2-xTB', basis=None, solvent=None):

# input optimization method, default GFN2-xTB

def compute(smiles, program='xtb', method='GFN2-xTB'):

    print(f"Calculating {smiles} conformer ensemble!")
    ce = ConformerEnsemble.from_rdkit(smiles,
                                      n_threads=os.cpu_count() - 1)
    print(f"Total of {ce}!")

    print(f"Optimizing {smiles} conformers!")
    # optimize conformer geometries
    ce.optimize_qc_engine(program=program,
                          model={'method': method},
                          procedure="geometric")
    print(f"Pruning!")

    ce.prune_rmsd()

    print(f"Total of {ce}!")

    print(f"Computing {smiles} energies!")
    # compute energies of the single point calculations
    ce.sp_qc_engine(program=program, model={'method': method})

    # sort on energy and generate an rdkit molecule
    ce.sort()
    ce.generate_mol()

    # compute xtb for the conformers for descriptor calculations
    for conf in ce:
        conf.xtb = morf_xtb(ce.elements, conf.coordinates, version='2')
        conf.morfeus_desc = morf_all(ce.elements, conf.coordinates)

    return ce

# calculates descriptors from conformer ensemble

def get_descriptors(conf_ensemble):
    print("Calculating global descriptors!")
    for conf in conf_ensemble.conformers:
        x = conf.xtb
        h = conf.morfeus_desc
        conf.properties['IP'] = x.get_ip(corrected=True)
        conf.properties['EA'] = x.get_ea()
        conf.properties['HOMO'] = x.get_homo()
        conf.properties['LUMO'] = x.get_lumo()

        dip = x.get_dipole()
        conf.properties['dipole'] = np.sqrt(dip.dot(dip))

        conf.properties['electro'] = x.get_global_descriptor("electrophilicity", corrected=True)
        conf.properties['nucleo'] = x.get_global_descriptor("nucleophilicity", corrected=True)

        # sasa = SASA(conf.elements, conf.coordinates)
        conf.properties['SASA'] = h.area

        # disp = Dispersion(conf.elements, conf.coordinates)
        conf.properties['p_int'] = h.p_int
        conf.properties['p_max'] = h.p_max
        conf.properties['p_min'] = h.p_min

    props = {}
    for key in conf_ensemble.get_properties().keys():
        props[f"{key}_Boltz"] = conf_ensemble.boltzmann_statistic(key)
        props[f"{key}_Emin"] = conf_ensemble.get_properties()[key][0]

    return pd.Series(props)

# finds boric acid or halogen and its neighboring carbon and outputs the IDs. If no match, only global descriptors are calculated

def neighbors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    ids = False
    a = None
    b = None

    while ids is False:

        for atom in mol.GetAtoms():

            if atom.GetSymbol() == 'B':
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'C':
                        if nbr.GetSymbol() == 'O':
                            a = (atom.GetIdx() + 1)
                            b = (nbr.GetIdx() + 1)
                            yield a, b
                            return True

        for atom in mol.GetAtoms():

            if atom.GetSymbol() == 'Br' or atom.GetSymbol() == 'Cl' or atom.GetSymbol() == 'I':
                a = (atom.GetIdx() + 1)
                for nbr in atom.GetNeighbors():
                    b = (nbr.GetIdx() + 1)
                    yield a, b
                    return True

        for atom in mol.GetAtoms():

            if atom.GetSymbol() == 'O':
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'C':
                        if nbr.GetSymbol() == 'S':
                            a = (atom.GetIdx() + 1)
                            b = (nbr.GetIdx() + 1)
                            yield a, b
                            return True
        print(f"No matches. Calculating only global descriptors.")
        yield a, b

# local descriptors for calculated IDs

def get_local_descriptors(conf_ensemble, hal, car):
    print("Calculating local descriptors!")
    for conf in conf_ensemble.conformers:
        x = conf.xtb

        sterimol = Sterimol(conf_ensemble.elements, conf.coordinates, hal, car)
        conf.properties['Hal_C_bond_length'] = sterimol.bond_length
        conf.properties['L_value'] = sterimol.L_value
        conf.properties['B1_value'] = sterimol.B_1_value
        conf.properties['B5_value'] = sterimol.B_5_value
        conf.properties['L_value_buried'] = sterimol.bury().L_value
        conf.properties['B1_value_buried'] = sterimol.bury().B_1_value
        conf.properties['B5_value_buried'] = sterimol.bury().B_5_value

        conf.properties["Hal_VBur"] = BuriedVolume(conf_ensemble.elements, conf.coordinates, hal).fraction_buried_volume
        conf.properties["Carbon_VBur"] = BuriedVolume(conf_ensemble.elements, conf.coordinates,
                                                      car).fraction_buried_volume

        # Pyramidalization only for tetracoordinate atoms
        """
        conf.properties["Hal_Pyr_P"] = Pyramidalization(conf.coordinates, hal).P
        conf.properties["Hal_Pyr_P_angle"] = Pyramidalization(conf.coordinates, hal).P_angle
        conf.properties["Carbon_Pyr_P"] = Pyramidalization(conf.coordinates, car).P
        conf.properties["Carbon_Pyr_P_angle"] = Pyramidalization(conf.coordinates, car).P_angle
        """

        charges = x.get_charges()
        electro = x.get_fukui('electrophilicity')
        nucleo = x.get_fukui('nucleophilicity')

        conf.properties["Halogen_charge"] = charges[hal]
        conf.properties["Halogen_electrophilicity"] = electro[hal]
        conf.properties["Halogen_nucleophilicity"] = nucleo[hal]

        conf.properties["Carbon_charge"] = charges[car]
        conf.properties["Carbon_electrophilicity"] = electro[car]
        conf.properties["Carbon_nucleophilicity"] = nucleo[car]

        conf.properties["Bond_order"] = x.get_bond_order(car, hal)
        """
        lf = LocalForce(conf_ensemble.elements, conf.coordinates)
        #lf.load_file("hessian", "xtb", "hessian")
        lf.normal_mode_analysis()
        lf.detect_bonds()
        lf.compute_local()
        lf.compute_frequencies()
        """

    props = {}
    for key in conf_ensemble.get_properties().keys():
        props[f"{key}_Boltz"] = conf_ensemble.boltzmann_statistic(key)
        props[f"{key}_Emin"] = conf_ensemble.get_properties()[key][0]

    return pd.Series(props)


if __name__ == "__main__":

    # read data and calculate conformers, input file space separated SMILES strings .csv
    df = pd.read_csv('Testitesti.csv')
    data = df['SMILES']

    m3 = pd.DataFrame()

    for smiles in data:
        m = compute(smiles)
        for k in neighbors(smiles):
            halogen = k[0]
            carbon = k[-1]
            print(k)
            print(halogen)
            print(carbon)
            descs = get_descriptors(m)
            descs_local = get_local_descriptors(m, halogen, carbon)
            m3 = m3.append(descs_local, ignore_index=True)
            """elif halogen and carbon is None:
                descs = get_descriptors(m)
                m3 = m3.append(descs, ignore_index=True)"""

    # write to file
    print(f"Done! Writing to file.")
    result = m3.join(data)
    with open(f"Basetest.csv", "wb") as f:
        result.set_index('SMILES', inplace=True)
        result.to_csv(f)
