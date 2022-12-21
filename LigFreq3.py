import pandas as pd
import numpy as np

from rdkit import Chem
from openbabel import pybel
import openbabel as ob
from rdkit import Chem
import pandas as pd
from io import StringIO
from xtb.interface import Calculator, Param

from morfeus.conformer import ConformerEnsemble
from morfeus import XTB as morf_xtb
from morfeus import SASA, Dispersion as morf_all
from morfeus import Sterimol, BuriedVolume, Pyramidalization, LocalForce, BiteAngle, Dispersion


def compute(file):
    
    ce = ConformerEnsemble.from_crest(file)
    
    for conf in ce:
        conf.xtb = morf_xtb(ce.elements, conf.coordinates, version='2')
        conf.morfeus_desc = morf_all(ce.elements, conf.coordinates)
        
    return ce
    
#electronic_temperature=1000
def carbonyl(xyz_file):
    
    mol = next(pybel.readfile("xyz", xyz_file))
   
    m = mol.OBMol
    
    ids = []
   
    for atom in ob.OBMolAtomIter(m):
        if atom.GetAtomicNum() == 28:
           
            ni = ids.append((atom.GetIdx()))
            for carbon_neighbour in ob.OBAtomAtomIter(atom):
               
                if carbon_neighbour.GetAtomicNum() == 6:
                    #carb = ids.append((carbon_neighbour.GetIdx()))
                   
                    for oxygen_neighbour in ob.OBAtomAtomIter(carbon_neighbour):
                        if oxygen_neighbour.GetAtomicNum() == 8:
                            carb = ids.append((carbon_neighbour.GetIdx()))
                            ox = ids.append((oxygen_neighbour.GetIdx()))
                       
    print(ids)            
    return ids

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
        props[f"{key}_ensemble_min"] = conf_ensemble.get_properties()[key].min()
        props[f"{key}_ensemble_max"] = conf_ensemble.get_properties()[key].max()

    return pd.Series(props)


def get_local_descriptors(conf_ensemble, ni, c1, o1, c2, o2):
    print("Calculating local descriptors!")
    for conf in conf_ensemble.conformers:
        x = conf.xtb
        h = conf.morfeus_desc
        """
        sterimol = Sterimol(conf_ensemble.elements, conf.coordinates, hal, car)
        conf.properties['Hal_C_bond_length'] = sterimol.bond_length
        conf.properties['L_value'] = sterimol.L_value
        conf.properties['B1_value'] = sterimol.B_1_value
        conf.properties['B5_value'] = sterimol.B_5_value
        conf.properties['L_value_buried'] = sterimol.bury().L_value
        conf.properties['B1_value_buried'] = sterimol.bury().B_1_value
        conf.properties['B5_value_buried'] = sterimol.bury().B_5_value
        """
        conf.properties['Ni_C1_bond_length'] = Sterimol(conf_ensemble.elements, conf.coordinates, ni, c1).bond_length
        conf.properties['Ni_C2_bond_length'] = Sterimol(conf_ensemble.elements, conf.coordinates, ni, c2).bond_length
        conf.properties['O1_C1_bond_length'] = Sterimol(conf_ensemble.elements, conf.coordinates, o1, c1).bond_length
        conf.properties['O2_C2_bond_length'] = Sterimol(conf_ensemble.elements, conf.coordinates, o2, c2).bond_length
        
        conf.properties['Ni_C_bond_length_average'] = (Sterimol(conf_ensemble.elements, conf.coordinates, ni, c1).bond_length + 
                                                       Sterimol(conf_ensemble.elements, conf.coordinates, ni, c2).bond_length) / 2
        conf.properties['O_C_bond_length_average'] = (Sterimol(conf_ensemble.elements, conf.coordinates, o1, c1).bond_length + 
                                                       Sterimol(conf_ensemble.elements, conf.coordinates, o2, c2).bond_length) / 2
        
        conf.properties["Ni_VBur"] = BuriedVolume(conf_ensemble.elements, conf.coordinates, ni).fraction_buried_volume
        
        conf.properties["C1_VBur"] = BuriedVolume(conf_ensemble.elements, conf.coordinates,
                                                      c1).fraction_buried_volume
        
        conf.properties["C2_VBur"] = BuriedVolume(conf_ensemble.elements, conf.coordinates,
                                                      c2).fraction_buried_volume
        
        conf.properties["C_VBur_average"] = (BuriedVolume(conf_ensemble.elements, conf.coordinates,
                                                      c2).fraction_buried_volume + BuriedVolume(conf_ensemble.elements, conf.coordinates,
                                                                                                    c1).fraction_buried_volume) / 2

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

        conf.properties["Ni_charge"] = charges[ni]
        conf.properties["Ni_electrophilicity"] = electro[ni]
        conf.properties["Ni_nucleophilicity"] = nucleo[ni]

        conf.properties["C1_charge"] = charges[c1]
        conf.properties["C1_electrophilicity"] = electro[c1]
        conf.properties["C1_nucleophilicity"] = nucleo[c1]
        
        conf.properties["C2_charge"] = charges[c2]
        conf.properties["C2_electrophilicity"] = electro[c2]
        conf.properties["C2_nucleophilicity"] = nucleo[c2]
        
        conf.properties["C_charge_average"] = (charges[c1] + charges[c2]) / 2
        conf.properties["C_electrophil_average"] = (electro[c1] + electro[c2]) / 2
        conf.properties["C_nucleophil_average"] = (nucleo[c1] + nucleo[c2]) / 2
        
        conf.properties["O1_charge"] = charges[o1]
        conf.properties["O1_electrophilicity"] = electro[o1]
        conf.properties["O1_nucleophilicity"] = nucleo[o1]
        
        conf.properties["O2_charge"] = charges[o2]
        conf.properties["O2_electrophilicity"] = electro[o2]
        conf.properties["O2_nucleophilicity"] = nucleo[o2]
        
        conf.properties["O_charge_average"] = (charges[o1] + charges[o2]) / 2
        conf.properties["O_electrophil_average"] = (electro[o1] + electro[o2]) / 2
        conf.properties["O_nucleophil_average"] = (nucleo[o1] + nucleo[o2]) / 2
        

        conf.properties["Bond_order_ni_c1"] = x.get_bond_order(ni, c1)
        conf.properties["Bond_order_ni_c2"] = x.get_bond_order(ni, c2)
        conf.properties["Bond_order_ni_c_average"] = (x.get_bond_order(ni, c1) + x.get_bond_order(ni, c2)) / 2
        
        conf.properties["Bond_order_o1_c1"] = x.get_bond_order(o1, c1)
        conf.properties["Bond_order_o2_c2"] = x.get_bond_order(o2, c2)
        conf.properties["Bond_order_o_c_average"] = (x.get_bond_order(o1, c1) + x.get_bond_order(o2, c2)) / 2
        
        conf.properties['SASA_ni'] = h.atom_areas[ni]
        
        conf.properties['p_int_ni'] = h.atom_p_int[ni]
        
        conf.properties['p_int_c1'] = h.atom_p_int[c1]
        conf.properties['p_int_c2'] = h.atom_p_int[c2]
        conf.properties['p_int_c_average'] = (h.atom_p_int[c1] + h.atom_p_int[c2]) / 2
        
        ba = BiteAngle(conf.coordinates, ni, c1, c2)
        conf.properties['Bite_angle_c1_c2_ni'] = ba.angle
        
        ba2 = BiteAngle(conf.coordinates, c1, ni, o1)
        conf.properties['Bite_angle_ni_c1_o1'] = ba2.angle
        
        ba3 = BiteAngle(conf.coordinates, c2, ni, o2)
        conf.properties['Bite_angle_ni_c2_o2'] = ba3.angle
        
        conf.properties['Bite_angle_ni_c_o_average'] = (ba3.angle + ba2.angle) / 2
        
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
        props[f"{key}_ensemble_min"] = conf_ensemble.get_properties()[key].min()
        props[f"{key}_ensemble_max"] = conf_ensemble.get_properties()[key].max()

    return pd.Series(props)

def freq(conf_ensemble, ligand, ni, c1, o1, c2, o2):
    print(f"Calculating freq descriptors for {ligand}!")
    n = 1
    for conf in conf_ensemble.conformers:
        #for n in range(1, len(conf_ensemble.conformers)+1):
            #print(n)
        
        
            
            
        
        hessian = f'/home/antti/Työpöytä/Optimoidut_GFN2/AQuick_opti/UudetEmakset/{ligand}/PROP/TMPCONF{n}/hessian'
            
        
        lf = LocalForce(conf_ensemble.elements, conf.coordinates)
        lf.load_file(hessian, "xtb", "hessian")
        lf.normal_mode_analysis()
        lf.detect_bonds()
        lf.compute_local()
        lf.compute_frequencies()
        
        conf.properties['Force_constant_Ni_C1'] = lf.get_local_force_constant([ni, c1])
        conf.properties['Force_constant_Ni_C2'] = lf.get_local_force_constant([ni, c2])
        conf.properties['Force_constant_Ni_C_Average'] = (lf.get_local_force_constant([ni, c2])+lf.get_local_force_constant([ni, c1])) / 2
        
        conf.properties['Force_constant_O1_C1'] = lf.get_local_force_constant([o1, c1])
        conf.properties['Force_constant_O2_C2'] = lf.get_local_force_constant([o2, c2])
        conf.properties['Force_constant_O_C_Average'] = (lf.get_local_force_constant([o2, c2])+lf.get_local_force_constant([o1, c1])) / 2
    
    
        conf.properties['Freq_Ni_C1'] = lf.get_local_frequency([ni, c1])
        conf.properties['Freq_Ni_C2'] = lf.get_local_frequency([ni, c2])
        conf.properties['Freq_Ni_C_Average'] = (lf.get_local_frequency([ni, c2])+lf.get_local_frequency([ni, c1])) / 2
        
        conf.properties['Freq_O1_C1'] = lf.get_local_frequency([o1, c1])
        conf.properties['Freq_O2_C2'] = lf.get_local_frequency([o2, c2])
        conf.properties['Freq_O_C_Average'] = (lf.get_local_frequency([o2, c2])+lf.get_local_frequency([o1, c1])) / 2
        n = n+1
    props = {}
    for key in conf_ensemble.get_properties().keys():
        props[f"{key}_Boltz"] = conf_ensemble.boltzmann_statistic(key)
        props[f"{key}_Emin"] = conf_ensemble.get_properties()[key][0]
        props[f"{key}_ensemble_min"] = conf_ensemble.get_properties()[key].min()
        props[f"{key}_ensemble_max"] = conf_ensemble.get_properties()[key].max()

    return pd.Series(props)     

    


if __name__ == "__main__":
    
    #fold = ['CMPhos', 'CyJohnPhos', 'JohnPhos', 'PhDavePhos', 'PtBu3', 'CPhos', 'BINAP', 'Xantphos']
                                                                                                
    #fold = ['AcaPhos', 'CgMe', 'DBU', 'DCEPhos', 'DCPP', 'DPPB', 'DPPE', 'DPPF', 'PAnis', 'PCF3','PhXPhos', 'PPh3','SPhos']                                                                                     
                                                                                                
    #fold = ['DBU', 'DPPF']
    
    #fold = ['BippyPhos', 'BuchPyr', 'Dimethamino', 'DPEPhos', 'Dppp', 'EPhos', 'EthaneP', 'Furyl', 'iPrPhosphite', 'NXant', 'PCy3', 'PentanePhenP', 'RuPhos', 'TBuCyP', 'TriFP']
    #file = r'/home/antti/Työpöytä/Optimoidut/DCPP/PROP'
    
    #fold = ['BippyPhos', 'BuchPyr', 'Dimethamino', 'DPEPhos', 'Dppp', 'EPhos', 'EthaneP']
    #fold = ['IPr_GFN1', 'IAd_GFN1', '2Oxazoline']
    
    fold = ['DBUN', 'DBUO', 'DBU66Ring', 'DBU68Ring', 'DBU77Ring', 'DBU86Ring']
    
    m3 = pd.DataFrame()
    m4 = pd.DataFrame()
    
    for ligand in fold:
        path = f'/home/antti/Työpöytä/Optimoidut_GFN2/AQuick_opti/UudetEmakset/{ligand}/PROP'
    
        m = compute(path)
    
        descs = get_descriptors(m)
    
        xyz_file = f'/home/antti/Työpöytä/Optimoidut_GFN2/AQuick_opti/UudetEmakset/{ligand}/crest_best.xyz'
    
        id = carbonyl(xyz_file)
    
        ni = id[0]
        c1 = id[1]
        o1 = id[2]
        c2 = id[3]
        o2 = id[4]
        print(id)
   
        descs_local = get_local_descriptors(m, ni, c1, o1, c2, o2)
        m3 = m3.append(descs_local, ignore_index=True)
        
        path2 = f'/home/antti/Työpöytä/Optimoidut_GFN2/AQuick_opti/UudetEmakset/{ligand}/'
        m2 = compute(path2)
        
        vib = freq(m2, ligand, ni, c1, o1, c2, o2)
        
        m4 = m4.append(vib, ignore_index=True)
        
    
    print(f"Done! Writing to file.")
    m3 = m3.join(m4)
    ind = pd.DataFrame(fold)
    ind.columns = ['Ligand name']
    
    result = m3.join(ind)
    with open(f"DBUBenzene_derivatives_v7.csv", "wb") as f:
        result.set_index('Ligand name', inplace=True)
        result.to_csv(f)
    
    