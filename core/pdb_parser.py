import io
import numpy as np
from Bio.PDB import PDBParser

# Map to convert 3-letter codes to 1-letter for the comparison logic
AA_MAP = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def _to_one_letter(resname):
    return AA_MAP.get(resname.upper(), 'X')

def parse_pdb(file_handle):
    parser = PDBParser(QUIET=True)
    # Ensure we are reading the file correctly regardless of input type
    structure = parser.get_structure("protein", file_handle)
    
    coords = []
    residues = []
    
    for model in structure:
        for chain in model:
            for res in chain:
                # We only want standard protein residues with a Central Carbon (CA)
                if res.id[0] == " " and "ca" in [atom.get_id().lower() for atom in res]:
                    ca = res["CA"]
                    coords.append(ca.get_coord())
                    residues.append({
                        "resnum": res.id[1],          # FIX: matches mutation_detector.py
                        "one_letter": _to_one_letter(res.get_resname()), # FIX: for sequence matching
                        "resname": res.get_resname(), # For display in your table
                        "chain": chain.id             # For context
                    })
        break # Process only the first model
        
    return np.array(coords), residues, None, structure