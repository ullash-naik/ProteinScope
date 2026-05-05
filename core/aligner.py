import numpy as np

def kabsch_align(P, Q):
    """Aligns P to Q using the Kabsch algorithm."""
    P_center = np.mean(P, axis=0)
    Q_center = np.mean(Q, axis=0)
    P_centered = P - P_center
    Q_centered = Q - Q_center
    
    H = np.dot(P_centered.T, Q_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    t = Q_center - np.dot(P_center, R.T)
    P_aligned = np.dot(P, R.T) + t
    
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))
    return P_aligned, R, t, rmsd

def per_residue_rmsd(P_aligned, Q):
    """Calculates distance per atom."""
    return np.linalg.norm(P_aligned - Q, axis=1)

def calculate_tm_score(P_aligned, Q):
    """Calculates the TM-score (0 to 1). 1 indicates a perfect match."""
    L = len(Q)
    if L <= 15:
        return 0.0 # TM-score isn't reliable for tiny peptides
    
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    d0 = max(0.5, d0) # Prevent division by zero or negative d0
    
    devs = per_residue_rmsd(P_aligned, Q)
    tm_score = np.sum(1 / (1 + (devs / d0)**2)) / L
    return tm_score

def calculate_gdt_ts(devs):
    """Global Distance Test - Total Score (Percentage of aligned atoms)."""
    p1 = np.mean(devs <= 1.0) * 100
    p2 = np.mean(devs <= 2.0) * 100
    p4 = np.mean(devs <= 4.0) * 100
    p8 = np.mean(devs <= 8.0) * 100
    return (p1 + p2 + p4 + p8) / 4.0