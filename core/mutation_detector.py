import pandas as pd

# Define a simulated active site for the hackathon demo 
# (In a real scenario, this comes from a database, but here we simulate it around residue 45-65)
ACTIVE_SITE_START = 45
ACTIVE_SITE_END = 65

def detect_mutations(res_ref, res_var, deviations, threshold=2.0):
    data = []
    for i, (r, v, dev) in enumerate(zip(res_ref, res_var, deviations)):
        is_mut = r["one_letter"] != v["one_letter"]
        is_hotspot = dev > threshold
        
        # Check if the amino acid is inside the drug binding pocket
        pos = int(r["resnum"])
        in_active_site = ACTIVE_SITE_START <= pos <= ACTIVE_SITE_END
        
        # If it's a hotspot AND inside the active site, the drug won't work!
        resistance_risk = is_hotspot and in_active_site

        data.append({
            "Position": pos,
            "Chain": r["chain"],
            "Ref AA": r["one_letter"],
            "Var AA": v["one_letter"],
            "Mutation": f"{r['one_letter']}{pos}{v['one_letter']}" if is_mut else "-",
            "Deviation (Å)": round(dev, 3),
            "Is Mutation": is_mut,
            "Is Hotspot": is_hotspot,
            "In Active Site": in_active_site,
            "Resistance Risk": resistance_risk
        })
    return pd.DataFrame(data)