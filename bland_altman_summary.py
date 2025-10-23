import os
import pandas as pd

base_dir = r"2d/results_avg_4_pixels_no_190"
targets = [
    "tapsefw", "tapsesep", "rvfac", "rvad", "rvas", "rvldfw", "rvldsep",
    "rvlsfw", "rvlssep", "tadd", "tasd", "rvldmid", "rvlsmid",
    "rvlsffw", "rvlsfglobal", "rvlsfsep", "rvlsfmid", "tapse"
]
output_file = os.path.join(base_dir, "bland_altman_summary_no_sudden.xlsx")

# dizionario: target -> lista di dataframe
results = {t: [] for t in targets}

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    excel_path = os.path.join(folder_path, "bland_altman_stats.xlsx")
    if not os.path.exists(excel_path):
        print(f"File non trovato in {folder}")
        continue

    try:
        df = pd.read_excel(excel_path)
        for t in targets:
            row = df[df.iloc[:, 0] == t]
            if not row.empty:
                row = row.copy()
                row.insert(0, "subfolder", folder)
                results[t].append(row)
    except Exception as e:
        print(f"Errore leggendo {excel_path}: {e}")

# scrivi tutto in un unico Excel con più fogli
with pd.ExcelWriter(output_file) as writer:
    for t, rows in results.items():
        if rows:
            final_df = pd.concat(rows, ignore_index=True)
            final_df.to_excel(writer, sheet_name=t, index=False)

print(f"Creato {output_file}")

