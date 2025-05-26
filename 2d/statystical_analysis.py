import pandas as pd
import matplotlib.pyplot as plt
import os

'''given the excel files produced by the manual annotations and the automatic predictions, this script performs statistical analysis on the estimated indexes.
It generates Bland-Altman plots for each index and saves them to a specified directory or displays them if no save path is provided.'''

def analysis(manual_path, automatic_path, patient_ids, save_path=None):
    '''Perform statistical analysis on estimated indexes.

    Args:
        manual_path (str): Path to the manual annotations Excel file.
        automatic_path (str): Path to the automatic predictions Excel file.
        patient_ids (list): List of patient IDs to filter the data.
        save_path (str, optional): Directory to save Bland-Altman plots. If None, plots are shown instead.
    
    Returns:
        None
    '''

    # Load Excel files
    annotations = pd.read_excel(manual_path)
    predictions = pd.read_excel(automatic_path)

    # Filter both datasets to those patient IDs
    annotations = annotations[annotations['id'].isin(patient_ids)]
    predictions = predictions[predictions['id'].isin(patient_ids)]

    # Merge on patient_id
    merged = pd.merge(annotations, predictions, on='id', suffixes=('_ann', '_pred'))

    # Get index columns (excluding id and path)
    index_columns = [col for col in predictions.columns if col not in ['id', 'path']]

    # Create output folder if needed
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Generate Bland-Altman plots for standard indices
    for col in index_columns:
        ann_col = f"{col}_ann"
        pred_col = f"{col}_pred"

        if ann_col not in merged.columns or pred_col not in merged.columns:
            continue  # skip if missing

        x = merged[[ann_col, pred_col]].dropna()
        mean = x.mean(axis=1)
        diff = x[ann_col] - x[pred_col]
        mean_diff = diff.mean()
        std_diff = diff.std()

        plt.figure(figsize=(6, 4))
        plt.scatter(mean, diff, alpha=0.6)
        plt.axhline(mean_diff, color='red', linestyle='--')
        plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle=':')
        plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle=':')
        plt.title(f'Bland-Altman Plot for {col}')
        plt.xlabel('Mean of Annotation and Prediction')
        plt.ylabel('Difference (Annotation - Prediction)')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            output_file = os.path.join(save_path, f"{col}.pdf")
            try:
                plt.savefig(output_file)
            except Exception as e:
                fallback_file = os.path.join(save_path, f"{col}.png")
                plt.savefig(fallback_file)
                print(f"Warning: Couldn't save {col}.pdf, saved as PNG instead. Error: {e}")
            plt.close()
        else:
            plt.show()

    # --- TAPSE calculation and plot ---
    try:
        # Compute tapse_ann and tapse_pred as row-wise means
        merged['tapse_ann'] = merged[['tapsefw_ann', 'tapsesep_ann']].mean(axis=1)
        merged['tapse_pred'] = merged[['tapsefw_pred', 'tapsesep_pred']].mean(axis=1)

        x = merged[['tapse_ann', 'tapse_pred']].dropna()
        mean = x.mean(axis=1)
        diff = x['tapse_ann'] - x['tapse_pred']
        mean_diff = diff.mean()
        std_diff = diff.std()

        plt.figure(figsize=(6, 4))
        plt.scatter(mean, diff, alpha=0.6)
        plt.axhline(mean_diff, color='red', linestyle='--')
        plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle=':')
        plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle=':')
        plt.title('Bland-Altman Plot for TAPSE')
        plt.xlabel('Mean of Annotation and Prediction (TAPSE)')
        plt.ylabel('Difference (Annotation - Prediction)')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            output_file = os.path.join(save_path, "tapse.pdf")
            try:
                plt.savefig(output_file)
            except Exception as e:
                fallback_file = os.path.join(save_path, "tapse.png")
                plt.savefig(fallback_file)
                print(f"Warning: Couldn't save tapse.pdf, saved as PNG instead. Error: {e}")
            plt.close()
        else:
            plt.show()
    except KeyError as e:
        print(f"Skipping TAPSE plot: missing column - {e}")


if __name__ == "__main__":
    manual_path = r"D:\mmissana\data/RV_PATIENTS/090525_Yu_manua_2D_and_3D_RV_TEE.xlsx"
    automatic_path = r"D:\mmissana\tapse_estimation\2d\results\best_unet_nofilter_projection_mean/best_unet.xlsx"
    patient_ids = [140, 141, 149, 160, 170, 184, 190, 198      , 100, 106, 111, 135, 199, 920]  # First ones are the one im sure of, other are the ones that weren't annotated
    save_path = r"D:\mmissana\tapse_estimation/2d/results/best_unet_nofilter_projection_mean"
    analysis(manual_path, automatic_path, patient_ids, save_path)