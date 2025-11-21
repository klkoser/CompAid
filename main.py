import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

# -----------------------------
# Inputs / configuration
# -----------------------------
target_markers = ['CD45RA', 'CD8', 'CD38', 'CCR7', 'CD20']

# Pull base directory from dev.env
base_dir = '/service'
input_dir = os.environ['INPUT_DIR']
pdf_path = f'{base_dir}/outputs/marker_pairplots_by_subject_with_labels.pdf'
model_path = f'{input_dir}/model.pth'
max_subjects = 3  # Only process first 3 subjects

# -----------------------------
# Load trained model
# -----------------------------
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"✓ Model loaded from {model_path}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Helper: Generate density plot and get model prediction
# -----------------------------
def get_model_prediction(data_df, x_marker, y_marker):
    """
    Generate a density plot for the given marker pair and return model prediction
    """
    # Normalize function
    def normalize(data, column):
        df_normalize = data[column]
        min_val = df_normalize.min()
        max_val = df_normalize.max()
        if max_val == min_val:
            return pd.Series([0.0] * len(df_normalize), index=df_normalize.index)
        return (df_normalize - min_val) / (max_val - min_val)
    
    # Create density matrix
    data_df_selected = data_df[[x_marker, y_marker]].copy()
    
    cofactor = 5000
    data_df_selected[x_marker] = np.arcsinh(data_df_selected[x_marker] / cofactor)
    data_df_selected[y_marker] = np.arcsinh(data_df_selected[y_marker] / cofactor)
    
    data_df_selected[x_marker] = normalize(data_df_selected, x_marker) * 100
    data_df_selected[y_marker] = normalize(data_df_selected, y_marker) * 100
    
    # Create density plot
    density = np.zeros((101, 101))
    data_df_selected = data_df_selected.round(0)
    data_df_selected_count = data_df_selected.groupby([x_marker, y_marker]).size().reset_index(name="count")
    
    coord = data_df_selected_count[[x_marker, y_marker]]
    coord = coord.to_numpy().round(0).astype(int).T
    coord = list(zip(coord[1], coord[0]))
    replace = data_df_selected_count[['count']].to_numpy()
    for index, value in zip(coord, replace):
        density[index] = value
    
    index_x = np.linspace(0, 100, 101).round(2)
    index_y = np.linspace(0, 100, 101).round(2)
    df_plot = pd.DataFrame(density, index_y, index_x)
    df_plot = df_plot.iloc[::-1]
    
    # Create image
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    jet_cmap = plt.cm.jet
    jet_cmap.set_under('white')
    
    plot_data = df_plot.values
    masked_data = np.ma.masked_where(plot_data <= 0, plot_data)
    
    ax.imshow(masked_data, cmap=jet_cmap, aspect='auto', 
              vmax=df_plot.max().max()/2, vmin=0.1, 
              interpolation='nearest', origin='upper')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    # Load as PIL image and get prediction
    img = Image.open(buf).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
    
    return pred, prob

# -----------------------------
# Clean up marker list (remove duplicates but keep order)
# -----------------------------
markers = list(dict.fromkeys(target_markers))  # e.g. ['CD45RA', 'CD8', 'CD38', 'CCR7', 'CD20']
n_markers = len(markers)
print("Using markers:", markers)

# -----------------------------
# Load your data_df and build subject list
# -----------------------------
data_df = pd.read_csv(f'{input_dir}/subset_dataset_labels.csv')

# Extract unique subjects: remove marker pairs and get base subject identifier
def extract_subject(filename):
    """Extract subject identifier from filename, removing marker pair suffix"""
    parts = filename.split('_')
    # Last two parts are marker names, remove them
    return '_'.join(parts[:-2])

subjects = data_df['filename'].apply(extract_subject).unique().tolist()
subjects = subjects[:max_subjects]  # Limit to first N subjects
print(f"Processing {len(subjects)} subjects:", subjects)

# Filter data to only selected subjects
data_df['subject'] = data_df['filename'].apply(extract_subject)
data_df = data_df[data_df['subject'].isin(subjects)].copy()
print(f"Total rows to process: {len(data_df)}")

# -----------------------------
# Get model predictions for each marker pair
# -----------------------------
print("\nGenerating predictions for each marker pair...")
data_df['true_label'] = data_df['label'].copy()  # Save the true label from CSV
data_df['predicted_label'] = 0
data_df['probability'] = 0.0

for idx, row in data_df.iterrows():
    filename = row['filename']
    x_marker = row['x_axis']
    y_marker = row['y_axis']
    
    # Extract subject name: remove the marker pair suffix (_CD8_CD45RA, etc.)
    parts = filename.split('_')
    subject_with_status = '_'.join(parts[:-2])
    
    # Find the actual CSV file
    csv_path = f'{input_dir}/{subject_with_status}.csv'
    
    if not os.path.exists(csv_path):
        print(f"  WARNING: File not found: {csv_path}, skipping...")
        continue
    
    # Load raw data for this subject
    raw_data = pd.read_csv(csv_path)
    
    # Get model prediction
    pred, prob = get_model_prediction(raw_data, x_marker, y_marker)
    data_df.at[idx, 'predicted_label'] = pred
    data_df.at[idx, 'probability'] = prob
    
    true_status = "SPILLOVER" if row['true_label'] == 1 else "CLEAN"
    pred_status = "SPILLOVER" if pred == 1 else "CLEAN"
    match = "✓" if row['true_label'] == pred else "✗"
    
    print(f"  {match} {os.path.basename(csv_path)}: {x_marker} vs {y_marker} | True: {true_status} | Pred: {pred_status} (prob={prob:.3f})")

print("\n✓ All predictions computed")

# Calculate accuracy
correct = (data_df['true_label'] == data_df['predicted_label']).sum()
total = len(data_df)
accuracy = correct / total * 100 if total > 0 else 0

print(f"\nPrediction Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"True Spillover: {(data_df['true_label'] == 1).sum()}")
print(f"Predicted Spillover: {(data_df['predicted_label'] == 1).sum()}")


# -----------------------------
# Helper: draw n x n grid for one subject
# -----------------------------
def plot_subject_pairgrid(sub_df, subject_with_status, markers):
    """
    sub_df: DataFrame containing rows for one subject
    subject_with_status: Full subject identifier including _clean or _bad
    markers: list of marker column names
    """
    n = len(markers)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
    title = f"Subject: {subject_with_status}\n"
    title += "Lower Triangle: T:✗=True Spillover, P:✗=Predicted Spillover | Red=Pred Spillover, Green=Pred Clean | ✓=Match, ✗=Mismatch"
    fig.suptitle(title, fontsize=14)

    # Load raw per-subject data
    csv_path = f'{input_dir}/{subject_with_status}.csv'
    if not os.path.exists(csv_path):
        print(f"  WARNING: File not found: {csv_path}")
        return None
    
    data = pd.read_csv(csv_path)
    data[markers] = np.arcsinh(data[markers] / 5000)

    for i, y_marker in enumerate(markers):
        for j, x_marker in enumerate(markers):
            ax = axes[i, j]

           # ---------- LOWER TRIANGLE: COLOR BLOCK WITH LABELS ----------
            if j < i:
                # Check both (x_marker, y_marker) and (y_marker, x_marker)
                match = sub_df[
                    ((sub_df['x_axis'] == x_marker) & (sub_df['y_axis'] == y_marker)) |
                    ((sub_df['x_axis'] == y_marker) & (sub_df['y_axis'] == x_marker))
                ]

                if not match.empty:
                    row = match.iloc[0]
                    true_label = row['true_label']
                    pred_label = row['predicted_label']
                    
                    # Color based on prediction
                    if pred_label == 1:
                        color = 'red'
                    else:
                        color = 'lightgreen'
                    
                    ax.set_facecolor(color)
                    
                    # Add text showing true vs predicted
                    true_text = "T:✓" if true_label == 0 else "T:✗"
                    pred_text = "P:✓" if pred_label == 0 else "P:✗"
                    
                    # Show if they match
                    if true_label == pred_label:
                        match_symbol = "✓"
                        text_color = 'darkgreen' if pred_label == 0 else 'darkred'
                    else:
                        match_symbol = "✗"
                        text_color = 'black'
                    
                    ax.text(0.5, 0.7, f"{true_text}", 
                           ha='center', va='center', fontsize=14, 
                           transform=ax.transAxes, fontweight='bold')
                    ax.text(0.5, 0.3, f"{pred_text}", 
                           ha='center', va='center', fontsize=14, 
                           transform=ax.transAxes, fontweight='bold')
                    ax.text(0.5, 0.5, match_symbol, 
                           ha='center', va='center', fontsize=20, 
                           transform=ax.transAxes, color=text_color, fontweight='bold')
                else:
                    ax.set_facecolor('lightgray')
                
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue


            # ---- upper triangle ----
            if j < i:
                ax.axis('off')
                continue

            # Extract data for this pair
            x = data[x_marker].values
            y = data[y_marker].values

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(x, bins=50, alpha=0.7)
                ax.set_ylabel('')
            elif j > i:
                # Upper triangle: 2D density with white background

                # Compute histogram
                counts, xedges, yedges = np.histogram2d(x, y, bins=250)

                # Mask zeros
                counts_masked = np.ma.masked_where(counts == 0, counts)

                # Jet colormap with white background for masked values
                cmap = plt.cm.get_cmap('jet').copy()
                cmap.set_bad(color='white')

                # Draw density
                ax.pcolormesh(xedges, yedges, counts_masked.T, cmap=cmap, shading='auto')
                ax.set_facecolor('white')

            # ---------- LABELS ----------
            # Bottom row x-label (only for diagonal and upper-triangle)
            if i == n - 1 and j >= i:
                ax.set_xlabel(x_marker)
            else:
                ax.set_xticklabels([])

            # Left column y-label
            if j == 0:
                ax.set_ylabel(y_marker)
            else:
                ax.set_yticklabels([])

            # ---------- NEW: TOP ROW MARKER NAME ----------
            if i == 0:
                ax.set_title(x_marker, fontsize=12, pad=6)

            # ---------- NEW: RIGHT COLUMN MARKER NAME ----------
            if j == n - 1:
                ax.annotate(y_marker,
                            xy=(1.05, 0.5),
                            xycoords='axes fraction',
                            va='center',
                            ha='left',
                            fontsize=12,
                            rotation=90)
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for title
    return fig

# -----------------------------
# Create multipage PDF
# -----------------------------
with PdfPages(pdf_path) as pdf:
    for subj in subjects:
        # Filter rows for this subject
        sub_df = data_df[data_df['filename'].apply(extract_subject) == subj]
        
        if len(sub_df) == 0:
            print(f"No data found for subject: {subj}")
            continue
        
        print(f"\nProcessing subject: {subj}")
        fig = plot_subject_pairgrid(sub_df, subj, markers)
        
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print(f"  Skipping {subj} - no figure generated")

print(f"\n{'='*80}")
print(f"✓ Saved PDF to {pdf_path}")
print(f"{'='*80}")
print(f"\nPDF Legend:")
print(f"  Lower Triangle Squares:")
print(f"    - Background Color: RED = Predicted Spillover, GREEN = Predicted Clean")
print(f"    - T:✗ = True label is Spillover (1), T:✓ = True label is Clean (0)")
print(f"    - P:✗ = Predicted Spillover (1), P:✓ = Predicted Clean (0)")
print(f"    - ✓ (green) = Prediction matches truth")
print(f"    - ✗ (black) = Prediction doesn't match truth")