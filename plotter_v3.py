#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import logomaker

from types import SimpleNamespace
import sys
import subprocess
import tempfile
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import Levenshtein
from Bio import pairwise2

# Optional imports if these methods are chosen (umap, tsne)
try:
    import umap
except ImportError:
    umap = None

from sklearn.manifold import TSNE

# Function to align sequences with MAFFT before running Levenshtein calculations and PCA/tSNE/UMAP
def run_mafft(seqs: List[str], op: float = 0.05, ep: float = 0.05) -> List[str]:
    """
    Run MAFFT on sequences and return aligned sequences in the same order.

    op = MAFFT gap-open penalty (lower => more gaps).
    ep = MAFFT gap-extend penalty (lower => more gaps).


    Requires: mafft in PATH.
    """
    names = []
    i=1
    for seq in seqs: 
        names.append(f"seq_{i}")
        i += 1

    # Write temp FASTA
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        fasta_path = tmp.name
        for name, seq in zip(names, seqs):
            tmp.write(f">{name}\n{seq}\n")

    try:
        # Run MAFFT
        result = subprocess.run(
            ["mafft", "--globalpair", "--maxiterate", "1000", "--op", str(op), "--ep", str(ep), fasta_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
    except FileNotFoundError:
        raise RuntimeError("MAFFT not found. Please install MAFFT or omit --align.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MAFFT failed with error code {e.returncode}")

    # Parse aligned FASTA from stdout
    aligned = {}
    current_name = None
    current_seq = []

    for line in result.stdout.splitlines():
        if line.startswith(">"):
            if current_name is not None:
                aligned[current_name] = "".join(current_seq)
            current_name = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_name is not None:
        aligned[current_name] = "".join(current_seq)

    # Reconstruct in original order
    aligned_seqs = []
    for name in names:
        if name not in aligned:
            raise RuntimeError(f"Aligned output missing sequence '{name}'")
        aligned_seqs.append(aligned[name])

    return aligned_seqs 

def ref_columns_from_aligned_ref(aligned_ref: str, ref_len: int) -> List[int]:
    """
    Return the alignment column indices corresponding to the ref_len non-gap bases
    in aligned_ref (i.e., columns where aligned_ref != '-').
    """
    cols = [i for i, c in enumerate(aligned_ref) if c != "-"]
    if len(cols) < ref_len:
        raise ValueError(
            f"Aligned reference has only {len(cols)} non-gap bases; expected >= {ref_len}."
        )
    return cols[:ref_len]


def crop_alignment_by_columns(aligned_seqs: List[str], cols: List[int]) -> List[str]:
    """
    Extract a cropped window from each aligned sequence using specific alignment columns. This window is between the min and max of the cols, so will include the gap "-" characters.
    """
    cropped = []
    for s in aligned_seqs:
        if max(cols)+1 < len(s):
            cropped.append(s[min(cols):max(cols)+1])
        else:
            cropped.append(s[min(cols):])
    return cropped

# Compute Needleman–Wunsch alignments with affine gaps
def nw_score(s: str, t: str, match: int, mismatch: int, gap_open: int, gap_extend: int) -> int:
    """
    Global alignment (Needleman–Wunsch) score with affine gaps using Biopython pairwise2.
    NOTE: pairwise2 expects gap penalties as NEGATIVE numbers (costs), e.g. -1, -1.
    """
    return pairwise2.align.globalms(
        s, t,
        match, mismatch,
        gap_open, gap_extend,
        score_only=True
    )

# Functions to draw PCA or t-SNE or UMAP plots of the mapped insertion sites
# ---------------------------

def one_hot_encode(seq: str, SEQ_LEN: int) -> np.ndarray: # One hot encoding of the mapped sequences 
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    vec = np.zeros((SEQ_LEN, 4), dtype=float)
    for i, base in enumerate(seq):
        if base in mapping:
            vec[i, mapping[base]] = 1.0
    return vec.ravel()


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True) # Center the data by subtracting the mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False) # Singular Value Decomposition of the centered data
    return Xc @ Vt[:2].T # Return just the first 2 principle components


def embed_sequences(X: np.ndarray, method: str) -> np.ndarray: # Embed the data based on the method chosen
    """
    method: 'pca', 'umap', or 'tsne'
    """
    if method == "pca":
        return pca_2d(X)

    elif method == "umap":
        if umap is None:
            raise ImportError("UMAP is not installed. Run: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, n_neighbors=4, random_state=42, n_jobs=1)
        return reducer.fit_transform(X)

    elif method == "tsne":
        tsne = TSNE(n_components=2, perplexity=10, learning_rate="auto", init="pca", random_state=42)
        return tsne.fit_transform(X)

    else:
        raise ValueError(f"Unknown method: {method}")


def classify_sequences(seqs: List[str], references: dict, classifylimit: float) -> dict: # Classify the mapped sequences as being closer to one of the references by Levenshtein distance. Note - the input sequences here ARE NOT aligned with MAFFT
    #create holding dict of dicts
    classified_dict = {s:{"classification":None, "lev_distance":None}for s in seqs}
        

    for s in seqs:
        best_label = None
        min_dist = classifylimit+1 #levenshein distance must be <= classifylimit
        
        for label, ref_seq in references.items():
            dist = Levenshtein.distance(s, ref_seq)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        if best_label:
            classified_dict[s] = {"classification":best_label, "lev_distance":min_dist}
        else:
            #If sequence doesn't classify to any of the references within the limit
            classified_dict[s] = {"classification":"unknown", "lev_distance":None}
            
    return classified_dict


def plot_embedding(coords: np.ndarray, labels: List[str], method: str, output: str):
    #plot the actual PCR/tsne/umap plot. Use a blue marker for the target seqeunce and an orange marker for the donor sequence 
    colors = {"input": "#cccccc", "ref1": "#3b82f6", "ref2": "#e69f00"}
    markers = {"input": "o", "ref1": "o", "ref2": "o"}
    sizes = {"input": 20, "ref1": 50, "ref2": 50}
    label_text = {"input": "Mapped Insertion Sequences", "ref1": "Target Sequence", "ref2": "Donor Sequence"}

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 8

    fig, ax = plt.subplots(figsize=(3,3))
    for label in set(labels):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=label_text[label],
            marker=markers[label],
            linewidths=0.5,
            edgecolors='black',
            facecolors=colors[label],
            s=sizes[label]
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(f"{method.upper()} of mapped insertion sites")

    #legend
    leg = ax.legend()
    leg.get_frame().set_facecolor("gray")
    leg.get_frame().set_alpha(0.2)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0)
    
    plt.tight_layout()
    plt.savefig(f"{output}_insertion_{method.upper()}.svg", dpi=300)
    plt.close()


def plot_distance(input_df: pd.DataFrame, reference_seq: str, output: list, highlight: list, total_reads_in_dataset: int, ax=None):
    """
    Plot a bar graph and scatter plot where Needleman-Wunsch score from a reference sequence is on the x axis and the corresponding read counts are on the y axis
    
    df must have columns:
      - 'mapping_sequence' (str)
      - 'count' (int or float)
    reference_seq is the sequence to compare against with Levenshtein distance.
    output is a list ["filename_prefix", "filename_suffix"]
    """

    df = input_df.copy()


    # Compute Needleman-Wunsch distance for each mapping_sequence relative to reference
    #Penalties:
    '''
    match = 2 
    mismatch = -1 
    gap_open = -1
    gap_extend = -1 
    '''

    match = 2 
    mismatch = -0 
    gap_open = -1
    gap_extend = -0.5 

    # // Calculate the NW scores
    df["NW_score"] = df["mapping_sequence"].apply(
        lambda s: nw_score(s.upper(), reference_seq.upper(), match, mismatch, gap_open, gap_extend)
    )    

    # // Calculate read count as a percent of total mapped reads in dataset
    df["count_percent"] = df["count"] / total_reads_in_dataset * 100


    # // Bin the NW scores for cleaner barplots. We are expecting a max score of 28 (match=2 * 14bp sequence)
    # Bins are <12, 12–15, 16–19, 20–23, 24–27, ≥28

    labels = ["<12", "12–15", "16–19", "20–23", "24–27", "28"]

    df["NW_bin"] = pd.cut(
        df["NW_score"],
        bins=[-float("inf"), 11, 15, 19, 23, 27, float("inf")],
        labels=labels
    )

    counts = df.groupby("NW_bin")["count"].sum().reindex(labels, fill_value=0)
    counts_percent = df.groupby("NW_bin")["count_percent"].sum().reindex(labels, fill_value=0)

    # // Bin the Levenshtein scores for cleaner barplots. Group everything above 5
    df_copy = df.copy()
    df_copy['lev_grouped'] = np.where(df_copy['levenshtein_distance'] > 5, 6, df_copy['levenshtein_distance'])
    lev_result = df_copy.groupby('lev_grouped')['count'].sum().reset_index()
    lev_result_percent = df_copy.groupby('lev_grouped')['count_percent'].sum().reset_index()

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8


    #highlight colors for scatterplot
    colors = ["#D4405E" if seq in highlight else "#000000" for seq in df["mapping_sequence"].tolist()]

#// NW PLOTS 

    if ax is None:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(4,6))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    axes={"log": ax1, "linear": ax3, "percent": ax5}
    for scale, ax_name in axes.items():
    # // Make bar plot of NW score

        # Bar plot
        if scale == "percent":
            this_index = counts_percent.index
            this_values = counts_percent.values
        else:
            this_index = counts.index
            this_values = counts.values
        ax_name.bar(this_index, this_values, color = "#f2f2f2", edgecolor = "black", linewidth = 0.5)

        ax_name.set_xlabel("Needleman-Wunsch Score")

        if scale == "percent":
            ax_name.set_ylabel("Read Percent")
        else:
            ax_name.set_ylabel("Read count")

        ax_name.set_title(f"{output[1]} NW Similarity Score Hist")
        ax_name.set_yscale("linear" if scale == "percent" else scale)
        ax_name.invert_xaxis()

        # Make x-axis tick labels rotated
        ax_name.tick_params(axis='x', rotation=90)

        # Set a minimum value for the y axis to 5 for linear scale
        ymin, ymax = ax_name.get_ylim()
        if ymax < 5:
            ax_name.set_ylim(top=5)
        #normalize the axes for log scale
        if scale == "log":
            ax_name.set_ylim(top=5000)
            ax_name.set_ylim(bottom=0.5)
        elif scale == "percent":
            ax_name.set_ylim(top=100)  
            ax_name.set_ylim(bottom=0)
        else:
            ax_name.set_ylim(bottom=0)


        ax_name.spines['top'].set_visible(False)
        ax_name.spines['right'].set_visible(False)
        ax_name.spines['bottom'].set_linewidth(0.5)
        ax_name.spines['left'].set_linewidth(0.5)
        ax_name.tick_params(width=0.5)

    # // Scatter plot log, linear, and percent of NW score 
    axes={"log": ax2, "linear": ax4, "percent": ax6}
    for scale, ax_name in axes.items():
        if scale == "percent":
            this_values = df["count_percent"]
        else:
            this_values = df["count"]
        ax_name.scatter(df["NW_score"], this_values, c=colors, edgecolor="black", linewidth=0.5, s=40, clip_on = False)

        ax_name.set_xlabel("Needleman-Wunsch Score")

        if scale == "percent":
            ax_name.set_ylabel("Read Percent")
        else:
            ax_name.set_ylabel("Read count")

        ax_name.set_title(f"{output[1]} NW Similarity Score")
        ax_name.set_yscale("linear" if scale == "percent" else scale)
        ax_name.set_xlim(left=9)
        ax_name.set_xlim(right=30)
        ax_name.set_xticks([10, 13, 16, 19, 22, 25, 28])
        ax_name.invert_xaxis()

        # Set a minimum value for the y axis to 5
        ymin, ymax = ax_name.get_ylim()
        if ymax < 5:
            ax_name.set_ylim(top=5)
        #normalize the axes for log scale
        if scale == "log":
            ax_name.set_ylim(top=5000)
            ax_name.set_ylim(bottom=1)
        elif scale == "percent":
            ax_name.set_ylim(top=100)  
            ax_name.set_ylim(bottom=0)
        else:
            ax_name.set_ylim(bottom=0)


        ax_name.spines['top'].set_visible(False)
        ax_name.spines['right'].set_visible(False)
        ax_name.spines['bottom'].set_linewidth(0.5)
        ax_name.spines['left'].set_linewidth(0.5)
        ax_name.tick_params(width=0.5)

    plt.tight_layout()
    plt.savefig(f"{output[0]}_Needleman-Wunsch-score_{output[1]}.svg", dpi=300)
    plt.clf()
    plt.close()

# // Levenshtein plots

    # // Make bar plot of Levenshtein distance
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(4,6))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    axes={"log": ax1, "linear": ax3, "percent": ax5}
    for scale, ax_name in axes.items():
        # Bar plot
        if scale == "percent":
            this_index = lev_result_percent["lev_grouped"]
            this_values = lev_result_percent["count_percent"]
        else:
            this_index = lev_result["lev_grouped"]
            this_values = lev_result["count"]
        ax_name.bar(this_index, this_values, color = "#f2f2f2", edgecolor = "black", linewidth = 0.5)

        ax_name.set_xlabel("Levenshtein Distance")

        if scale == "percent":
            ax_name.set_ylabel("Read Percent")
        else:
            ax_name.set_ylabel("Read count")

        ax_name.set_title(f"{output[1]} Levenshtein Distance Hist")
        ax_name.set_yscale("linear" if scale == "percent" else scale)
        ax_name.set_xlim(left=-1)
        ax_name.set_xlim(right=7)
        ax_name.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax_name.set_xticklabels(['0', '1', '2', '3', '4', '5', '>5'])

        # Make x-axis tick labels rotated
        ax_name.tick_params(axis='x', rotation=90)

        # Set a minimum value for the y axis to 5
        ymin, ymax = ax_name.get_ylim()
        if ymax < 5:
            ax_name.set_ylim(top=5)
        #normalize the axes for log scale
        if scale == "log":
            ax_name.set_ylim(top=5000)
            ax_name.set_ylim(bottom=0.5)
        elif scale == "percent":
            ax_name.set_ylim(top=100)  
            ax_name.set_ylim(bottom=0)
        else:
            ax_name.set_ylim(bottom=0)

        ax_name.spines['top'].set_visible(False)
        ax_name.spines['right'].set_visible(False)
        ax_name.spines['bottom'].set_linewidth(0.5)
        ax_name.spines['left'].set_linewidth(0.5)
        ax_name.tick_params(width=0.5)

    # // Scatter plot log and linear of Levenstein distance 
    axes={"log": ax2, "linear": ax4, "percent": ax6}
    for scale, ax_name in axes.items():
        if scale == "percent":
            this_values = df["count_percent"]
        else:
            this_values = df["count"]
        ax_name.scatter(df["levenshtein_distance"], this_values, c=colors, edgecolor="black", linewidth=0.5, s=40, clip_on = False)

        ax_name.set_xlabel("Levenshtein Distance")

        if scale == "percent":
            ax_name.set_ylabel("Read Percent")
        else:
            ax_name.set_ylabel("Read count")

        ax_name.set_title(f"{output[1]} Levenshtein Distance")
        ax_name.set_yscale("linear" if scale == "percent" else scale)
        ax_name.set_xlim(left=-1)
        ax_name.set_xlim(right=6)
        ax_name.set_xticks([0,2,4,6])

        # Set a minimum value for the y axis to 5 for linear mode and fixed at 5000 for log scale 
        ymin, ymax = ax_name.get_ylim()
        if ymax < 5:
            ax_name.set_ylim(top=5)
        #normalize the axes for log scale
        if scale == "log":
            ax_name.set_ylim(top=5000)
            ax_name.set_ylim(bottom=1)
        elif scale == "percent":
            ax_name.set_ylim(top=100)  
            ax_name.set_ylim(bottom=0)
        else:
            ax_name.set_ylim(bottom=0)

        ax_name.spines['top'].set_visible(False)
        ax_name.spines['right'].set_visible(False)
        ax_name.spines['bottom'].set_linewidth(0.5)
        ax_name.spines['left'].set_linewidth(0.5)
        ax_name.tick_params(width=0.5)


    plt.tight_layout()
    plt.savefig(f"{output[0]}_levenshtein_distance_{output[1]}.svg", dpi=300)
    plt.clf()
    plt.close()

    return df



# Generate sequence logos
# ---------------------------

def make_sequence_logo(sequences, output: list, title=None):
    """
    Create a simple sequence logo (motif) image from a list of sequences.

    Parameters
    ----------
    sequences : list of str
        All sequences must be the same length. Typically DNA strings (A/C/G/T).
    output : list
        [Output filename prefix,suffix]
    title : str or None
        Optional plot title.
    """


    # Colors for bases
    color_scheme = {
        "A": "#f19ead",
        "C": "#d4405e",
        "G": "#b1cdfb",
        "T": "#3b82f6",
    }

    #convert sequences to uppercase
    sequences = [seq.upper() for seq in sequences]

    logo_df = logomaker.alignment_to_matrix(sequences, to_type='counts')

    fig, ax = plt.subplots(figsize = (3,0.75))

    logo = logomaker.Logo(logo_df, ax=ax, color_scheme = color_scheme, font_name = "arial", stack_order = "big_on_top", flip_below = False)
    logo.style_spines(spines=('bottom', 'left'), visible=True, color='black', linewidth=0.5, bounds=None)
    logo.style_spines(spines=('top', 'right'), visible=False, color='black', linewidth=0.5, bounds=None)
    ax.set_xticklabels([])
    ax.set_yticklabels([]) 

    logo.fig.tight_layout()
    fig.savefig(f"{output[0]}_sequence_logo_{output[1]}.svg", dpi=300)


# Output pca/tsne/umap, sequence logo broken down by "target-like" and "donor-like", and levenshtein distance bar graph
# ---------------------------
def plot_mapping_distances(output: str, method: str, unique_mapped_reads_counted: Path, references: dict, classifylimit: float, read_cutoff: float, highlight: list):
    """
    Further analyze the mapped sequences
    Inputs:
        unique_mapped_reads_counted: Path to csv file of unqiue mapped reads. Must have headers [idx,mapping_coordinate,contig_name,mapping_sequence,strand,count]
        output: Prefix for outputted files
        method: "pca", "umap", or "tsne"
        references: Dictionary of {label: sequence} for classification
    Outputs:

    """

    #open csv of mapped reads
    mapping_df_counted = pd.read_csv(unique_mapped_reads_counted, header=0)
    total_reads_in_dataset = mapping_df_counted["count"].sum()

    input_seqs = list(set(mapping_df_counted["mapping_sequence"].tolist()))

    # // Classify the sequences as closer to one of the references
    classified_seqs = classify_sequences(input_seqs, references, classifylimit)

    # Add classification data to the dataframe
    mapping_df_counted['classification'] = mapping_df_counted['mapping_sequence'].map(
        lambda x: classified_seqs.get(x, {}).get('classification')
    )

    mapping_df_counted['levenshtein_distance'] = mapping_df_counted['mapping_sequence'].map(
        lambda x: classified_seqs.get(x, {}).get('lev_distance')
    )

    # Image that is outputted if there is no data available
    svg_content = """<?xml version="1.0" encoding="UTF-8"?>
        <svg width="200" height="100"
            xmlns="http://www.w3.org/2000/svg"
            version="1.1">
        <text x="20" y="50"
                font-family="Arial"
                font-size="24"
                fill="black">No_Data</text>
        </svg>
        """
    
    output_df = pd.DataFrame()

    #all classifications that were made
    labels = list(set(mapping_df_counted["classification"].tolist()))
    
    # // Remove the unclassified from further analysis
    if "unknown" in labels:
        labels.remove("unknown")
        output_df = pd.concat([output_df, mapping_df_counted[mapping_df_counted["classification"] == "unknown"]])

    references_used = {label: ref_seq for label, ref_seq in references.items() if label in labels}

    # // Make logo plot for each classification category
    for label, ref_seq in references_used.items():
        
        # Filter DF to include subset of reads that match the current reference classification
        class_df = mapping_df_counted[mapping_df_counted['classification'] == label]

        # Remove reads that fall below the abundance cutoff for the logo plot
        total_count = class_df["count"].sum()
        cutoff = read_cutoff/100 * total_count
        
        class_df_cutoff = class_df[class_df["count"] >= cutoff].copy()

        # Align for the logo plot
        seq_list = [ref_seq] + class_df_cutoff['mapping_sequence'].to_list()
        aligned_seq_list = run_mafft(seq_list)
        
        if len(aligned_seq_list) > 1:
            class_df_cutoff["mapping_sequence_aligned"] = aligned_seq_list[1:]
        else:
            class_df_cutoff["mapping_sequence_aligned"] = np.nan

        # Weighted list for logo
        weighted_list = np.repeat(class_df_cutoff["mapping_sequence_aligned"].values, class_df_cutoff["count"].values).tolist()
        
        # Logo
        if len(weighted_list) > 0:
            make_sequence_logo(sequences=weighted_list, output=[output, label])
        else:
            with open(f"{output}_sequence_logo_{label}.svg", "w", encoding="utf-8") as f:
                f.write(svg_content)
                
        # // Make NW and Levensteins Distance plot
        class_df = plot_distance(input_df=class_df, reference_seq=ref_seq, output=[output, label], highlight = highlight, total_reads_in_dataset=total_reads_in_dataset)

        #update the output df with the new columns
        output_df = pd.concat([output_df, class_df])

    #export output_df to replace the original csv
    output_df.to_csv(unique_mapped_reads_counted, index=False)

    # PCA/Embedding logic commented out as it requires consistent alignment across all classes
    # X = np.vstack([one_hot_encode(s,SEQ_LEN) for s in all_seqs_aligned])
    # coords = embed_sequences(X, method)
    # plot_embedding(coords, labels, method, output)


# Output insertion maps and sequence logo of all sequences
# ---------------------------

def plot_mappings(unique_mapped_reads_counted: Path, output: str, contigs: Path, highlight: list) -> None:
    """
    Write mapping results to output files.
    Inputs:
        unique_mapped_reads_counted: Path to csv file of unqiue mapped reads. Must have headers [idx,mapping_coordinate,contig_name,mapping_sequence,strand,count]
        output: Prefix for outputted files
        contigs: Path to csv file of contigs. Must have headers [contig_name, contig_size] 
    Outputs:
        insetion_map.svg
        sequence_logo.svg
    """

    data_for_plotting = []
    mapping_df_counted = pd.read_csv(unique_mapped_reads_counted, header=0)
    contig_sizes = pd.read_csv(contigs, header=0)

    for idx,contig in contig_sizes.iterrows():
        contig_name = contig["contig_name"]
        contig_size = contig["contig_size"]

        this_df = mapping_df_counted[mapping_df_counted["contig_name"] == contig_name]

        data_for_plotting.append({
            "contig_name": contig_name,
            "contig_size": contig_size,
            "x_values": this_df["mapping_coordinate"].tolist(),
            "y_values": this_df["count"].tolist(),
            "colors": ["#d4405e" if seq in highlight else "#000000" for seq in this_df["mapping_sequence"].tolist()], #colors for the insertion maps with selective highlighting 
            "y_max": max(this_df["count"].tolist(),default=0)
            })



    # create insertion map
    y_extent = max(data_for_plotting, key=lambda item: item["y_max"])["y_max"]
    max_contig_size = max(data_for_plotting, key=lambda item: item["contig_size"])["contig_size"]

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 8

    weights = [max([data["contig_size"], 0.1*max_contig_size]) for data in data_for_plotting] #weights for x axis size. Have a minimum value set as 10% of max contig size
    names   = [data.get("contig_name", f"set{i}") for i, data in enumerate(data_for_plotting)]

    fig, axes = plt.subplots(
        1,
        len(data_for_plotting),
        figsize=(5, 2),
        gridspec_kw={"width_ratios": weights},
        squeeze=False,           # ensures we always get a 2D array for axes
    )
    axes = axes[0]

    i = 0
    for ax, entry, name in zip(axes, data_for_plotting, names):
        x = entry["x_values"]
        y = entry["y_values"]
        colors = entry["colors"]

        #background
        ax.set_facecolor('#f2f2f2')

        # Draw a vertical line for each point
        for xi, yi in zip(x, y):
            ax.plot([xi, xi], [0, yi], linewidth=0.5, linestyle="--", color="black")

        # Add a small circle at the top
        ax.scatter(x, y, s=40, color=colors, edgecolors='black', linewidth=0.5, clip_on = False)

        ax.set_title(f"{name}")
        ax.set_ylim(0, 1.2*y_extent) 
        ax.set_xlim(0, entry["contig_size"])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)

        xmin, xmax = ax.get_xlim()
        #ax.set_xticks([xmin, xmax])
        #ax.xaxis.set_major_formatter(
        #    mticker.FuncFormatter(lambda x, _: f"${x:.1e}$")
        #)

        if i != 0:
           ax.set_yticks([])
           ax.set_yticklabels([]) 
           ax.spines['left'].set_visible(False)
        #ax.set_xlabel("Coordinate")
        #ax.set_ylabel("Insertions")

        i =+ 1
    
    fig.tight_layout()
    fig.savefig(f"{output}_insertion_map.svg", dpi=300)
    plt.close(fig)  

    #Make the sequence logo - here, we weight the sequences by occurance and DO NOT pre-align them with MAFFT since that would probably lead to weird logo, since we're no grouping by Donor-like and Target-like
    #mapping_sequences = mapping_df_counted["mapping_sequence"].to_list()
    mapping_sequences_weghted = np.repeat(mapping_df_counted["mapping_sequence"].values, mapping_df_counted["count"].values).tolist()

    make_sequence_logo(sequences = mapping_sequences_weghted, output=[output,"allseqs"])
