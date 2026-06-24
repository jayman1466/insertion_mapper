#!/usr/bin/env python3

import argparse
import logging
import re
import sys
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from collections import Counter
import numpy as np
import csv

from plotter_v3 import plot_mappings, plot_mapping_distances

"""
General use case

python mapper.py --fasta_file "raw_data/YFG6YX_fastq/YFG6YX_1_rerun_BHR06-C2-rep1_1.fastq" --genome "genomes/MG1655_split16S.gb" --donor "genomes/pJRP1063.gb" --insertion_end "left_end" --flanking_seq "gttgtattatcctagggata" --output "BHR06_C2_rep1" --min_mean_qual 10 --mapping_kmer 20
"""

def valid_flanking(seq: str) -> str:
    """
    Ensure flanking sequence contains only A/T/C/G (case-insensitive).
    """
    if not re.fullmatch(r"[ATCGatcg]+", seq):
        raise argparse.ArgumentTypeError(
            "flanking_seq must contain only characters ATCGatcg"
        )
    return seq

def positive_int(value: str) -> int:
    """
    Ensure mapping_kmer is a positive integer.
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("mapping_kmer must be an integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("mapping_kmer must be a positive integer")
    return ivalue

def existing_file(path_str: str) -> Path:
    """
    Ensure the provided path exists and is a file.
    """
    p = Path(path_str)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path_str}")
    return p

def parse_dict(arg):
    """
    Parse the input from the --classify arguement into a dictionary.
    """
    # Converts 'key1=val1,key2=val2' into {'key1': 'val1', 'key2': 'val2'}
    if arg != '':
        arg = arg.strip('"')
        return dict((k.strip(), v.strip().upper()) for k, v in (pair.split('=') for pair in arg.split(',')))
    else:
        return {}
    
def parse_highlight(arg):
    """
    Parse the input from the --highlight arguement into a list.
    """
    # Converts 'seq1,seq2' into ['seq1', 'seq2']
    if arg != '':
        arg = arg.strip('"')
        return [seq.upper() for seq in arg.split(',')]
    else:
        return {}

def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Map insertion flanking sequences from a FASTA/FASTQ file to a target "
            "genome (GenBank), using a specified k-mer length."
        )
    )

    parser.add_argument(
        "--fasta_file",
        "-f",
        required=True,
        type=existing_file,
        help="Location of input FASTA or FASTQ file of the raw insertion mapping reads",
    )

    parser.add_argument(
        "--genome",
        "-g",
        required=True,
        type=existing_file,
        help="Location of target genome GenBank file",
    )

    parser.add_argument(
        "--donor",
        "-d",
        required=False,
        type=existing_file,
        help="Location of donor plasmid GenBank file. This can weed out donor plasmid contamination.",
    )

    parser.add_argument(
        "--insertion_end",
        "-e",
        choices=["left_end", "right_end"],
        default="left_end",
        help="Which end of insertion to analyze. Can be either 'left_end' or 'right_end'. (default: left_end)"
    )

    parser.add_argument(
        "--flanking_seq",
        "-s",
        required=True,
        type=valid_flanking,
        help="The sequence expected at the edge of the insertion - use 20-25bp. Must contain only A/T/C/G."
    )

    parser.add_argument(
        "--mapping_kmer",
        "-k",
        type=positive_int,
        default=25,
        help="K-mer length used for genome mapping. ie. This many bp of genome adjacent to the insertion end must match for mapping call. (default: 25)"
    )

    parser.add_argument(
        "--context",
        "-l",
        type=positive_int,
        required=False,
        default=7,
        help="Length of sequence +/- around the insertion coordinate to consider for generating sequence logos and Needleman–Wunsch similarity plots. Please keep at 7, based on the IS621 14bp context window. Changing this may produce unpredicted results. (default: 7)"
    )

    parser.add_argument(
        "--min_mean_qual",
        "-q",
        type=positive_int,
        default=10,
        help="Minimum mean Phred quality for FASTQ reads to be kept. (default: 10; ignored for FASTA)"
    )

    parser.add_argument(
        "--merge_adjacent",
        "-m",
        type=positive_int,
        default=5,
        help="The bp window used to merge adjacent mappings. This corrects for sequencing errors that cause the insertion coordinate to be off by a few bp. (default: 5)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=Path("mapping_results"),
        help="Output filename for mapping results (default: mapping_results)",
    )

    parser.add_argument(
        "--plot",
        "-p",
        type=bool,
        required=False,
        default=True,
        help="Plot insertion maps, distance plots, and sequence logos? (default: True)",
    )

    parser.add_argument(
        "--highlight",
        "-t",
        type=parse_highlight,
        required=False,
        default="",
        help="Highlight particular mapped insertion sites on insertion and distance plots",
    )

    parser.add_argument(
        "--classify",
        "-c",
        type=parse_dict,
        required=False,
        default="Target-like=AGCCGCGGTAATAC, Donor-like=ACAGTATGTTGTAT",
        help = "Classify the insertion sites by expected sequence motifs. Please keep sequences 2 x context (ie 14bp) - otherwise results won't make sense. Format: label1=sequence1,label2=sequence2 (default: Target-like=AGCCGCGGTAATAC, Donor-like=ACAGTATGTTGTAT).",
    )

    parser.add_argument(
        "--classify_limit",
        "-b",
        type=float,
        required=False,
        default=float('inf'),
        help = "Maximum Levenstein Distance allowed to classify the insertion sites by sequence motifs. If insertion site fails to classify, will be placed in a unclassified bin. (default: infinity)",
    )

    parser.add_argument(
        "--read_cutoff",
        "-r",
        type=float,
        required=False,
        default=0,
        help="Remove reads that are less than X percent abundance from the logo plots (default: 0)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase logging verbosity (-v = display major steps in pipeline, -vv = more details)",
    )

    return parser.parse_args(argv)

# Logging
# ---------------------------

def setup_logging(verbosity: int, output: str) -> None:
    """
    Configure logging level based on -v flags.
    """
    verbosity = 1
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        filename=f"{output}_log.log",
        filemode='w',
        level=level,
        encoding='utf-8',
        format="[%(levelname)s] %(message)s",
    )

# FASTA / FASTQ parsing
# ---------------------------

def detect_sequence_format(path: Path) -> str:
    """
    Detect whether the file is FASTA or FASTQ based on the first non-empty line.
    Returns 'fasta' or 'fastq'.
    """
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                logging.debug("Detected FASTA format")
                return "fasta"
            elif line.startswith("@"):
                logging.debug("Detected FASTQ format")
                return "fastq"
            else:
                break

    raise ValueError(f"Could not determine format of file: {path}")


def parse_fasta(path: Path):
    """
    Parse a FASTA file and return a list of records:
    [{"name": <str>, "sequence": <str>}, ...]
    """
    logging.info(f"Parsing FASTA file: {path}")
    records = []
    name = None
    seq_chunks = []

    with path.open() as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous record
                if name is not None:
                    sequence = "".join(seq_chunks)
                    records.append({"name": name, "sequence": sequence})
                    #logging.debug(f"FASTA record added: {name}, len={len(sequence)}")
                # start new
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

        # flush last record
        if name is not None:
            sequence = "".join(seq_chunks)
            records.append({"name": name, "sequence": sequence})
            #logging.debug(f"FASTA record added: {name}, len={len(sequence)}")

    logging.info(f"Total FASTA records parsed: {len(records)}")
    return records


def mean_phred_score(qual_str: str, phred_offset: int = 33) -> float:
    """
    Compute mean Phred score from a quality string (assumes Phred+33 by default).
    """
    if not qual_str:
        return 0.0
    scores = [ord(c) - phred_offset for c in qual_str]
    return sum(scores) / len(scores)


def parse_fastq_with_filter(path: Path, min_mean_qual: int):
    """
    Parse a FASTQ file, filter reads by mean quality, and return a list of records:
    [{"name": <str>, "sequence": <str>}, ...]
    """
    logging.info(f"Parsing FASTQ file with quality filter: {path}")
    records = []
    total = 0
    kept = 0

    with path.open() as fh:
        while True:
            header = fh.readline()
            if not header:
                break  # EOF

            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()

            if not (seq and plus and qual):
                logging.warning("FASTQ file appears truncated; stopping early.")
                break

            total += 1

            header = header.rstrip()
            seq = seq.rstrip()
            plus = plus.rstrip()
            qual = qual.rstrip()

            if not header.startswith("@"):
                logging.warning(
                    f"Unexpected FASTQ header format at read {total}: {header}"
                )
                continue

            name = header[1:].strip()
            m = mean_phred_score(qual)

            if m >= min_mean_qual:
                kept += 1
                records.append({"name": name, "sequence": seq})
                logging.debug(f"Kept FASTQ read {name}: len={len(seq)}, mean_qual={m:.2f}")

    logging.info(
        f"Total FASTQ reads: {total}, kept after quality filter (>= {min_mean_qual}): {kept}"
    )
    return records

def load_fasta_or_fastq(fasta_path: Path, min_mean_qual: int):
    """
    Load sequences from FASTA or FASTQ, returning a list of records:
    [{"name": <str>, "sequence": <str>}, ...].

    For FASTQ, reads are filtered by mean Phred quality.
    """
    fmt = detect_sequence_format(fasta_path)
    if fmt == "fasta":
        return parse_fasta(fasta_path)
    elif fmt == "fastq":
        return parse_fastq_with_filter(fasta_path, min_mean_qual)
    else:
        raise ValueError(f"Unsupported sequence format: {fmt}")


def load_genome_genbank(genome_path: Path):
    """
    Load target genome from a GenBank file using Biopython.

    Returns
    -------
    List[dict]
        A list of records, one per contig/sequence in the GenBank file:
        [
            {"name": <record_name>, "sequence": <sequence_string>},
            ...
        ]
    """
    logging.info(f"Loading sequence from GenBank file: {genome_path}")

    records = []
    with open(genome_path, "r") as handle:
        for rec in SeqIO.parse(handle, "genbank"):
            name = rec.name  # or rec.id / rec.description if you prefer
            seq = str(rec.seq)
            records.append({"name": name, "sequence": seq})
            logging.debug(
                f"GenBank record added: {name}, len={len(seq)}"
            )

    return records

# Extract flanking regions from reads
# ---------------------------

def extract_flanking_regions(records, flanking_seq: str, insertion_end: str, kmer_len: int):
    """
    Extract flanking sequences of length kmer_len either before (left_end)
    or after (right_end) the flanking_seq. Searches both forward and reverse
    complement orientations of the read sequence.

    Inputs:
    records = list of sequences as a dict {"name": name, "sequence": sequence}

    Returns
    -------
    List[dict]
        Each element:
        {
            "read_name": <str>,
            "sequence": <extracted sequence in forward orientation>,
            "strand": "+" or "-"
        }
    """
    logging.info(
        f"Extracting flanking regions using insertion_end={insertion_end}, "
        f"flanking_seq={flanking_seq}, kmer_len={kmer_len}"
    )

    flanking_seq_upper = flanking_seq.upper()
    flanks_to_map = []

    for rec in records:
        read_name = rec["name"]
        seq = rec["sequence"]
        seq_upper = seq.upper()

        # ----- SEARCH FORWARD STRAND -----
        idx = seq_upper.find(flanking_seq_upper)
        if idx != -1:
            strand = "+"
            orig_seq = seq
            flank_seq = None

            if insertion_end == "left_end":
                flank_end = idx
                flank_start = flank_end - kmer_len
                if flank_start >= 0:
                    flank_seq = orig_seq[flank_start:flank_end]

            elif insertion_end == "right_end":
                flank_start = idx + len(flanking_seq_upper)
                flank_end = flank_start + kmer_len
                if flank_end <= len(orig_seq):
                    flank_seq = orig_seq[flank_start:flank_end]

            if flank_seq:
                flanks_to_map.append(
                    {
                        "read_name": read_name,
                        "sequence": flank_seq,
                        "strand": strand,
                    }
                )
                continue  # no need to check reverse complement if forward matched

        # ----- SEARCH REVERSE COMPLEMENT STRAND -----
        seq_rev = str(Seq(seq_upper).reverse_complement())    # uppercase rc for searching
        orig_rev = str(Seq(seq).reverse_complement())          # original case rc for slicing

        idx_rev = seq_rev.find(flanking_seq_upper)
        if idx_rev != -1:
            strand = "-"
            flank_seq = None

            if insertion_end == "left_end":
                flank_end = idx_rev
                flank_start = flank_end - kmer_len
                if flank_start >= 0:
                    flank_seq = orig_rev[flank_start:flank_end]

            elif insertion_end == "right_end":
                flank_start = idx_rev + len(flanking_seq_upper)
                flank_end = flank_start + kmer_len
                if flank_end <= len(orig_rev):
                    flank_seq = orig_rev[flank_start:flank_end]

            if flank_seq:

                flanks_to_map.append(
                    {
                        "read_name": read_name,
                        "sequence": flank_seq,
                        "strand": strand,
                    }
                )

    logging.info(f"Total insertion reads extracted: {len(flanks_to_map)}")
    return flanks_to_map

# Map to genome and donor plasmid contigs
# ---------------------------

def map_flanks_to_genome(flanks, contig_name: str, contig_sequence: str,
                         insertion_end: str, mapping_kmer: int, context:int):

    results = []
    non_unique_matches = []

    contig_upper = contig_sequence.upper()
    contig_len = len(contig_sequence)
    contig_rc_upper = str(Seq(contig_upper).reverse_complement())

    for flank in flanks:
        read_name = flank["read_name"]
        flank_seq = flank["sequence"]
        flank_upper = flank_seq.upper()
        flank_len = len(flank_seq)

        match_coords = []

        # ---- Search forward strand ----
        start = 0
        while True:
            idx = contig_upper.find(flank_upper, start)
            if idx == -1:
                break
            
            if insertion_end == "left_end":
                mapping_coordinate = idx + flank_len
            elif insertion_end == "right_end":
                mapping_coordinate = idx - 2 #ACCOUNT FOR CORE HERE
            else:
                logging.error(f"Unsupported insertion_end value: {insertion_end}")
                continue

            # Enforce EXACTLY 14-nt mapping_sequence:
            # 7 bases before and 7 bases after mapping_coordinate must exist.
            start = mapping_coordinate - context
            end = mapping_coordinate + context

            if start < 0 or end > contig_len:
                # Not enough sequence for full 14-nt context -> treat as unusable
                continue

            mapping_sequence = contig_upper[start:end]
            
            match_coords.append([mapping_coordinate,mapping_sequence,"+"])
            start = idx + 1  # allow overlapping matches
            

        # ---- Search reverse complement ----
        start = 0
        while True:
            idx_rc = contig_rc_upper.find(flank_upper, start)
            if idx_rc == -1:
                break

            if insertion_end == "left_end":
                mapping_coordinate = idx_rc + flank_len
            elif insertion_end == "right_end":
                mapping_coordinate = idx_rc - 2 #ACCOUNT FOR CORE HERE
            else:
                logging.error(f"Unsupported insertion_end value: {insertion_end}")
                continue

            # Convert RC index to forward index:
            # rc is contig_upper[::-1], so position idx_rc in rc corresponds to
            # position (contig_len - (idx_rc + flank_len)) in forward sequence.
            idx_fwd = contig_len - (idx_rc)

            # Enforce EXACTLY 2X-nt mapping_sequence:
            # X bases before and X bases after mapping_coordinate must exist.
            start = mapping_coordinate - context
            end = mapping_coordinate + context

            if start < 0 or end > contig_len:
                # Not enough sequence for full 14-nt context -> treat as unusable
                continue

            mapping_sequence = contig_rc_upper[start:end]
            
            match_coords.append([idx_fwd,mapping_sequence,"-"])

            start = idx_rc + 1

        # ---- Decide on uniqueness ----
        if len(match_coords) > 1:
            non_unique_matches.append(
                {
                    "contig_name": contig_name,
                    "read_name": read_name,
                    "flank_sequence": flank_seq,
                    "match_coordinates": sorted(match_coords, key=lambda x: x[0]),
                }
            )
            continue

        # Exactly one unique match
        if len(match_coords) == 1:
            mapping_coordinate = match_coords[0][0]
            mapping_sequence = match_coords[0][1]
            strand = match_coords[0][2]
 
            # sanity check length
            if len(mapping_sequence) != 2*context:
                print("size_error")
                continue

            result = {
                "contig_name": contig_name,
                "read_name": read_name,
                "mapping_coordinate": mapping_coordinate,
                "mapping_sequence": mapping_sequence,
                "strand": strand
            }
            results.append(result)

    return results, non_unique_matches

# collapse the mapped reads to unique values
def tabulate_unique_mappings(mapping_df,contig_sizes:list, output):
    mapping_df_counted = pd.DataFrame()

    for contig in contig_sizes:
        contig_name = contig[0]
        
        #group by coordinate
        contig_df = mapping_df[mapping_df["contig_name"] == contig_name]
        this_df = contig_df.groupby("mapping_coordinate", as_index=False).agg({
            "contig_name": "first",
            "mapping_sequence": "first",
            "strand": "first"
            # ...
        })

        #add counts
        this_df["count"] = contig_df.groupby("mapping_coordinate").size().values

        mapping_df_counted = pd.concat([mapping_df_counted,this_df],ignore_index=True)
    
    # export csv of counted mappings
    mapping_df_counted.to_csv(f"{output}_unique_mapped_reads_counted.csv")

# Merge adjacent mappings into majority mapping
def group_by_mapping_coordinate(df, coord_col="mapping_coordinate",
                                contig_col="contig_name", max_diff=5):
    """
    Assign group_id such that:
    - Rows are grouped only within each contig_name
    - mapping_coordinate values must be within `max_diff` of neighbors
    """
    df = df.copy()

    group_ids = []
    current_group = 0

    # Process each contig independently
    for contig, subdf in df.groupby(contig_col):
        subdf = subdf.sort_values(coord_col)

        prev_coord = None
        for _, row in subdf.iterrows():
            coord = row[coord_col]

            if prev_coord is None or abs(coord - prev_coord) > max_diff:
                current_group += 1  # new group

            group_ids.append((row.name, current_group))
            prev_coord = coord

    # assign group ids into df
    for idx, gid in group_ids:
        df.loc[idx, "group_id"] = gid

    return df


def collapse_groups(df, coord_col="mapping_coordinate", strand_col="strand",
                    count_col="count", contig_col="contig_name",
                    seq_col="mapping_sequence"):
    """
    Collapse each group into a single row:
    - Sum counts within each group
    - mapping_coordinate = coordinate of row with highest counts
    - mapping_sequence = sequence of the row with highest counts
    - contig_name preserved
    """
    df = df.copy()

    # Index of the row with max counts for each group
    idx_max = df.groupby("group_id")[count_col].idxmax()

    # Representative rows include contig, coordinate, and sequence
    reps = df.loc[idx_max, ["group_id", coord_col, strand_col, contig_col, seq_col]].set_index("group_id")

    # Sum counts within each group
    summed = df.groupby("group_id")[count_col].sum()

    # Merge summed counts
    collapsed = reps.join(summed).reset_index(drop=True)

    return collapsed


# ---------------------------
# Main workflow
# ---------------------------

def main(argv=None) -> int:
    args = get_args(argv)
    setup_logging(args.verbose, args.output)

    logging.debug(f"CLI arguments: {args}")

    # Load inputs
    records = load_fasta_or_fastq(args.fasta_file, args.min_mean_qual)
    genome = load_genome_genbank(args.genome) #returns list of contigs
    if args.donor is not None:
        donor = load_genome_genbank(args.donor)
        donor_used = True
    else:
        donor = []

    # Extract flanking regions
    flanks = extract_flanking_regions(records, args.flanking_seq, args.insertion_end, args.mapping_kmer)
    all_flanks = pd.DataFrame(flanks)
    all_flanks.to_csv(f"{args.output}_flanks.csv")

    # Map to genome and donor plasmid
    mapping_data = [] # data of all uniquely mapped reads. Will be a dict of {"contig_name","read_name","mapping_coordinate","mapping_sequence","strand"}
    non_unique_matches_data = [] # data of reads mapped to multiple sites. Will be a dict of {"contig_name","read_name","flank_sequence","match_coordinates"}
    donor_data = [] # data of reads mapped to the donor plasmid. Will be a dict of {"contig_name","read_name","mapping_coordinate","mapping_sequence","strand"}
    contig_sizes = []  #list to hold data on genome contigs for plotting. Will be a list of lists:["contig_name", "contig_size"], ordered by contig size.
    donor_contig_sizes = [] #list to hold data on donor plasmid contigs for plotting

    for contig in genome:
        results, non_unique_matches = map_flanks_to_genome(flanks, contig["name"], contig["sequence"], args.insertion_end, args.mapping_kmer, context=args.context)
        mapping_data.extend(results)
        non_unique_matches_data.extend(non_unique_matches)

        contig_sizes.append([contig["name"],len(contig["sequence"])])

        logging.info(f"{len(results)} unique insertions were matched to {contig['name']}")
        logging.info(f"{len(non_unique_matches)} non-unique insertions were matched to {contig['name']}")

    for contig in donor:
        results, non_unique_matches = map_flanks_to_genome(flanks, contig['name'], contig["sequence"], args.insertion_end, args.mapping_kmer, context=args.context)
        donor_data.extend(results)
        donor_contig_sizes.append([contig["name"],len(contig["sequence"])])

        logging.info(f"{len(results)} unique insertions were matched to donor plasmid {contig['name']}")
        logging.info(f"{len(non_unique_matches)} non-unique insertions were matched to donor plasmid {contig['name']}")

    # Sort the contigs by size
    contig_sizes = sorted(contig_sizes, key=lambda x: x[1])
    if donor_used == True:
        donor_contig_sizes = sorted(donor_contig_sizes, key=lambda x: x[1])

    # Convert to dataframes
    mapping_df = pd.DataFrame(mapping_data)
    non_unique_matches_df = pd.DataFrame(non_unique_matches_data)
    if donor_used == True:
        donor_data_df = pd.DataFrame(donor_data)

    # Export csv of dataframes
    mapping_df.to_csv(f"{args.output}_genome_unique_mapped_reads.csv") #all reads that mapped once to the genome
    non_unique_matches_df.to_csv(f"{args.output}_genome_non-unique_mapped_reads.csv") #all reads that mapped multiple times to the genome
    if donor_used == True:
        donor_data_df.to_csv(f"{args.output}_donorplasmid_unique_mapped_reads.csv")

    # Export csv of contig sizes. Each row will be [contig_name, contig_size]
    with open(f"{args.output}_contigs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["contig_name","contig_size"])
        writer.writerows(contig_sizes)

    # Tabulate the data for reads mapping to the donor plasmid
    if donor_used == True and len(donor_data_df) > 0:
        tabulate_unique_mappings(donor_data_df, donor_contig_sizes, f"{args.output}_donorplasmid")
        # Merge adjacent reads - because of the low quality of nanopore reads, merge mappings within a few nt of each other into a consensus
        if args.merge_adjacent > 0:
            df = pd.read_csv(f"{args.output}_donorplasmid_unique_mapped_reads_counted.csv", header = 0)
            df_grouped = group_by_mapping_coordinate(df, max_diff=args.merge_adjacent)
            df_collapsed = collapse_groups(df_grouped)

            newfile = f"{args.output}_donorplasmid_unique_mapped_reads_counted.csv" #Currently just overwriting the old file
            df_collapsed.to_csv(newfile) #output a csv of the collapsed mappings 

    #see if anything was uniquely mapped to genome before continuing 
    if len(mapping_df) > 0:

        # Tabulate unique mappings. eg for each unique mapping site, count the number of instances observed
        tabulate_unique_mappings(mapping_df, contig_sizes, f"{args.output}_genome") #will output a csv with the headings idx,mapping_coordinate,contig_name,mapping_sequence,strand,count

        # Merge adjacent reads - because of the low quality of nanopore reads, merge mappings within a few nt of each other into a consensus
        if args.merge_adjacent > 0:
            df = pd.read_csv(f"{args.output}_genome_unique_mapped_reads_counted.csv", header = 0)
            df_grouped = group_by_mapping_coordinate(df, max_diff=args.merge_adjacent)
            df_collapsed = collapse_groups(df_grouped)

            newfile = f"{args.output}_genome_unique_mapped_reads_counted.csv" #Currently just overwriting the old file
            df_collapsed.to_csv(newfile) #output a csv of the collapsed mappings 
            reads_counted_csv = newfile #file that will be used for plotting stuff

        else:
            reads_counted_csv = f"{args.output}_unique_mapped_reads_counted.csv" #file that will be used for plotting stuff

        # Output insertion maps and sequence logo of all insertion sequences
        if args.plot == True:
            logging.info("Plotting Insertion Maps")
            plot_mappings(unique_mapped_reads_counted=reads_counted_csv, output=args.output, contigs=f"{args.output}_contigs.csv", highlight=args.highlight) #plot insertion map and sequence logo

            #only plot distances, NW score, and classified logo plots if --classify is provided
            if len(args.classify) > 0:
                logging.info("Plotting Distance Maps")
                plot_mapping_distances(output=args.output, method='pca', unique_mapped_reads_counted=reads_counted_csv, references=args.classify, classifylimit=args.classify_limit, read_cutoff= args.read_cutoff, highlight=args.highlight) 
            

        logging.info("Done.")
        return 0

    else:
        print(f"No unique mapping were found for {args.output}")

if __name__ == "__main__":
    sys.exit(main())