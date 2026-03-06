# Insertion Mapper

A Python-based bioinformatics tool designed to map transposon insertion sites to a target genome, derived from insertion-sequencing or arbitrary-primed PCR data. It identifies insertion sites by locating specific flanking donor sequences of the transposon, and uses k-mer based mapping to precisely anchor these sites within a GenBank-formatted genome.

---

## 🚀 Features

* **K-mer Mapping:** Customizable k-mer lengths for precise genomic localization.
* **Quality Control:** Automatic filtering of FASTQ reads based on mean Phred quality scores.
* **Error Correction:** Merges reads mapped to adjacent positions (within a configurable window) to account for sequencing noise.
* **Sequence Classification:** Categorize insertion sites by user-defined motifs (e.g., distinguishing between target-like and donor-like sequences).
* **Automated Plotting:** Generates insertion maps, distance distribution plots, and sequence logos.

---

## 💻 Usage

### Basic Command
```bash
python mapper_v3.py --fasta_file "reads.fastq" --genome "genome.gb" --donor "genomes/pJRP1063.gb" --insertion_end "left_end" --flanking_seq "GTTGTATTATCCTAGGGATA" --classify "Target-like=AGCCGCGGTAATAC, Donor-like=ACAGTATGTTGTAT"
```

### Full Example (General Use Case)
```bash
python mapper_v3.py \
  --fasta_file "raw_data/YFG6YX_fastq/YFG6YX_1_rerun_BHR06-C2-rep1_1.fastq" \
  --genome "genomes/MG1655_split16S.gb" \
  --donor "genomes/pJRP1063.gb" \
  --insertion_end "left_end" \
  --flanking_seq "gttgtattatcctagggata" \
  --output "BHR06_C2_rep1" \
  --min_mean_qual 10 \
  --mapping_kmer 20 \
  --classify "Target-like=AGCCGCGGTAATAC, Donor-like=ACAGTATGTTGTAT"
```

### Required Arguments
| Argument | Short | Description |
| :--- | :--- | :--- |
| --fasta_file | -f | Path to the input FASTA or FASTQ file of raw insertion mapping reads. |
| --genome | -g | Path to the target genome GenBank file. |
| --flanking_seq | -s | The sequence expected at the edge of the insertion - use 20-25bp. Must contain only A/T/C/G. |

### Optional Arguments
| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| --donor |	-d | None | Path to donor plasmid GenBank file. This can weed out donor plasmid contamination. | 
| --insertion_end | -e | left_end | End to analyze: left_end or right_end. |
| --mapping_kmer | -k | 25 | K-mer length used for genome mapping. ie. This many bp of genome adjacent to the insertion end must match for mapping call. |
| --context | -l | 7 | Length of sequence +/- around the insertion coordinate to consider for generating sequence logos and Needleman–Wunsch similarity plots. Please keep at 7, based on the IS621 14bp context window. Changing this may produce unpredicted results. |
| --min_mean_qual | -q | 10 | Min mean Phred quality (ignored for FASTA). |
| --merge_adjacent | -m | 5 | The bp window used to merge adjacent mappings. This corrects for sequencing errors that cause the insertion coordinate to be off by a few bp. |
| --output | -o | mapping_results | Output filename prefix. |
| --plot | -p | True | Generate maps and sequence logos. |
| --classify | -c | Target-like=AGCCGCGGTAATAC, Donor-like=ACAGTATGTTGTAT | Classify the insertion sites by expected sequence motifs. Please keep sequences 2 x context (ie 14bp) - otherwise results won't make sense. Format: label1=sequence1,label2=sequence2 | 
| --read_cutoff | -r | 0 | Remove reads that are less than X percent abundance from the logo plots | 
| --verbose | -v | 1 | Logging level (-v for info, -vv for debug). |

### Requirements
*   python >= 3.9
*   pandas
*   matplotlib
*   numpy
*   logomaker
*   biopython
*   Levenshtein
*   scikit-learn
*   umap-learn (optional, for UMAP plots)

### External Tools
*   [MAFFT](https://mafft.cbrc.jp/alignment/software/) (must be in your PATH)

Note: The --flanking_seq must contain only A, T, C, or G characters. The --mapping_kmer and --min_mean_qual must be positive integers.