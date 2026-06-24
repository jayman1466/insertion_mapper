# mapper_v3.py

Maps insertion flanking sequences from a FASTA/FASTQ file to a target genome (GenBank format) using exact k-mer matching. For each read, the script locates a user-defined flanking sequence at the edge of an insertion element, extracts the adjacent genomic k-mer, and maps it to the target genome. Optionally filters out reads that map to a donor plasmid, merges adjacent insertion calls caused by sequencing errors, and outputs insertion site maps, distance plots, and sequence logos.

## Usage

```bash
python mapper_v3.py --fasta_file <reads> --genome <genome.gb> --flanking_seq <sequence> [options]
```

**Minimal example:**
```bash
python mapper_v3.py \
  --fasta_file raw_data/sample.fastq \
  --genome genomes/MG1655.gb \
  --flanking_seq gttgtattatcctagggata \
  --output my_sample
```

## Arguments

### Required

| Argument | Short | Description |
|----------|-------|-------------|
| `--fasta_file` | `-f` | Path to input FASTA or FASTQ file of raw insertion mapping reads |
| `--genome` | `-g` | Path to target genome GenBank (.gb) file |
| `--flanking_seq` | `-s` | Sequence at the edge of the insertion element (20–25 bp recommended). Must contain only A/T/C/G |

### Optional

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--donor` | `-d` | None | Path to donor plasmid GenBank file. Reads that map to this plasmid are separated out to identify donor contamination |
| `--insertion_end` | `-e` | `left_end` | Which end of the insertion to analyze. Choices: `left_end`, `right_end` |
| `--mapping_kmer` | `-k` | `25` | K-mer length (bp) used for genome mapping. This many bp of genomic sequence adjacent to the insertion end must exactly match for a mapping call |
| `--context` | `-l` | `7` | Length of sequence extracted on each side of the insertion coordinate for sequence logos and Needleman–Wunsch similarity plots. Keep at `7` for IS621 (14 bp context window); changing this may produce unpredictable results |
| `--min_mean_qual` | `-q` | `10` | Minimum mean Phred quality score for FASTQ reads to be retained. Ignored for FASTA input |
| `--merge_adjacent` | `-m` | `5` | Window size (bp) for merging nearby insertion coordinates. Corrects for sequencing errors that shift the insertion coordinate by a few bp. Set to `0` to disable |
| `--output` | `-o` | `mapping_results` | Prefix for all output files |
| `--plot` | `-p` | `True` | Generate insertion maps, distance plots, and sequence logos |
| `--highlight` | `-t` | None | Comma-separated list of sequences to highlight on insertion and distance plots. Example: `ACGT,TTGG` |
| `--classify` | `-c` | `Target-like=AGCCGCGGTAATAC,`<br>`Donor-like=ACAGTATGTTGTAT` | Classify insertion sites by sequence motif. Format: `label1=seq1,label2=seq2`. Sequences should be 2× context length (14 bp by default) |
| `--classify_limit` | `-b` | `inf` | Maximum Levenshtein distance allowed when classifying insertion sites by motif. Sites exceeding this threshold are placed in an unclassified bin |
| `--read_cutoff` | `-r` | `0` | Exclude insertion sites below this percent abundance from sequence logo plots |
| `--verbose` | `-v` | `1` | Logging verbosity. Use `-v` for major pipeline steps, `-vv` for detailed debug output |

## Output Files

All output files are prefixed with the value of `--output`.

| File | Description |
|------|-------------|
| `{output}_flanks.csv` | All flanking sequences extracted from reads before mapping |
| `{output}_genome_unique_mapped_reads.csv` | All reads that mapped to exactly one genome location |
| `{output}_genome_non-unique_mapped_reads.csv` | Reads that mapped to multiple genome locations (excluded from analysis) |
| `{output}_donorplasmid_unique_mapped_reads.csv` | Reads that mapped to the donor plasmid (only if `--donor` is provided) |
| `{output}_contigs.csv` | Genome contig names and sizes |
| `{output}_genome_unique_mapped_reads_counted.csv` | Final tabulated and merged insertion sites with read counts |
| `{output}_donorplasmid_unique_mapped_reads_counted.csv` | Tabulated donor plasmid insertion sites (only if `--donor` is provided) |
| `{output}_log.log` | Run log |
| Plots | Insertion maps, distance/PCA plots, and sequence logos (if `--plot True`) |

## Dependencies

- Python 3
- Biopython
- pandas
- matplotlib
- numpy
- `plotter_v3.py` (must be in the same directory)
