# Essential Linux Commands for Bioinformatics

A practical reference guide for commonly used Linux commands in bioinformatics workflows.

---

## Table of Contents
1. [Text Processing](#text-processing)
2. [Pattern Matching](#pattern-matching)
3. [File Operations](#file-operations)
4. [Counting and Statistics](#counting-and-statistics)
5. [Compression](#compression)
6. [Advanced Tips](#advanced-tips)

---

## Text Processing

### awk / gawk

AWK is a powerful text processing language. GNU AWK (gawk) is the most common implementation with additional features.

**Basic Syntax:**
```bash
awk 'pattern {action}' file
```

**Common Operations:**

```bash
# Print specific columns
awk '{print $1, $3}' file.txt

# Print columns with custom separator
awk -F'\t' '{print $1, $2}' file.tsv

# Filter rows by condition
awk '$3 > 100' data.txt

# Calculate column sum
awk '{sum += $2} END {print sum}' numbers.txt

# Count lines matching condition
awk '$5 == "+" {count++} END {print count}' genes.gff

# Print header and filter
awk 'NR==1 || $4 > 50' results.txt

# Multiple field separators
awk -F'[:\t]' '{print $1, $3}' file.txt
```

**Bioinformatics Examples:**

```bash
# Extract sequences from FASTA by length
awk '/^>/ {if (seq) {if (length(seq) > 1000) print header "\n" seq}; header=$0; seq=""; next} {seq=seq$0} END {if (length(seq) > 1000) print header "\n" seq}' sequences.fasta

# Calculate GC content from FASTA
awk '/^>/ {if (seq) {gc=gsub(/[GCgc]/,"",seq); print header, gc/length(seq)*100}; header=$0; seq=""; next} {seq=seq$0} END {gc=gsub(/[GCgc]/,"",seq); print header, gc/length(seq)*100}' sequences.fasta

# Convert GFF to BED format
awk -F'\t' '{print $1, $4-1, $5, $9, $6, $7}' OFS='\t' genes.gff

# Filter SAM file by mapping quality
awk '$5 >= 30' alignments.sam

# Calculate average read depth from BED
awk '{sum+=$4; count++} END {print sum/count}' coverage.bed
```

**AWK vs GAWK Differences:**

| Feature | AWK (POSIX) | GAWK (GNU) |
|---------|-------------|------------|
| Regex intervals | Not standard | `{n,m}` supported |
| Multiple input separators | Limited | `-F'[sep1sep2]'` |
| BEGINFILE/ENDFILE | No | Yes |
| Array sorting | No | `asort()`, `asorti()` |
| Two-way pipes | No | Yes (`|&`) |
| Include files | No | `@include` |
| Bitwise operations | No | Yes (`and()`, `or()`, etc.) |

**GAWK-specific features:**
```bash
# Sort array (GAWK only)
gawk '{arr[$1]=$2} END {n=asort(arr,sorted); for(i=1;i<=n;i++) print sorted[i]}'

# BEGINFILE for processing multiple files (GAWK only)
gawk 'BEGINFILE {print "Processing:", FILENAME} {print}' *.txt
```

---

### sed

Stream editor for filtering and transforming text.

**Basic Syntax:**
```bash
sed 's/pattern/replacement/flags' file
```

**Common Operations:**

```bash
# Substitute first occurrence per line
sed 's/old/new/' file.txt

# Substitute all occurrences (global)
sed 's/old/new/g' file.txt

# Delete lines matching pattern
sed '/pattern/d' file.txt

# Delete specific line numbers
sed '5d' file.txt              # Delete line 5
sed '2,5d' file.txt            # Delete lines 2-5

# Print specific lines
sed -n '10,20p' file.txt       # Print lines 10-20
sed -n '/pattern/p' file.txt   # Print matching lines

# In-place editing
sed -i 's/old/new/g' file.txt  # Linux
sed -i '' 's/old/new/g' file.txt  # macOS

# Multiple operations
sed -e 's/foo/bar/' -e 's/baz/qux/' file.txt
```

**Bioinformatics Examples:**

```bash
# Remove FASTQ quality lines
sed -n '1~4p;2~4p' reads.fastq

# Convert FASTQ to FASTA
sed -n '1~4s/^@/>/p;2~4p' reads.fastq

# Remove version from sequence IDs
sed 's/\.[0-9]*$//' sequences.fasta

# Add prefix to sequence names
sed 's/^>/>sample1_/' sequences.fasta

# Extract sequences between line numbers
sed -n '100,200p' large_file.fasta

# Remove empty lines
sed '/^$/d' data.txt

# Replace tabs with commas
sed 's/\t/,/g' data.tsv
```

---

### grep

Search for patterns in text.

**Basic Syntax:**
```bash
grep [options] pattern file
```

**Common Options:**

```bash
# Case-insensitive search
grep -i "pattern" file.txt

# Count matching lines
grep -c "pattern" file.txt

# Show line numbers
grep -n "pattern" file.txt

# Invert match (non-matching lines)
grep -v "pattern" file.txt

# Recursive search in directories
grep -r "pattern" /path/to/dir

# Search multiple files
grep "pattern" *.txt

# Extended regex
grep -E "pattern1|pattern2" file.txt
# OR
egrep "pattern1|pattern2" file.txt

# Fixed strings (no regex)
grep -F "literal.string" file.txt
# OR
fgrep "literal.string" file.txt

# Show context (lines before/after)
grep -A 3 "pattern" file.txt  # 3 lines after
grep -B 2 "pattern" file.txt  # 2 lines before
grep -C 2 "pattern" file.txt  # 2 lines before and after

# Only show matching part
grep -o "pattern" file.txt

# Multiple patterns from file
grep -f patterns.txt data.txt

# Quiet mode (exit status only)
grep -q "pattern" file.txt && echo "Found"
```

**Bioinformatics Examples:**

```bash
# Extract sequences by ID
grep -A 1 ">gene_name" sequences.fasta

# Count sequences in FASTA
grep -c "^>" sequences.fasta

# Find genes on positive strand
grep "+" genes.gff

# Extract reads with quality issues
grep -v "^@" reads.fastq | grep -c "N"

# Search for motif
grep -io "TATAAA" genome.fasta

# Find all genes in specific chromosome
grep "^chr1\t" genes.gtf

# Filter VCF for SNPs only
grep -v "^#" variants.vcf | grep "SNP"

# Case-insensitive pattern in sequences
grep -i "ATGC" sequences.txt
```

---

## Pattern Matching

### Regular Expressions Quick Reference

```bash
.       # Any single character
^       # Start of line
$       # End of line
*       # Zero or more occurrences
+       # One or more occurrences (extended regex)
?       # Zero or one occurrence (extended regex)
[abc]   # Any character in set
[^abc]  # Any character not in set
[a-z]   # Character range
|       # OR (extended regex)
()      # Grouping (extended regex)
\       # Escape special character
\t      # Tab
\n      # Newline
{n,m}   # Between n and m occurrences
```

**Examples:**
```bash
# Match email addresses
grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" file.txt

# Match gene IDs
grep -E "^ENSG[0-9]{11}" genes.txt

# Match DNA sequences
grep -E "^[ATGC]+$" sequences.txt
```

---

## File Operations

### cut

Extract columns from files.

```bash
# Extract specific columns (tab-delimited)
cut -f 1,3 file.txt

# Extract column range
cut -f 2-5 file.txt

# Custom delimiter
cut -d ',' -f 1,2 file.csv

# Extract characters by position
cut -c 1-10 file.txt

# All columns except specified
cut -f 2- file.txt  # From column 2 onwards
```

**Bioinformatics Examples:**
```bash
# Extract chromosome and position from VCF
grep -v "^#" variants.vcf | cut -f 1,2

# Get gene names from GTF
cut -f 9 genes.gtf | cut -d ';' -f 1

# Extract first field from FASTA headers
grep "^>" sequences.fasta | cut -d ' ' -f 1
```

---

### sort

Sort lines of text.

```bash
# Alphabetical sort
sort file.txt

# Numeric sort
sort -n numbers.txt

# Reverse sort
sort -r file.txt

# Sort by specific column (tab-delimited)
sort -k 2 file.txt

# Numeric sort by column
sort -k 3n file.txt

# Sort by multiple columns
sort -k 1,1 -k 2n file.txt

# Unique lines only
sort -u file.txt

# Case-insensitive sort
sort -f file.txt

# Sort by delimiter
sort -t ',' -k 2 file.csv
```

**Bioinformatics Examples:**
```bash
# Sort BED file
sort -k1,1 -k2,2n regions.bed

# Sort SAM by position
sort -k3,3 -k4,4n alignments.sam

# Sort genes by expression level
sort -k7,7nr expression.txt
```

---

### uniq

Report or filter repeated lines (input must be sorted).

```bash
# Remove duplicate lines
sort file.txt | uniq

# Count occurrences
sort file.txt | uniq -c

# Show only duplicates
sort file.txt | uniq -d

# Show only unique lines
sort file.txt | uniq -u

# Ignore case
sort file.txt | uniq -i

# Compare specific fields
sort file.txt | uniq -f 1  # Skip first field
```

**Bioinformatics Examples:**
```bash
# Count unique sequences
sort sequences.txt | uniq -c

# Find duplicate read IDs
cut -f 1 reads.txt | sort | uniq -d

# Count gene occurrences
cut -f 1 gene_list.txt | sort | uniq -c | sort -rn
```

---

## Counting and Statistics

### wc

Word, line, character, and byte count.

```bash
# Count lines
wc -l file.txt

# Count words
wc -w file.txt

# Count characters
wc -m file.txt

# Count bytes
wc -c file.txt

# All counts
wc file.txt  # lines, words, bytes

# Multiple files
wc -l *.txt
```

**Bioinformatics Examples:**
```bash
# Count sequences in FASTA
grep -c "^>" sequences.fasta
# OR
grep "^>" sequences.fasta | wc -l

# Count reads in FASTQ
echo $(cat reads.fastq | wc -l)/4 | bc

# Count genes in GFF
grep -v "^#" genes.gff | wc -l

# Total sequence length
grep -v "^>" sequences.fasta | wc -m
```

---

### tr

Translate or delete characters.

```bash
# Translate characters
tr 'a-z' 'A-Z' < file.txt  # Lowercase to uppercase

# Delete characters
tr -d '\n' < file.txt  # Remove newlines

# Squeeze repeated characters
tr -s ' ' < file.txt  # Multiple spaces to single

# Complement (keep only specified)
tr -cd 'ATGC\n' < sequences.txt

# Replace tabs with spaces
tr '\t' ' ' < file.txt
```

**Bioinformatics Examples:**
```bash
# Convert sequence to uppercase
tr 'atgc' 'ATGC' < sequences.fasta

# Remove line breaks from sequence
grep -v "^>" sequences.fasta | tr -d '\n'

# Count nucleotide composition
grep -v "^>" sequences.fasta | tr -cd 'ATGCatgc' | wc -c

# Replace ambiguous bases
tr 'N' 'A' < genome.fasta
```

---

## Compression

### Working with Compressed Files

```bash
# View compressed file
zcat file.gz              # gzip
bzcat file.bz2            # bzip2
xzcat file.xz             # xz

# Search in compressed files
zgrep "pattern" file.gz
zgrep -c "^>" sequences.fasta.gz

# Compare compressed files
zdiff file1.gz file2.gz

# Compress files
gzip file.txt             # Creates file.txt.gz
bzip2 file.txt            # Creates file.txt.bz2
xz file.txt               # Creates file.txt.xz

# Decompress files
gunzip file.gz
bunzip2 file.bz2
unxz file.xz

# Keep original when compressing
gzip -k file.txt

# Parallel compression (pigz)
pigz file.txt             # Faster multi-threaded gzip
```

**Bioinformatics Examples:**
```bash
# Count sequences in compressed FASTA
zgrep -c "^>" sequences.fasta.gz

# Extract from compressed file
zcat reads.fastq.gz | head -n 4

# Pipe compressed file to analysis
zcat genome.fasta.gz | awk '{if(NR%2==0) print length}'

# Search compressed VCF
zgrep "chr1" variants.vcf.gz
```

---

## Advanced Tips

### Piping Commands

```bash
# Chain multiple commands
cat file.txt | grep "pattern" | cut -f 1 | sort | uniq -c

# Process and save
zcat sequences.fasta.gz | grep -A 1 ">gene" | gzip > filtered.fasta.gz
```

### One-liners for Bioinformatics

```bash
# Count sequences per FASTA file
for f in *.fasta; do echo "$f: $(grep -c '^>' $f)"; done

# Get sequence lengths from FASTA
awk '/^>/ {if (seq) print length(seq); seq=""; next} {seq=seq$0} END {print length(seq)}' sequences.fasta

# Convert multi-line FASTA to single-line
awk '/^>/ {if (seq) print seq; print; seq=""; next} {seq=seq$0} END {print seq}' sequences.fasta

# Extract specific columns from VCF
grep -v "^##" variants.vcf | cut -f 1,2,4,5,10

# Reverse complement DNA sequence
echo "ATGC" | tr 'ATGCatgc' 'TACGtacg' | rev

# Calculate N50 from sequence lengths
awk '{len[NR]=$1; sum+=$1} END {asort(len); target=sum/2; for(i=NR;i>=1;i--){cumsum+=len[i]; if(cumsum>=target){print len[i]; break}}}' lengths.txt

# Count kmers (4-mers example)
grep -v "^>" sequences.fasta | grep -o "...."|sort|uniq -c|sort -rn

# Filter FASTQ by quality score average
awk 'NR%4==1{h=$0} NR%4==2{s=$0} NR%4==0{q=$0; sum=0; for(i=1;i<=length(q);i++)sum+=(ord(substr(q,i,1))-33); if(sum/length(q)>=30)print h"\n"s"\n+\n"q}' reads.fastq
```

### Performance Tips

```bash
# Use parallel processing for large files
cat large_file.txt | parallel --pipe grep "pattern"

# Use mawk for faster AWK processing
mawk '{print $1}' huge_file.txt

# Avoid unnecessary cat
grep "pattern" file.txt          # Good
cat file.txt | grep "pattern"    # Unnecessary cat

# Process compressed files directly
zgrep "pattern" file.gz          # Better than: zcat file.gz | grep "pattern"
```

---

## Quick Reference Summary

| Command | Primary Use | Example |
|---------|-------------|---------|
| `awk` | Column processing, calculations | `awk '{print $1, $3}' file.txt` |
| `sed` | Find and replace, line deletion | `sed 's/old/new/g' file.txt` |
| `grep` | Pattern searching | `grep "pattern" file.txt` |
| `cut` | Column extraction | `cut -f 1,3 file.txt` |
| `sort` | Sorting lines | `sort -k2n file.txt` |
| `uniq` | Remove duplicates | `sort file.txt \| uniq -c` |
| `wc` | Counting | `wc -l file.txt` |
| `tr` | Character translation | `tr 'a-z' 'A-Z' < file.txt` |

---

## Additional Resources

- AWK: `man awk` or `man gawk`
- Sed: `man sed`
- Grep: `man grep`
- Online regex tester: regex101.com
- GNU AWK manual: gnu.org/software/gawk/manual/

---

**Note:** Most examples assume tab-delimited files. Adjust field separators (`-F` for awk, `-d` for cut) based on your file format.
