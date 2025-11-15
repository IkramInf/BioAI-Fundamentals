# Genome Assembly Tutorial with Dummy FASTQ Data

## Part 1: Creating Realistic Dummy FASTQ Reads

Let's create a small reference genome and generate realistic paired-end reads from it, then walk through the entire assembly process step-by-step.

### 1.1 Understanding FASTQ Format First

Before we create data, let's understand what we're making:

```
@SEQ_ID                          ← Line 1: Sequence identifier (starts with @)
GATTTGGGGTTCAAAGCAGTATCGATCAA   ← Line 2: DNA sequence
+                                ← Line 3: Separator (starts with +)
!''*((((***+))%%%++)(%%%%).1   ← Line 4: Quality scores (Phred+33 encoding)
```

**Quality Score Encoding (Phred+33):**
```
Symbol  ASCII   Phred   Error Rate      Accuracy
!       33      0       100%            0%
"       34      1       79.4%           20.6%
#       35      2       63.1%           36.9%
...
*       42      9       12.6%           87.4%
+       43      10      10%             90%
5       53      20      1%              99%
:       58      25      0.32%           99.68%
?       63      30      0.1%            99.9%
I       73      40      0.01%           99.99%
```

**Key concept:** Q30 means 1 error per 1000 bases (99.9% accuracy)

### 1.2 Creating Reference Genome

```bash
# Create working directory
mkdir -p fastq_tutorial/{reference,reads,qc,trimmed,assembly,results}
cd fastq_tutorial
```

Let's create a small reference genome (10kb) to make examples tractable:

```python
#!/usr/bin/env python3
"""
Generate a realistic reference genome with various features
"""
import random
random.seed(42)

def generate_reference_genome(length=10000):
    """
    Create a reference genome with realistic features:
    - Random sequence
    - Some repetitive regions
    - GC content ~50%
    """
    genome = []
    
    # Region 1: Normal sequence (0-3000 bp)
    for _ in range(3000):
        genome.append(random.choice(['A', 'T', 'G', 'C']))
    
    # Region 2: AT-rich region (3000-4000 bp) - harder to assemble
    for _ in range(1000):
        genome.append(random.choice(['A', 'T']))
    
    # Region 3: Repetitive region (4000-4500 bp) - causes assembly issues
    repeat_unit = "ATCGATCGATCG"
    for _ in range(500 // len(repeat_unit)):
        genome.extend(list(repeat_unit))
    
    # Region 4: GC-rich region (4500-5500 bp)
    for _ in range(1000):
        genome.append(random.choice(['G', 'C']))
    
    # Region 5: Another repetitive region (5500-6000 bp) - tests repeat resolution
    repeat_unit2 = "GCTAGCTAGCTA"
    for _ in range(500 // len(repeat_unit2)):
        genome.extend(list(repeat_unit2))
    
    # Region 6: Normal sequence (6000-10000 bp)
    for _ in range(4000):
        genome.append(random.choice(['A', 'T', 'G', 'C']))
    
    return ''.join(genome)

# Generate reference
reference = generate_reference_genome(10000)

# Save reference
with open('reference/reference.fasta', 'w') as f:
    f.write('>reference_genome length=10000 description=Tutorial_reference\n')
    # Write in 80 bp lines (standard FASTA format)
    for i in range(0, len(reference), 80):
        f.write(reference[i:i+80] + '\n')

print(f"Reference genome created: {len(reference)} bp")
print(f"GC content: {(reference.count('G') + reference.count('C')) / len(reference) * 100:.1f}%")
```

Run it:
```bash
python3 << 'EOF'
# [paste the above code]
EOF
```

### 1.3 Understanding Paired-End Sequencing

**Key Concepts:**

```
DNA Fragment (Insert):
|<-------------- 500 bp insert size -------------->|
         |<-- 150bp -->|               |<-- 150bp -->|
5'-------===============---------------===============-------3'
         Read 1 (R1) →                ← Read 2 (R2)
         Forward                      Reverse complement

Key terms:
- Insert size: Total fragment length (~500 bp typical for Illumina)
- Read length: Sequence length from each end (100-150 bp typical)
- Inner distance: Insert size - (2 × read length) = 500 - 300 = 200 bp
- Orientation: FR (forward-reverse) is standard for Illumina
```

**Why paired-end?**
1. **Scaffolding**: Know two reads are ~500bp apart
2. **Repeat resolution**: Can span across short repeats
3. **Structural variants**: Detect if insert size is abnormal
4. **Increased confidence**: Two reads confirming same region

### 1.4 Generating Realistic Paired-End Reads

```python
#!/usr/bin/env python3
"""
Generate realistic paired-end FASTQ reads with various quality issues
"""
import random
import gzip
random.seed(42)

def reverse_complement(seq):
    """Get reverse complement of DNA sequence"""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def generate_quality_string(length, base_quality=35, degradation=0):
    """
    Generate quality scores that degrade toward 3' end (realistic)
    
    Args:
        length: read length
        base_quality: starting quality (Phred score)
        degradation: how much quality drops per base
    """
    qualities = []
    for i in range(length):
        # Quality degrades along read
        q = max(2, base_quality - (i * degradation))
        # Add some random variation
        q = max(2, min(40, int(q + random.gauss(0, 3))))
        # Convert to ASCII (Phred+33)
        qualities.append(chr(q + 33))
    return ''.join(qualities)

def introduce_errors(seq, error_rate=0.01):
    """
    Introduce sequencing errors (substitutions, insertions, deletions)
    
    Args:
        seq: DNA sequence
        error_rate: probability of error per base
    """
    seq_list = list(seq)
    bases = ['A', 'T', 'G', 'C']
    
    for i in range(len(seq_list)):
        if random.random() < error_rate:
            error_type = random.choice(['substitution', 'insertion', 'deletion'])
            
            if error_type == 'substitution':
                # Replace with different base
                original = seq_list[i]
                seq_list[i] = random.choice([b for b in bases if b != original])
            
            elif error_type == 'insertion' and i < len(seq_list) - 1:
                # Insert random base
                seq_list[i] = seq_list[i] + random.choice(bases)
            
            elif error_type == 'deletion':
                # Delete base
                seq_list[i] = ''
    
    return ''.join(seq_list)

def add_adapters(seq, adapter_rate=0.1):
    """
    Add Illumina adapter sequences (contamination)
    Happens when insert < 2×read_length
    """
    if random.random() < adapter_rate:
        # Illumina TruSeq adapter
        adapter = "AGATCGGAAGAGC"
        # Add partial adapter to end
        adapter_length = random.randint(5, len(adapter))
        return seq[:-adapter_length] + adapter[:adapter_length]
    return seq

def generate_paired_end_reads(reference, num_reads=500, read_length=100, 
                               insert_size=300, insert_std=50,
                               coverage=None):
    """
    Generate paired-end reads from reference
    
    Args:
        reference: reference genome sequence
        num_reads: number of read pairs to generate
        read_length: length of each read
        insert_size: mean insert size
        insert_std: standard deviation of insert size
        coverage: if specified, calculate num_reads to achieve this coverage
    """
    
    if coverage:
        # Calculate number of reads needed for desired coverage
        # Coverage = (num_reads × read_length × 2) / genome_length
        genome_length = len(reference)
        num_reads = int((coverage * genome_length) / (read_length * 2))
        print(f"Generating {num_reads} read pairs for {coverage}× coverage")
    
    reads_r1 = []
    reads_r2 = []
    
    for read_id in range(num_reads):
        # Random insert size (normal distribution)
        current_insert = int(random.gauss(insert_size, insert_std))
        current_insert = max(read_length * 2, current_insert)  # Minimum size
        
        # Random starting position
        max_start = len(reference) - current_insert
        if max_start < 0:
            continue
        start_pos = random.randint(0, max_start)
        
        # Extract fragment
        fragment = reference[start_pos:start_pos + current_insert]
        
        # R1: forward strand, from start
        r1_seq = fragment[:read_length]
        
        # R2: reverse strand, from end
        r2_seq = reverse_complement(fragment[-read_length:])
        
        # Introduce sequencing errors (higher quality = lower error rate)
        r1_seq = introduce_errors(r1_seq, error_rate=0.005)  # 0.5% error
        r2_seq = introduce_errors(r2_seq, error_rate=0.01)   # 1% error (R2 often worse)
        
        # Some reads have adapter contamination
        r1_seq = add_adapters(r1_seq, adapter_rate=0.05)  # 5% have adapters
        r2_seq = add_adapters(r2_seq, adapter_rate=0.05)
        
        # Generate quality scores (R2 typically lower quality)
        r1_qual = generate_quality_string(len(r1_seq), base_quality=37, degradation=0.1)
        r2_qual = generate_quality_string(len(r2_seq), base_quality=33, degradation=0.15)
        
        # Create FASTQ entries
        # Modern Illumina format: @instrument:run:flowcell:lane:tile:x:y read:filtered:control:index
        header = f"@SEQ_ID_{read_id}:1:1101:1000:{read_id}:0 1:N:0:ATCG"
        
        reads_r1.append(f"{header}\n{r1_seq}\n+\n{r1_qual}\n")
        
        # R2 header has 2:N instead of 1:N
        header_r2 = header.replace(" 1:N:", " 2:N:")
        reads_r2.append(f"{header_r2}\n{r2_seq}\n+\n{r2_qual}\n")
    
    return reads_r1, reads_r2

# Load reference
with open('reference/reference.fasta', 'r') as f:
    lines = f.readlines()
    reference = ''.join(line.strip() for line in lines if not line.startswith('>'))

# Generate reads for 30× coverage
print("Generating paired-end reads (30× coverage)...")
reads_r1, reads_r2 = generate_paired_end_reads(
    reference,
    read_length=100,
    insert_size=300,
    coverage=30
)

print(f"Generated {len(reads_r1)} paired-end reads")

# Write FASTQ files (gzipped, like real data)
with gzip.open('reads/sample_R1.fastq.gz', 'wt') as f:
    f.writelines(reads_r1)

with gzip.open('reads/sample_R2.fastq.gz', 'wt') as f:
    f.writelines(reads_r2)

print("FASTQ files created:")
print("  - reads/sample_R1.fastq.gz")
print("  - reads/sample_R2.fastq.gz")

# Also create low-quality version for demonstration
print("\nGenerating low-quality reads (showing quality issues)...")
low_qual_r1, low_qual_r2 = [], []

for i in range(100):
    start_pos = random.randint(0, len(reference) - 300)
    fragment = reference[start_pos:start_pos + 300]
    
    r1_seq = fragment[:100]
    r2_seq = reverse_complement(fragment[-100:])
    
    # Introduce MORE errors
    r1_seq = introduce_errors(r1_seq, error_rate=0.03)
    r2_seq = introduce_errors(r2_seq, error_rate=0.05)
    
    # MORE adapters
    r1_seq = add_adapters(r1_seq, adapter_rate=0.3)
    r2_seq = add_adapters(r2_seq, adapter_rate=0.3)
    
    # LOWER quality scores
    r1_qual = generate_quality_string(len(r1_seq), base_quality=25, degradation=0.3)
    r2_qual = generate_quality_string(len(r2_seq), base_quality=20, degradation=0.4)
    
    header = f"@LOWQUAL_{i}:1:1101:1000:{i}:0 1:N:0:ATCG"
    low_qual_r1.append(f"{header}\n{r1_seq}\n+\n{r1_qual}\n")
    
    header_r2 = header.replace(" 1:N:", " 2:N:")
    low_qual_r2.append(f"{header_r2}\n{r2_seq}\n+\n{r2_qual}\n")

with gzip.open('reads/lowqual_R1.fastq.gz', 'wt') as f:
    f.writelines(low_qual_r1)

with gzip.open('reads/lowqual_R2.fastq.gz', 'wt') as f:
    f.writelines(low_qual_r2)

print("Low-quality FASTQ files created for comparison")
EOF
```

### 1.5 Inspect the Generated FASTQ Files

```bash
# Look at first read pair
echo "=== First Read Pair (R1) ==="
zcat reads/sample_R1.fastq.gz | head -4

echo -e "\n=== First Read Pair (R2) ==="
zcat reads/sample_R2.fastq.gz | head -4

# Count reads
echo -e "\n=== Read Count ==="
echo "R1 reads: $(zcat reads/sample_R1.fastq.gz | wc -l | awk '{print $1/4}')"
echo "R2 reads: $(zcat reads/sample_R2.fastq.gz | wc -l | awk '{print $1/4}')"

# Calculate theoretical coverage
GENOME_SIZE=10000
READ_LENGTH=100
NUM_READS=$(zcat reads/sample_R1.fastq.gz | wc -l | awk '{print $1/4}')
COVERAGE=$(echo "scale=1; $NUM_READS * $READ_LENGTH * 2 / $GENOME_SIZE" | bc)
echo "Coverage: ${COVERAGE}×"
```

**Example output you'll see:**
```
=== First Read Pair (R1) ===
@SEQ_ID_0:1:1101:1000:0:0 1:N:0:ATCG
CGATTACGATCGATTACGATCGATTAGCATCGATCGATTACGATCGATTACGATCGATTAGCATCGATCGATTACGATCGATTACGATCGATTAGCAT
+
IIIIIHHHGGGGFFFFFFEEEEEDDDDDCCCCCBBBBB@@@@@>>>>>====<<<<;;;;;:::::99998888877777666665555544444333

=== First Read Pair (R2) ===
@SEQ_ID_0:1:1101:1000:0:0 2:N:0:ATCG
ATGCTAATCGATCGATAGTCGATCGATATGCTAATCGATCGATAGTCGATCGATATGCTAATCGATCGATAGTCGATCGATATGCTAATCGATCGAT
+
HHHHGGGGFFFFEEEEDDDCCCCBBBB@@@@>>>>====<<<;;;;::::9999888877776666555544443333222211110000////....
```

**Understanding this output:**

1. **Header line (@SEQ_ID_0:1:1101:1000:0:0 1:N:0:ATCG)**
   - `@` = FASTQ identifier marker
   - `SEQ_ID_0` = unique sequence ID
   - `1:1101:1000:0:0` = instrument:flowcell:tile:x:y coordinates
   - `1:N:0` = read number (1 or 2), filtered flag, control bits
   - `ATCG` = sample barcode/index

2. **Sequence line**
   - 100 bp DNA sequence
   - Only A, T, G, C (or N for unknown)

3. **Separator line (+)**
   - Can optionally repeat header
   - Usually just `+`

4. **Quality line**
   - One character per base in sequence
   - ASCII characters encoding Phred scores
   - `I` = Q40 (99.99% accurate)
   - `5` = Q20 (99% accurate)
   - `!` = Q0 (completely unreliable)

---

## Part 2: Step-by-Step Analysis with Algorithms Explained

### 2.1 Quality Control with FastQC

**What FastQC does:**
1. Reads FASTQ files sequentially
2. For each read, calculates statistics
3. Aggregates across all reads
4. Generates visual reports

```bash
# Run FastQC on our dummy data
fastqc -o qc/ reads/sample_R1.fastq.gz reads/sample_R2.fastq.gz

# Also run on low-quality data for comparison
fastqc -o qc/ reads/lowqual_R1.fastq.gz reads/lowqual_R2.fastq.gz
```

**FastQC Algorithm Breakdown:**

```python
# Simplified version of what FastQC does internally
def fastqc_analysis(fastq_file):
    """
    Simplified FastQC algorithm
    """
    
    # 1. Per-base quality scores
    position_qualities = [[] for _ in range(150)]  # Max read length
    
    # 2. Per-sequence quality scores
    sequence_qualities = []
    
    # 3. Sequence content per position
    position_content = {
        'A': [0] * 150,
        'T': [0] * 150,
        'G': [0] * 150,
        'C': [0] * 150,
        'N': [0] * 150
    }
    
    # 4. GC content distribution
    gc_contents = []
    
    # 5. Sequence length distribution
    length_counts = {}
    
    # 6. Adapter content
    adapter_seq = "AGATCGGAAGAGC"
    adapter_matches = [0] * 150
    
    # Read FASTQ file
    read_count = 0
    for record in parse_fastq(fastq_file):
        sequence = record['sequence']
        quality = record['quality']
        
        read_count += 1
        seq_len = len(sequence)
        
        # Length distribution
        length_counts[seq_len] = length_counts.get(seq_len, 0) + 1
        
        # Per-base statistics
        for pos, (base, qual) in enumerate(zip(sequence, quality)):
            if pos < 150:
                # Quality score (convert ASCII to Phred)
                phred_score = ord(qual) - 33
                position_qualities[pos].append(phred_score)
                
                # Base content
                position_content[base][pos] += 1
        
        # Per-sequence quality (mean)
        mean_quality = sum(ord(q) - 33 for q in quality) / len(quality)
        sequence_qualities.append(mean_quality)
        
        # GC content
        gc_count = sequence.count('G') + sequence.count('C')
        gc_percent = (gc_count / seq_len) * 100
        gc_contents.append(gc_percent)
        
        # Adapter detection (simple k-mer matching)
        for i in range(len(sequence) - len(adapter_seq) + 1):
            if sequence[i:i+len(adapter_seq)] == adapter_seq:
                for j in range(i, min(i + len(adapter_seq), 150)):
                    adapter_matches[j] += 1
    
    # Calculate statistics
    results = {
        'total_sequences': read_count,
        'mean_quality_per_position': [
            sum(q) / len(q) if q else 0 
            for q in position_qualities
        ],
        'mean_sequence_quality': sum(sequence_qualities) / len(sequence_qualities),
        'gc_content_mean': sum(gc_contents) / len(gc_contents),
        'adapter_percentage': [
            (count / read_count) * 100 
            for count in adapter_matches
        ]
    }
    
    return results
```

**Key FastQC Modules Explained:**

**1. Per Base Sequence Quality**
```
Position    Mean_Q    Median_Q    Lower_Q    Upper_Q
1           37.2      38          35         40
2           37.1      38          34         40
3           36.9      37          34         40
...
98          28.3      29          22         35
99          27.1      28          20         34
100         25.8      26          18         33
```
- Shows quality degradation along read
- **Warning if:** any position < Q20
- **Failure if:** any position < Q10

**2. Adapter Content**
```
Position    %Adapter
1-50        0.0%
51-70       0.5%
71-85       2.3%
86-95       5.8%
96-100      8.2%
```
- Increases toward 3' end (expected for short inserts)
- **Warning if:** >5% at any position
- **Failure if:** >10% at any position

**Let's see what our QC reports show:**
```bash
# View in browser or cat the summary
firefox qc/sample_R1_fastqc.html &

# Or check text summary
unzip -p qc/sample_R1_fastqc.zip sample_R1_fastqc/fastqc_data.txt | head -50
```

### 2.2 Read Trimming with fastp

**fastp Algorithm:**

```
For each read pair (R1, R2):
    
    1. ADAPTER DETECTION
       - Slide adapter sequences along read
       - Find best match using Smith-Waterman alignment
       - If match score > threshold, mark adapter position
    
    2. QUALITY TRIMMING (sliding window)
       - Window size = 4 bases (default)
       - For each window from 3' end:
           mean_quality = sum(quality_scores) / window_size
           if mean_quality < threshold:
               trim from this position
               break
    
    3. LENGTH FILTERING
       - After trimming, check read length
       - If length < minimum (e.g., 50 bp):
           discard read pair
    
    4. QUALITY FILTERING
       - Count bases with Q < threshold
       - If (low_quality_bases / total_bases) > max_percentage:
           discard read pair
    
    5. PAIRED-END CORRECTION
       - Check if R1 and R2 overlap (for short inserts)
       - If overlap found:
           * Correct mismatches using quality scores
           * Merge into single longer read (optional)
    
    Output: Trimmed, filtered read pairs
```

**Run fastp on our data:**

```bash
# Good quality data
fastp \
    -i reads/sample_R1.fastq.gz \
    -I reads/sample_R2.fastq.gz \
    -o trimmed/sample_R1.trim.fastq.gz \
    -O trimmed/sample_R2.trim.fastq.gz \
    --detect_adapter_for_pe \
    --qualified_quality_phred 20 \
    --unqualified_percent_limit 40 \
    --length_required 50 \
    --thread 4 \
    --html qc/fastp_good.html \
    --json qc/fastp_good.json

# Low quality data (for comparison)
fastp \
    -i reads/lowqual_R1.fastq.gz \
    -I reads/lowqual_R2.fastq.gz \
    -o trimmed/lowqual_R1.trim.fastq.gz \
    -O trimmed/lowqual_R2.trim.fastq.gz \
    --detect_adapter_for_pe \
    --qualified_quality_phred 20 \
    --unqualified_percent_limit 40 \
    --length_required 50 \
    --thread 4 \
    --html qc/fastp_lowqual.html \
    --json qc/fastp_lowqual.json
```

**Understanding fastp output:**

```bash
# Check fastp JSON report
python3 << 'EOF'
import json

with open('qc/fastp_good.json') as f:
    data = json.load(f)

print("=== FASTP SUMMARY (Good Quality Data) ===\n")

print("BEFORE filtering:")
print(f"  Total reads: {data['summary']['before_filtering']['total_reads']:,}")
print(f"  Total bases: {data['summary']['before_filtering']['total_bases']:,}")
print(f"  Q20 bases: {data['summary']['before_filtering']['q20_rate']*100:.1f}%")
print(f"  Q30 bases: {data['summary']['before_filtering']['q30_rate']*100:.1f}%")
print(f"  GC content: {data['summary']['before_filtering']['gc_content']*100:.1f}%")

print("\nAFTER filtering:")
print(f"  Total reads: {data['summary']['after_filtering']['total_reads']:,}")
print(f"  Total bases: {data['summary']['after_filtering']['total_bases']:,}")
print(f"  Q20 bases: {data['summary']['after_filtering']['q20_rate']*100:.1f}%")
print(f"  Q30 bases: {data['summary']['after_filtering']['q30_rate']*100:.1f}%")

print("\nFILTERING RESULT:")
passed_reads = data['summary']['after_filtering']['total_reads']
total_reads = data['summary']['before_filtering']['total_reads']
retention = (passed_reads / total_reads) * 100
print(f"  Reads passed: {retention:.1f}%")
print(f"  Reads filtered: {100-retention:.1f}%")

if 'adapter_cutting' in data:
    print("\nADAPTER TRIMMING:")
    print(f"  R1 with adapters: {data['adapter_cutting']['read1_adapter_counts']}")
    print(f"  R2 with adapters: {data['adapter_cutting']['read2_adapter_counts']}")
EOF
```

**Compare before and after trimming:**

```bash
echo "=== BEFORE TRIMMING ===" 
zcat reads/sample_R1.fastq.gz | head -8

echo -e "\n=== AFTER TRIMMING ==="
zcat trimmed/sample_R1.trim.fastq.gz | head -8
```

### 2.3 Understanding De Bruijn Graph Assembly

This is the core algorithm SPAdes uses. Let's explain with a simple example.

**De Bruijn Graph Theory:**

```
Given reads:
ATGGCGT
TGGCGTA
GGCGTAC
GCGTACG

Step 1: Break into k-mers (k=4)
Read 1: ATGG, TGGC, GGCG, GCGT
Read 2: TGGC, GGCG, GCGT, CGTA
Read 3: GGCG, GCGT, CGTA, GTAC
Read 4: GCGT, CGTA, GTAC, TACG

Step 2: Create (k-1)-mers (3-mers) as nodes
ATG, TGG, GGC, GCG, CGT, GTA, TAC, ACG

Step 3: Create edges (k-mers)
ATG --ATGG--> TGG
TGG --TGGC--> GGC
GGC --GGCG--> GCG
GCG --GCGT--> CGT
CGT --CGTA--> GTA
GTA --GTAC--> TAC
TAC --TACG--> ACG

Step 4: Find Eulerian path
ATG → TGG → GGC → GCG → CGT → GTA → TAC → ACG

Step 5: Reconstruct sequence
ATGGCGTACG (original sequence recovered!)
```

**Visual representation:**

```
         ATGG      TGGC      GGCG      GCGT      CGTA      GTAC      TACG
    ATG -----> TGG -----> GGC -----> GCG -----> CGT -----> GTA -----> TAC -----> ACG
    
This is a simple path - assembly is easy!
```

**What makes assembly hard? REPEATS:**

```
Reads with repeats:
ATGCGCGCGTA  (CG repeat 3 times)
TGCGCGCGTAT
GCGCGCGTATT

K-mer graph (k=3):
         GC       GC       GC
    TG -----> CG -----> CG -----> CG -----> GT
                ↑         ↑         ↑
                └─────────┴─────────┘
              (3 different copies of CG, but we can't distinguish!)

Result: Graph has CYCLES
- Can't determine correct path
- Assembly breaks into fragments
```

**SPAdes solution: Multiple k-mer sizes**
```
Small k (k=21):
- More overlaps found
- Graph more connected
- But: more ambiguity

Large k (k=77):
- Fewer overlaps
- Less ambiguity
- But: gaps in low-coverage regions

SPAdes strategy:
1. Start with small k (capture all data)
2. Iteratively increase k (resolve ambiguity)
3. Use paired-end info to resolve remaining issues
```

### 2.4 Running Assembly on Our Data

```bash
# Assemble with SPAdes
spades.py \
    -1 trimmed/sample_R1.trim.fastq.gz \
    -2 trimmed/sample_R2.trim.fastq.gz \
    -o assembly/spades \
    --threads 4 \
    --memory 8 \
    -k 21,33,55,77 \
    --careful

echo "Assembly complete!"

# Continue from SPAdes Assembly

## 2.5 Understanding SPAdes Step-by-Step

Let's trace what SPAdes does internally with our data:

```bash
# Watch SPAdes log to see what's happening
cat assembly/spades/spades.log
```

**SPAdes Pipeline Stages:**

```
Stage 1: READ ERROR CORRECTION (BayesHammer)
│
├─> Input: Raw reads with ~1% errors
├─> Process: 
│   ├─ Count all k-mers (k=21)
│   ├─ Build k-mer frequency histogram
│   ├─ Identify trusted k-mers (high frequency)
│   ├─ Identify erroneous k-mers (low frequency)
│   └─ Correct reads using trusted k-mers
└─> Output: Error-corrected reads

Stage 2: ASSEMBLY (iterative k-mer approach)
│
├─> For k in [21, 33, 55, 77]:
│   │
│   ├─ K-MER COUNTING
│   │  └─> Count all k-mers and their frequencies
│   │
│   ├─ DE BRUIJN GRAPH CONSTRUCTION
│   │  ├─> Create nodes from (k-1)-mers
│   │  ├─> Create edges from k-mers
│   │  └─> Weight edges by k-mer coverage
│   │
│   ├─ GRAPH SIMPLIFICATION
│   │  ├─> Remove tips (dead ends)
│   │  ├─> Collapse bubbles (sequencing errors)
│   │  ├─> Remove low-coverage edges
│   │  └─> Merge unambiguous paths
│   │
│   ├─> Use result from k-1 to seed k
│   │
│   └─> CONTIG EXTRACTION
│       └─> Find non-branching paths = contigs
│
└─> Final contigs from largest k

Stage 3: SCAFFOLDING
│
├─> Use paired-end insert size info
├─> Connect contigs that are ~300bp apart
└─> Create scaffolds with N-gaps

Stage 4: GAP CLOSING (if --careful)
│
├─> Try to fill N-gaps using reads
└─> Polish with MismatchCorrector

Stage 5: REPEAT RESOLUTION
│
├─> Use paired-end constraints
├─> Use coverage information
└─> Attempt to resolve ambiguous paths
```

### 2.6 Detailed Algorithm Walkthrough

Let's manually trace the assembly algorithm with a subset of our data:

```python
#!/usr/bin/env python3
"""
Manual demonstration of De Bruijn graph assembly
Using actual reads from our dataset
"""
import gzip
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt

def parse_fastq(filename, max_reads=20):
    """Parse FASTQ and return sequences"""
    sequences = []
    with gzip.open(filename, 'rt') as f:
        while len(sequences) < max_reads:
            try:
                header = next(f).strip()
                seq = next(f).strip()
                plus = next(f).strip()
                qual = next(f).strip()
                sequences.append(seq)
            except StopIteration:
                break
    return sequences

def generate_kmers(sequence, k):
    """Generate all k-mers from a sequence"""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers

def build_debruijn_graph(reads, k):
    """
    Build De Bruijn graph from reads
    
    Returns:
        nodes: (k-1)-mers
        edges: k-mers connecting nodes
        edge_coverage: how many times each edge is seen
    """
    # Count all k-mers
    kmer_counts = Counter()
    for read in reads:
        for kmer in generate_kmers(read, k):
            kmer_counts[kmer] += 1
    
    # Build graph
    edges = defaultdict(list)  # node -> list of (next_node, kmer, coverage)
    nodes = set()
    
    for kmer, count in kmer_counts.items():
        # Split k-mer into (k-1)-mer prefix and suffix
        prefix = kmer[:-1]  # First k-1 bases
        suffix = kmer[1:]   # Last k-1 bases
        
        nodes.add(prefix)
        nodes.add(suffix)
        
        # Add edge from prefix to suffix
        edges[prefix].append((suffix, kmer, count))
    
    return nodes, edges, kmer_counts

def simplify_graph(nodes, edges, min_coverage=2):
    """
    Simplify graph by removing low-coverage edges (errors)
    """
    # Remove edges with coverage < threshold
    filtered_edges = defaultdict(list)
    
    for node, next_nodes in edges.items():
        for next_node, kmer, coverage in next_nodes:
            if coverage >= min_coverage:
                filtered_edges[node].append((next_node, kmer, coverage))
    
    return filtered_edges

def remove_tips(edges, max_tip_length=3):
    """
    Remove tips (dead-end paths) from graph
    These are usually sequencing errors at read ends
    """
    # Find nodes with no outgoing edges (tips)
    all_nodes = set(edges.keys())
    for node in edges:
        for next_node, _, _ in edges[node]:
            all_nodes.add(next_node)
    
    # Simple tip removal (in real assembler, more sophisticated)
    tips = []
    for node in all_nodes:
        if node not in edges or len(edges[node]) == 0:
            tips.append(node)
    
    return tips

def extract_contigs(edges, min_length=50):
    """
    Extract contigs by finding non-branching paths
    
    A contig is a maximal non-branching path in the graph
    """
    contigs = []
    visited = set()
    
    # Find all linear paths
    for start_node in edges:
        if start_node in visited:
            continue
        
        # Check if this is a start of a linear path
        # (no incoming edges from other paths, or multiple outgoing)
        
        # Extend path as far as possible
        path = [start_node]
        current = start_node
        
        while current in edges and len(edges[current]) == 1:
            next_node, kmer, coverage = edges[current][0]
            
            # Check if next_node has only one incoming edge (from current)
            incoming_count = sum(
                1 for n, nexts in edges.items() 
                for next_n, _, _ in nexts 
                if next_n == next_node
            )
            
            if incoming_count == 1:
                path.append(next_node)
                current = next_node
            else:
                break
        
        # Convert path to sequence
        if len(path) >= 2:
            # First node gives us k-1 bases
            contig_seq = path[0]
            # Each subsequent node adds one base (last base)
            for node in path[1:]:
                contig_seq += node[-1]
            
            if len(contig_seq) >= min_length:
                contigs.append(contig_seq)
                visited.update(path)
    
    return contigs

def visualize_graph(edges, filename='graph.png', max_nodes=30):
    """Visualize De Bruijn graph"""
    G = nx.DiGraph()
    
    # Limit nodes for visibility
    node_count = 0
    for node, next_nodes in list(edges.items())[:max_nodes]:
        for next_node, kmer, coverage in next_nodes:
            G.add_edge(node, next_node, label=f"{kmer}\n(cov={coverage})")
            node_count += 1
            if node_count >= max_nodes:
                break
        if node_count >= max_nodes:
            break
    
    if len(G.nodes()) == 0:
        print("No nodes to visualize")
        return
    
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='lightblue',
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title("De Bruijn Graph (partial)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {filename}")

# Main assembly demonstration
print("=" * 70)
print("MANUAL ASSEMBLY WALKTHROUGH")
print("=" * 70)

# Load small subset of reads
print("\n1. Loading reads...")
reads_r1 = parse_fastq('trimmed/sample_R1.trim.fastq.gz', max_reads=100)
reads_r2 = parse_fastq('trimmed/sample_R2.trim.fastq.gz', max_reads=100)
all_reads = reads_r1 + reads_r2

print(f"   Loaded {len(all_reads)} reads")
print(f"   Example read: {all_reads[0][:50]}...")

# Try different k-mer sizes
for k in [15, 21, 31]:
    print(f"\n{'='*70}")
    print(f"2. ASSEMBLY WITH k={k}")
    print(f"{'='*70}")
    
    # Generate k-mers
    print(f"\n   Generating {k}-mers...")
    all_kmers = []
    for read in all_reads:
        all_kmers.extend(generate_kmers(read, k))
    
    unique_kmers = len(set(all_kmers))
    print(f"   Total k-mers: {len(all_kmers)}")
    print(f"   Unique k-mers: {unique_kmers}")
    
    # Build graph
    print(f"\n   Building De Bruijn graph...")
    nodes, edges, kmer_counts = build_debruijn_graph(all_reads, k)
    
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges: {sum(len(v) for v in edges.values())}")
    
    # Show k-mer coverage distribution
    coverages = list(kmer_counts.values())
    print(f"\n   K-mer coverage statistics:")
    print(f"     Mean: {sum(coverages)/len(coverages):.1f}")
    print(f"     Min: {min(coverages)}")
    print(f"     Max: {max(coverages)}")
    
    # Coverage histogram
    coverage_hist = Counter(coverages)
    print(f"\n   Coverage histogram:")
    for cov in sorted(coverage_hist.keys())[:10]:
        print(f"     Coverage {cov}: {coverage_hist[cov]} k-mers {'*' * min(coverage_hist[cov]//10, 50)}")
    
    # Simplify graph
    print(f"\n   Simplifying graph (removing low-coverage edges)...")
    edges = simplify_graph(nodes, edges, min_coverage=2)
    print(f"   Edges after filtering: {sum(len(v) for v in edges.values())}")
    
    # Extract contigs
    print(f"\n   Extracting contigs...")
    contigs = extract_contigs(edges, min_length=50)
    
    print(f"   Number of contigs: {len(contigs)}")
    if contigs:
        contig_lengths = [len(c) for c in contigs]
        print(f"   Contig lengths: min={min(contig_lengths)}, max={max(contig_lengths)}, mean={sum(contig_lengths)/len(contig_lengths):.1f}")
        
        # Calculate N50
        sorted_lengths = sorted(contig_lengths, reverse=True)
        total_length = sum(sorted_lengths)
        cumsum = 0
        n50 = 0
        for length in sorted_lengths:
            cumsum += length
            if cumsum >= total_length / 2:
                n50 = length
                break
        print(f"   N50: {n50}")
        
        # Show longest contig
        longest_contig = max(contigs, key=len)
        print(f"\n   Longest contig ({len(longest_contig)} bp):")
        print(f"   {longest_contig[:80]}...")
        
        # Compare to reference
        with open('reference/reference.fasta', 'r') as f:
            ref_lines = f.readlines()
            reference = ''.join(line.strip() for line in ref_lines if not line.startswith('>'))
        
        # Check if longest contig is in reference
        if longest_contig in reference:
            position = reference.find(longest_contig)
            print(f"   ✓ Found exact match in reference at position {position}")
        else:
            # Try reverse complement
            complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
            rev_comp = ''.join(complement.get(base, 'N') for base in reversed(longest_contig))
            if rev_comp in reference:
                position = reference.find(rev_comp)
                print(f"   ✓ Found reverse complement match in reference at position {position}")
            else:
                # Try partial match
                max_match = 0
                for i in range(len(reference) - len(longest_contig) + 1):
                    matches = sum(1 for a, b in zip(longest_contig, reference[i:]) if a == b)
                    max_match = max(max_match, matches)
                identity = (max_match / len(longest_contig)) * 100
                print(f"   Best match: {identity:.1f}% identity")
    
    # Visualize graph for k=21 only
    if k == 21:
        print(f"\n   Generating graph visualization...")
        visualize_graph(edges, f'results/debruijn_graph_k{k}.png', max_nodes=30)

print("\n" + "="*70)
print("ASSEMBLY DEMONSTRATION COMPLETE")
print("="*70)
```

Save and run it:

```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

**Expected output explanation:**

```
=== k=15 (small k-mer) ===
Total k-mers: 18000
Unique k-mers: 8234
Nodes: 8456
Edges: 12389

Coverage histogram:
  Coverage 1: 4567 k-mers  (likely errors)
  Coverage 2: 892 k-mers   (low coverage regions)
  Coverage 3-10: 2145 k-mers (normal coverage)
  Coverage >10: 630 k-mers  (repetitive regions)

Contigs: 234
N50: 187 bp

Analysis:
→ Small k means many k-mers match by chance
→ Graph is highly connected (many edges)
→ Hard to find unique paths
→ Many short, fragmented contigs
→ BUT: captures low-coverage regions well
```

```
=== k=31 (larger k-mer) ===
Total k-mers: 14000
Unique k-mers: 5678
Nodes: 5912
Edges: 7234

Coverage histogram:
  Coverage 1: 1234 k-mers (fewer errors survive)
  Coverage 2-5: 2456 k-mers
  Coverage >5: 1988 k-mers

Contigs: 87
N50: 543 bp

Analysis:
→ Larger k means fewer spurious matches
→ Graph is less connected (fewer edges)
→ Easier to find unique paths
→ Longer contigs
→ BUT: gaps in low-coverage regions
```

### 2.7 Understanding Why Paired-End Helps

Let's visualize the power of paired-end information:

```python
#!/usr/bin/env python3
"""
Demonstrate how paired-end reads resolve assembly ambiguity
"""

def demonstrate_repeat_problem():
    """Show how repeats cause assembly problems"""
    
    print("="*70)
    print("REPEAT RESOLUTION WITH PAIRED-END READS")
    print("="*70)
    
    # Simulated genome with repeat
    genome = (
        "ATCGATCGTAGCTAGC"  # Region A (unique)
        "CGCGCGCGCGCGCGCG"  # Repeat (16 bp)
        "TATATATATATATATA"  # Region B (unique)
        "CGCGCGCGCGCGCGCG"  # Same repeat again!
        "GCTAGCTAGCTAGCTA"  # Region C (unique)
    )
    
    print(f"\nGenome structure:")
    print(f"  [Region A]--[REPEAT]--[Region B]--[REPEAT]--[Region C]")
    print(f"  Length: {len(genome)} bp")
    
    # Single-end reads (can't distinguish repeat copies)
    print(f"\n{'='*70}")
    print("SCENARIO 1: Single-End Reads (100bp)")
    print("="*70)
    
    # Read spanning into repeat from A
    read1 = genome[10:110]  # Starts in A, enters repeat
    print(f"\nRead 1: ...{read1[:20]}...{read1[-20:]}...")
    print(f"        (from region A into repeat)")
    
    # Read in repeat
    read2 = genome[20:120]  # Entirely in/near repeat
    print(f"\nRead 2: ...{read2[:20]}...{read2[-20:]}...")
    print(f"        (could be either repeat copy!)")
    
    # Read spanning out of repeat into B
    read3 = genome[30:130]  # Exits repeat into B
    print(f"\nRead 3: ...{read3[:20]}...{read3[-20:]}...")
    print(f"        (from repeat into region B)")
    
    print(f"\n❌ PROBLEM:")
    print(f"   Can't tell if Read 2 belongs to first or second repeat copy")
    print(f"   Assembly breaks into 3+ contigs")
    
    # Paired-end reads (can distinguish using insert size)
    print(f"\n{'='*70}")
    print("SCENARIO 2: Paired-End Reads (100bp reads, 300bp insert)")
    print("="*70)
    
    # Pair 1: R1 in region A, R2 in region B
    pair1_r1 = genome[10:110]   # Region A → repeat
    pair1_r2_start = 10 + 300 - 100  # 210
    pair1_r2 = genome[pair1_r2_start:pair1_r2_start+100]  # Region B
    
    print(f"\nPair 1:")
    print(f"  R1 (fwd): {pair1_r1[:15]}...{pair1_r1[-15:]}")
    print(f"  R2 (rev): {pair1_r2[:15]}...{pair1_r2[-15:]}")
    print(f"  Insert: 300bp")
    print(f"  ✓ Links region A → first repeat → region B")
    
    # Pair 2: R1 in region B, R2 in region C
    pair2_r1 = genome[40:140]   # Region B → second repeat
    pair2_r2_start = 40 + 300 - 100  # 240
    pair2_r2 = genome[pair2_r2_start:pair2_r2_start+100]  # Region C
    
    print(f"\nPair 2:")
    print(f"  R1 (fwd): {pair2_r1[:15]}...{pair2_r1[-15:]}")
    print(f"  R2 (rev): {pair2_r2[:15]}...{pair2_r2[-15:]}")
    print(f"  Insert: 300bp")
    print(f"  ✓ Links region B → second repeat → region C")
    
    print(f"\n✓ SOLUTION:")
    print(f"   Insert size constraint tells us:")
    print(f"   - First repeat connects A to B")
    print(f"   - Second repeat connects B to C")
    print(f"   - Can reconstruct: A → repeat → B → repeat → C")
    
    print(f"\n{'='*70}")
    print("ASSEMBLY RESULTS COMPARISON")
    print("="*70)
    
    print(f"\nSingle-End Assembly:")
    print(f"  Contig 1: Region A + partial repeat")
    print(f"  Contig 2: Region B (ambiguous)")  
    print(f"  Contig 3: partial repeat + Region C")
    print(f"  N50: ~50bp")
    print(f"  Status: ❌ Fragmented, ambiguous")
    
    print(f"\nPaired-End Assembly:")
    print(f"  Contig 1: Complete genome")
    print(f"  N50: {len(genome)}bp")
    print(f"  Status: ✓ Correctly resolved")

demonstrate_repeat_problem()
```

Run it:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

### 2.8 Assembly Quality Assessment with QUAST

**QUAST Algorithm:**

```
Input: Assembly FASTA, Reference FASTA (optional)

1. BASIC STATISTICS (no reference needed)
   ├─> Count contigs
   ├─> Calculate total length
   ├─> Find largest contig
   ├─> Calculate N50/L50
   ├─> Calculate GC content
   └─> Calculate N-content (gaps)

2. IF REFERENCE PROVIDED:
   │
   ├─> ALIGN CONTIGS TO REFERENCE
   │   └─> Use MUMmer (nucmer) for whole-genome alignment
   │
   ├─> CALCULATE GENOME FRACTION
   │   └─> % of reference covered by contigs
   │
   ├─> DETECT MISASSEMBLIES
   │   ├─> Relocations (contig maps to 2+ distant regions)
   │   ├─> Inversions (contig maps in wrong orientation)
   │   ├─> Translocations (contig spans chromosomes)
   │   └─> Local misassemblies (small rearrangements)
   │
   ├─> COUNT MISMATCHES AND INDELS
   │   └─> Compare aligned bases to reference
   │
   └─> IDENTIFY UNALIGNED CONTIGS
       └─> Possible contamination or novel sequence

3. GENERATE REPORTS
   └─> HTML, PDF, text formats
```

Run QUAST on our assembly:

```bash
# Run QUAST with reference
quast.py \
    assembly/spades/contigs.fasta \
    -r reference/reference.fasta \
    -o results/quast \
    --threads 4 \
    --min-contig 200 \
    --labels "SPAdes_k21-77"

# View results
cat results/quast/report.txt
```

**Understanding QUAST output:**

```bash
python3 << 'EOF'
"""
Parse and explain QUAST results
"""
import os

print("="*70)
print("QUAST QUALITY METRICS EXPLAINED")
print("="*70)

# Read QUAST report
report_file = 'results/quast/report.txt'
if os.path.exists(report_file):
    with open(report_file, 'r') as f:
        lines = f.readlines()
    
    metrics = {}
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 2:
                metric = parts[0].strip()
                value = parts[1].strip()
                metrics[metric] = value
    
    print("\n1. CONTIGUITY METRICS")
    print("-" * 70)
    
    if '# contigs' in metrics:
        num_contigs = metrics['# contigs']
        print(f"Number of contigs: {num_contigs}")
        print(f"  → How fragmented is the assembly?")
        print(f"  → Lower is better")
        print(f"  → Our 10kb genome ideally: 1 contig")
        print(f"  → Realistic: 1-10 contigs")
    
    if 'Largest contig' in metrics:
        largest = metrics['Largest contig']
        print(f"\nLargest contig: {largest} bp")
        print(f"  → Longest continuous sequence")
        print(f"  → Should approach genome size (10,000 bp)")
    
    if 'Total length' in metrics:
        total = metrics['Total length']
        print(f"\nTotal assembly length: {total} bp")
        print(f"  → Sum of all contigs")
        print(f"  → Should match reference: 10,000 bp")
        print(f"  → If much larger: contamination or duplications")
        print(f"  → If much smaller: missing sequences")
    
    if 'N50' in metrics:
        n50 = metrics['N50']
        print(f"\nN50: {n50} bp")
        print(f"  → 50% of assembly is in contigs ≥ this length")
        print(f"  → Higher is better")
        print(f"  → Good: >5,000 bp (for our 10kb genome)")
        print(f"  → Excellent: >8,000 bp")
    
    if 'L50' in metrics:
        l50 = metrics['L50']
        print(f"\nL50: {l50}")
        print(f"  → Number of contigs containing 50% of assembly")
        print(f"  → Lower is better")
        print(f"  → Ideal: 1 (one contig has >50% of bases)")
    
    print("\n2. ALIGNMENT TO REFERENCE")
    print("-" * 70)
    
    if 'Genome fraction (%)' in metrics:
        fraction = metrics['Genome fraction (%)']
        print(f"Genome fraction: {fraction}%")
        print(f"  → % of reference covered by contigs")
        print(f"  → Should be >95%")
        print(f"  → If <90%: significant missing regions")
    
    if '# misassemblies' in metrics:
        misasm = metrics['# misassemblies']
        print(f"\nMisassemblies: {misasm}")
        print(f"  → Structural errors in assembly")
        print(f"  → Types:")
        print(f"    - Relocation: wrong position")
        print(f"    - Inversion: wrong orientation")
        print(f"    - Translocation: wrong chromosome")
        print(f"  → Should be 0-1 for our small genome")
    
    if '# mismatches per 100 kbp' in metrics:
        mismatches = metrics['# mismatches per 100 kbp']
        print(f"\nMismatches: {mismatches} per 100 kbp")
        print(f"  → Single base differences from reference")
        print(f"  → Caused by: sequencing errors, variants, assembly errors")
        print(f"  → Good: <100 per 100 kbp")
    
    if '# indels per 100 kbp' in metrics:
        indels = metrics['# indels per 100 kbp']
        print(f"\nIndels: {indels} per 100 kbp")
        print(f"  → Small insertions/deletions")
        print(f"  → Good: <10 per 100 kbp")
    
    print("\n3. QUALITY ASSESSMENT")
    print("-" * 70)
    
    # Overall quality judgment
    try:
        num_contigs_val = int(num_contigs)
        n50_val = int(n50)
        genome_frac = float(fraction)
        
        print("\nOverall Assembly Quality:")
        
        if num_contigs_val == 1 and genome_frac > 99:
            print("  ★★★★★ EXCELLENT")
            print("  → Perfect assembly!")
        elif num_contigs_val <= 5 and n50_val > 5000 and genome_frac > 95:
            print("  ★★★★☆ VERY GOOD")
            print("  → High quality, minor fragmentation")
        elif num_contigs_val <= 20 and n50_val > 2000 and genome_frac > 90:
            print("  ★★★☆☆ GOOD")
            print("  → Acceptable quality, some fragmentation")
        elif n50_val > 1000 and genome_frac > 80:
            print("  ★★☆☆☆ FAIR")
            print("  → Highly fragmented, usable with caution")
        else:
            print("  ★☆☆☆☆ POOR")
            print("  → Failed assembly, troubleshoot needed")
    except:
        pass

else:
    print("QUAST report not found. Run QUAST first.")

EOF
```

### 2.9 Biological Validation with BUSCO

**BUSCO Algorithm:**

```
Input: Assembly FASTA

1. SELECT LINEAGE DATABASE
   └─> E.g., "primates_odb10" contains ~13,000 conserved genes

2. FOR EACH BUSCO GENE:
   │
   ├─> SEARCH FOR GENE IN ASSEMBLY
   │   └─> Use BLAST or HMMER (profile HMM search)
   │
   ├─> CLASSIFY RESULT:
   │   ├─> Complete & Single-copy (C:S): Found once, full-length
   │   ├─> Complete & Duplicated (C:D): Found multiple times
   │   ├─> Fragmented (F): Found partially
   │   └─> Missing (M): Not found
   │
   └─> CALCULATE PERCENTAGE

3. GENERATE SCORE
   └─> C:95%[S:94%,D:1%],F:3%,M:2%
```

For our tiny 10kb demo genome, BUSCO won't work (genome too small). Let's create a mock demonstration:

```python
#!/usr/bin/env python3
"""
Simulate BUSCO assessment concept
"""

def simulate_busco():
    print("="*70)
    print("BUSCO CONCEPT DEMONSTRATION")
    print("="*70)
    
    print("\nWhat BUSCO does:")
    print("  1. Takes a database of universal single-copy genes")
    print("  2. Searches for these genes in your assembly")
    print("  3. Reports how many are found (= assembly completeness)")
    
    print("\n" + "-"*70)
    print("EXAMPLE: Hypothetical results for our assembly")
    print("-"*70)
    
    # Simulate results
    total_buscos = 100  # Simplified (real = ~13,000 for primates)
    complete_single = 92
    complete_duplicate = 3
    fragmented = 3
    missing = 2
    
    complete_total = complete_single + completed coverage: {expected_cov}×")
    print(f"Actual mean coverage: {actual_cov:.1f}×")
    print(f"Difference: {actual_cov - expected_cov:+.1f}×")
    
    if abs(actual_cov - expected_cov) < expected_cov * 0.1:
        print(f"✓ Coverage matches expectation (within 10%)")
    elif abs(actual_cov - expected_cov) < expected_cov * 0.2:
        print(f"⚠ Coverage somewhat differs (within 20%)")
    else:
        print(f"✗ Coverage significantly differs")
        print(f"  Possible causes:")
        print(f"    - Quality filtering removed many reads")
        print(f"    - Length filtering removed many reads")
        print(f"    - Assembly collapsed repeats (higher than expected)")
        print(f"    - Uneven coverage distribution")

try:
    analyze_coverage('assembly/spades/contigs.fasta')
except Exception as e:
    print(f"Error: {e}")
```

Run coverage analysis:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

---

## Part 4: Essential Bioinformatics Knowledge

### 4.1 Understanding K-mer Theory

**What exactly is a k-mer?**

```python
#!/usr/bin/env python3
"""
K-mer tutorial with visual examples
"""

def kmer_tutorial():
    print("="*70)
    print("K-MER FUNDAMENTALS")
    print("="*70)
    
    sequence = "ATCGATCGAT"
    print(f"\nOriginal sequence: {sequence}")
    print(f"Length: {len(sequence)} bp")
    
    for k in [3, 4, 5]:
        print(f"\n{'─'*70}")
        print(f"k = {k}")
        print(f"{'─'*70}")
        
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmers.append(kmer)
            print(f"Position {i:2d}: {kmer}")
        
        print(f"\nTotal k-mers: {len(kmers)}")
        print(f"Unique k-mers: {len(set(kmers))}")
        print(f"Duplicates: {len(kmers) - len(set(kmers))}")
        
        # Show duplicates
        from collections import Counter
        kmer_counts = Counter(kmers)
        duplicates = {k: v for k, v in kmer_counts.items() if v > 1}
        
        if duplicates:
            print(f"\nRepeated k-mers:")
            for kmer, count in duplicates.items():
                print(f"  {kmer}: appears {count} times")
                # Show positions
                positions = [i for i, km in enumerate(kmers) if km == kmer]
                print(f"    at positions: {positions}")
    
    print(f"\n{'='*70}")
    print("K-MER SIZE SELECTION GUIDE")
    print("="*70)
    
    print("""
Small k-mers (k=21-31):
  ✓ More overlaps found
  ✓ Better for low coverage regions
  ✓ More sensitive to all data
  ✗ More false overlaps (by chance)
  ✗ Less specific
  ✗ Harder to resolve repeats

Medium k-mers (k=41-55):
  → Balanced approach
  → Good for moderate coverage (30-50×)
  → Reasonable specificity and sensitivity

Large k-mers (k=77-127):
  ✓ Very specific (fewer false overlaps)
  ✓ Better repeat resolution
  ✓ Longer contigs (if coverage sufficient)
  ✗ Requires higher coverage
  ✗ Gaps in low coverage regions
  ✗ Miss some overlaps

Rule of thumb:
  k_max should be < read_length
  For 100bp reads: use k ≤ 77
  For 150bp reads: use k ≤ 127
  
  Start with k = read_length / 3
  End with k = read_length * 0.7-0.8
""")

kmer_tutorial()
```

Run it:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

### 4.2 Understanding Phred Quality Scores

```python
#!/usr/bin/env python3
"""
Complete guide to Phred quality scores
"""

def quality_score_tutorial():
    print("="*70)
    print("PHRED QUALITY SCORES EXPLAINED")
    print("="*70)
    
    print("""
Quality Score Formula:
  Q = -10 × log₁₀(P)
  
  Where:
    Q = Phred quality score
    P = Probability of error

Conversion:
  P = 10^(-Q/10)
""")
    
    print(f"\n{'─'*70}")
    print(f"{'Score':>6} {'Symbol':>8} {'Error Rate':>12} {'Accuracy':>10} {'Meaning'}")
    print(f"{'─'*70}")
    
    scores = [0, 10, 20, 30, 40]
    for q in scores:
        error_rate = 10 ** (-q / 10)
        accuracy = 1 - error_rate
        symbol = chr(q + 33)
        
        if q == 0:
            meaning = "Completely unreliable"
        elif q == 10:
            meaning = "Very poor"
        elif q == 20:
            meaning = "Acceptable (99% accurate)"
        elif q == 30:
            meaning = "Good (99.9% accurate)"
        elif q == 40:
            meaning = "Excellent (99.99% accurate)"
        
        print(f"{q:>6} {symbol:>8} {error_rate:>12.6f} {accuracy*100:>9.2f}% {meaning}")
    
    print(f"\n{'='*70}")
    print("PRACTICAL IMPLICATIONS")
    print("="*70)
    
    read_length = 100
    
    for q in [20, 30, 40]:
        error_rate = 10 ** (-q / 10)
        expected_errors = read_length * error_rate
        
        print(f"\nQ{q} read ({read_length} bp):")
        print(f"  Expected errors per read: {expected_errors:.2f}")
        print(f"  Reads needed for 1 error: {1/error_rate:.0f}")
        
        # In an assembly
        genome_size = 10000
        coverage = 30
        total_bases = genome_size * coverage
        total_errors = total_bases * error_rate
        
        print(f"  In 30× coverage of 10kb genome:")
        print(f"    Total sequenced bases: {total_bases:,}")
        print(f"    Expected errors: {total_errors:,.0f}")
    
    print(f"\n{'='*70}")
    print("QUALITY FILTERING RECOMMENDATIONS")
    print("="*70)
    
    print("""
For different applications:

Genome Assembly:
  Minimum: Q20 (99% accuracy)
  Recommended: Q25-Q30
  Rationale: Assemblers can handle some errors

Variant Calling:
  Minimum: Q30 (99.9% accuracy)
  Recommended: Q33+
  Rationale: Single errors become false variants

RNA-seq:
  Minimum: Q20-Q25
  Rationale: High coverage compensates

Metagenomics:
  Minimum: Q25-Q30
  Rationale: Mixed species make error correction harder
""")
    
    print(f"\n{'='*70}")
    print("INTERPRETING QUALITY STRINGS")
    print("="*70)
    
    # Examples
    examples = [
        ("IIIIIHHHHGGGGFFFFEEEE", "Excellent start, gradual degradation"),
        ("555555555555555555555", "Consistent Q20 throughout"),
        ("IIIII!!!!!!!!!!!!!!!!!", "Good start, catastrophic end (trim!)"),
        ("######$$$$$$$$$$$$$$$", "Poor quality throughout (discard!)"),
    ]
    
    for qual_str, description in examples:
        print(f"\nQuality string: {qual_str}")
        print(f"Description: {description}")
        
        scores = [ord(c) - 33 for c in qual_str]
        mean_q = sum(scores) / len(scores)
        min_q = min(scores)
        
        print(f"  Mean Q: {mean_q:.1f}")
        print(f"  Min Q: {min_q}")
        print(f"  Mean accuracy: {(1 - 10**(-mean_q/10))*100:.2f}%")

quality_score_tutorial()
```

Run it:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

### 4.3 Understanding Assembly Metrics (N50, L50, etc.)

```python
#!/usr/bin/env python3
"""
Interactive demonstration of assembly metrics
"""

def assembly_metrics_tutorial():
    print("="*70)
    print("ASSEMBLY METRICS EXPLAINED")
    print("="*70)
    
    # Example assembly
    contigs = [5000, 3000, 2000, 1500, 1000, 800, 500, 300, 200, 100]
    total_length = sum(contigs)
    
    print(f"\nExample Assembly:")
    print(f"Contigs: {contigs}")
    print(f"Total length: {total_length:,} bp")
    
    print(f"\n{'─'*70}")
    print("N50 CALCULATION (step-by-step)")
    print(f"{'─'*70}")
    
    print("""
N50 Definition:
  The length N for which 50% of all bases in the assembly
  are in contigs of size ≥ N
  
  Higher N50 = Better assembly (fewer, longer contigs)
""")
    
    # Calculate N50
    print(f"\nStep 1: Sort contigs by length (descending)")
    sorted_contigs = sorted(contigs, reverse=True)
    print(f"  {sorted_contigs}")
    
    print(f"\nStep 2: Calculate cumulative sum")
    print(f"{'Contig':>8} {'Length':>8} {'Cumsum':>10} {'% of total':>12}")
    print(f"{'─'*45}")
    
    cumsum = 0
    n50 = None
    n50_index = None
    
    for i, length in enumerate(sorted_contigs):
        cumsum += length
        pct = (cumsum / total_length) * 100
        marker = " ← N50" if n50 is None and cumsum >= total_length / 2 else ""
        print(f"{i+1:>8} {length:>8} {cumsum:>10} {pct:>11.1f}%{marker}")
        
        if n50 is None and cumsum >= total_length / 2:
            n50 = length
            n50_index = i + 1
    
    print(f"\nResult: N50 = {n50:,} bp")
    print(f"        L50 = {n50_index} (number of contigs)")
    print(f"\nInterpretation:")
    print(f"  50% of the assembly ({total_length//2:,} bp) is contained")
    print(f"  in {n50_index} contigs of length ≥ {n50:,} bp")
    
    # Calculate N90
    print(f"\n{'─'*70}")
    print("N90 CALCULATION")
    print(f"{'─'*70}")
    
    cumsum = 0
    n90 = None
    
    for length in sorted_contigs:
        cumsum += length
        if cumsum >= total_length * 0.9:
            n90 = length
            break
    
    print(f"N90 = {n90:,} bp")
    print(f"Meaning: 90% of assembly is in contigs ≥ {n90} bp")
    print(f"\nN90 is more stringent than N50")
    print(f"N90 shows quality of smaller contigs")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARING ASSEMBLIES")
    print("="*70)
    
    assemblies = {
        "Excellent": [45000, 5000],  # One huge contig
        "Good": [20000, 15000, 10000, 5000],  # Few large contigs
        "Poor": [2000]*25,  # Many small contigs
    }
    
    print(f"\n{'Assembly':>12} {'Contigs':>8} {'N50':>8} {'L50':>5} {'Quality'}")
    print(f"{'─'*50}")
    
    for name, contig_lens in assemblies.items():
        total = sum(contig_lens)
        sorted_lens = sorted(contig_lens, reverse=True)
        
        cumsum = 0
        n50_val = None
        l50_val = None
        
        for i, length in enumerate(sorted_lens):
            cumsum += length
            if cumsum >= total / 2:
                n50_val = length
                l50_val = i + 1
                break
        
        if name == "Excellent":
            quality = "★★★★★"
        elif name == "Good":
            quality = "★★★★☆"
        else:
            quality = "★★☆☆☆"
        
        print(f"{name:>12} {len(contig_lens):>8} {n50_val:>8} {l50_val:>5} {quality}")
    
    print(f"\n{'='*70}")
    print("OTHER IMPORTANT METRICS")
    print("="*70)
    
    print("""
Total Length:
  Sum of all contig lengths
  Should match expected genome size
  If much larger: duplication/contamination
  If much smaller: missing sequences

Largest Contig:
  Length of longest contig
  Should approach chromosome size (ideally)
  Indicates best-case assembly quality

Number of Contigs:
  How fragmented the assembly is
  Lower is better
  Affected by: coverage, repeats, quality

NG50 (with reference):
  Like N50 but uses reference genome size
  More fair for comparing assemblies
  
Genome Fraction:
  % of reference covered
  Requires reference genome
  Should be >95% for good assembly

Misassemblies:
  Structural errors in assembly
  Joins that shouldn't exist
  Should be minimized (ideally 0)
""")

assembly_metrics_tutorial()
```

Run it:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

### 4.4 Understanding Coverage Depth

```python
#!/usr/bin/env python3
"""
Coverage depth concepts and calculations
"""

def coverage_tutorial():
    print("="*70)
    print("SEQUENCING COVERAGE EXPLAINED")
    print("="*70)
    
    print("""
Coverage (Depth) Definition:
  Average number of reads covering each base in genome

Formula:
  Coverage = (Number_of_Reads × Read_Length) / Genome_Size
""")
    
    # Example calculations
    genome_size = 10000  # Our reference
    read_length = 100
    
    print(f"\n{'─'*70}")
    print(f"EXAMPLE: Our 10kb genome")
    print(f"{'─'*70}")
    
    print(f"\nGenome size: {genome_size:,} bp")
    print(f"Read length: {read_length} bp")
    
    print(f"\n{'Target Cov':>12} {'Reads Needed':>15} {'Total Bases':>15}")
    print(f"{'─'*45}")
    
    for coverage in [10, 20, 30, 50, 100]:
        reads_needed = (coverage * genome_size) // read_length
        total_bases = reads_needed * read_length
        
        print(f"{coverage:>12}× {reads_needed:>15,} {total_bases:>15,}")
    
    print(f"\n{'='*70}")
    print("COVERAGE REQUIREMENTS BY APPLICATION")
    print("="*70)
    
    applications = [
        ("Bacterial genome (draft)", "20-30×", "Quick assembly, some gaps OK"),
        ("Bacterial genome (complete)", "50-100×", "High quality, few gaps"),
        ("Human genome (draft)", "30-40×", "Acceptable quality"),
        ("Human genome (reference)", "80-100×", "Publication quality"),
        ("Variant calling", "30-50×", "Detect variants confidently"),
        ("De novo (no reference)", "50-100×", "Overcome repeat regions"),
    ]
    
    print(f"\n{'Application':>30} {'Coverage':>12} {'Rationale'}")
    print(f"{'─'*70}")
    
    for app, cov, rationale in applications:
        print(f"{app:>30} {cov:>12} {rationale}")
    
    print(f"\n{'='*70}")
    print("COVERAGE vs ASSEMBLY QUALITY")
    print("="*70)
    
    print("""
10× Coverage:
  ✗ Insufficient for good assembly
  ✗ Many gaps and errors
  → Only for quick draft or screening

20× Coverage:
  ⚠ Minimum acceptable
  → Many fragmented contigs
  → Missing low-complexity regions
  → OK for presence/absence analysis

30× Coverage:
  ✓ Good for standard assembly
  → Decent N50
  → Most regions covered
  → Standard for human genome projects

50× Coverage:
  ✓ Very good
  → Resolves most repeats
  → High N50
  → Few gaps

100× Coverage:
  ✓ Excellent but overkill
  → Diminishing returns
  → Wastes resources
  → Use for difficult genomes only

200×+ Coverage:
  ✗ Excessive
  → Actually can hurt assembly (too much data)
  → Computational burden
  → Consider downsampling to 80-100×
""")
    
    print(f"\n{'='*70}")
    print("UNEVEN COVERAGE")
    print("="*70}")
    
    print("""
Real sequencing has uneven coverage:

Causes:
  • GC bias (GC-rich regions under-represented)
  • PCR amplification bias
  • Sequencing platform bias
  • Random sampling variation

Impact:
  • Some regions: 50× coverage
  • Other regions: 10× coverage
  • Assembly breaks at low-coverage regions

Example:
  Mean coverage: 30×
  But actual distribution: 5× to 60×
  
  Regions with <20× will be problematic
  Need higher mean to ensure minimum everywhere
""")
    
    # Simulate coverage distribution
    import numpy as np
    
    print(f"\n{'─'*70}")
    print("COVERAGE DISTRIBUTION SIMULATION")
    print(f"{'─'*70}")
    
    np.random.seed(42)
    
    for mean_cov in [20, 30, 50]:
        # Simulate Poisson-like distribution
        coverages = np.random.poisson(mean_cov, 1000)
        
        low_cov = np.sum(coverages < 10)
        very_low_cov = np.sum(coverages < 5)
        
        print(f"\nMean coverage: {mean_cov}×")
        print(f"  Actual range: {np.min(coverages)}× to {np.max(coverages)}×")
        print(f"  Regions <10×: {low_cov/10:.1f}% (problematic)")
        print(f"  Regions <5×: {very_low_cov/10:.1f}% (very problematic)")
        
        if very_low_cov > 10:
            print(f"  ⚠ Too many low-coverage regions!")
        elif very_low_cov > 0:
            print(f"  → Some gaps expected")
        else:
            print(f"  ✓ Good coverage uniformity")

coverage_tutorial()
```

Run it:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

### 4.5 Why Repeats Are Hard

```python
#!/usr/bin/env python3
"""
Visual demonstration of why repeats cause assembly problems
"""

def repeat_problem_demo():
    print("="*70)
    print("THE REPEAT PROBLEM IN GENOME ASSEMBLY")
    print("="*70)
    
    # Simple example
    print("\n" + "─"*70)
    print("SCENARIO 1: No Repeats (Easy)")
    print("─"*70)
    
    genome_simple = "ATCGATCGTAGCTAGCGCTAGCTAGCTATGCATGC"
    print(f"\nGenome: {genome_simple}")
    print(f"Length: {len(genome_simple)} bp")
    
    # Generate reads
    read_len = 10
    reads = []
    for i in range(0, len(genome_simple) - read_len + 1, 5):
        reads.append(genome_simple[i:i+read_len])
    
    print(f"\nReads ({read_len} bp, step 5):")
    for i, read in enumerate(reads):
        print(f"  Read {i+1}: {read}")
    
    print("\nAssembly:")
    print(f"  Each read overlaps uniquely")
    print(f"  Path through reads is unambiguous")
    print(f"  Result: {genome_simple}")
    print(f"  ✓ Perfect assembly!")
    
    # With repeats
    print("\n" + "="*70)
    print("SCENARIO 2: With Repeats (Hard)")
    print("="*70)
    
    genome_repeat = "ATCG" + "CGCG"*5 + "TAGC" + "CGCG"*5 + "GCTA"
    print(f"\nGenome: {genome_repeat}")
    print(f"Length: {len(genome_repeat)} bp")
    print(f"\nStructure:")
    print(f"  [ATCG] - [CGCGCGCGCGCGCGCGCGCG] - [TAGC] - [CGCGCGCGCGCGCGCGCGCG] - [GCTA]")
    print(f"  unique      repeat (20bp)       unique      repeat (20bp)       unique")
    
    # Generate reads
    reads_repeat = []
    for i in range(0, len(genome_repeat) - read_len + 1, 5):
        reads_repeat.append(genome_repeat[i:i+read_len])
    
    print(f"\nSome reads:")
    for i in [0, 5, 10, 15]:
        if i < len(reads_repeat):
            read = reads_repeat[i]
            # Check if in repeat
            if "CGCGCGCG" in read:
                location = "← IN REPEAT (ambiguous!)"
            else:
                location = "← unique region"
            print(f"  Read {i+1}: {read} {location}")
    
    print("\n❌ PROBLEM:")
    print(f"  Reads in repeat region could come from EITHER copy")
    print(f"  Assembler cannot distinguish which repeat they're from")
    print(f"  Graph has multiple valid paths")
    
    print("\nPossible assembly outcomes:")
    print(f"  1. Collapse both repeats (WRONG - missing 20bp)")
    print(f"  2. Include both but break into fragments")
    print(f"  3. Misassemble (connect wrong regions)")
    
    # How to solve
    print("\n" + "="*70)
    print("SOLUTIONS TO REPEAT PROBLEM")
    print("="*70)
    
    print("""
1. LONGER READS
   If repeat = 20bp, but read = 30bp
   → Read spans entire repeat
   → Can uniquely place it
   
   Technologies:
   • PacBio HiFi: 10-25 kb reads
   • Oxford Nanopore: up to 100+ kb reads
   
2. PAIRED-END READS
   If repeat = 20bp, but insert size = 300bp
   → R1 and R2 span across repeat
   → Know which flanking regions connect
   
   Example:
   R1: [uniqueA]---repeat
   R2: repeat---[uniqueB]
   Insert constraint: 300bp apart
   → uniqueA connects to uniqueB via this repeat
   
3. MATE-PAIR LIBRARIES
   Longer insert sizes (2-10 kb)
   → Span multiple repeats
   → Scaffold assembly
   
4. OPTICAL MAPPING / Hi-C
   Very long-range information
   → Chromosome-scale scaffolding
   → Resolve complex repeat structures
   
5. HIGHER COVERAGE
   More reads = more spanning evidence
   BUT: Can't resolve repeats longer than read length
""")
    
    # Real-world examples
    print("\n" + "="*70)
    print("REAL-WORLD REPEAT CHALLENGES")
    print("="*70)
    
    print("""
Human Genome Repeats:
  • Short tandem repeats (STRs): 2-6 bp units
    Example: (CAG)n in Huntington's disease gene
    
  • Alu elements: ~300 bp, >1 million copies
    → Huge problem for short-read assembly
    
  • LINE-1 elements: ~6 kb, >500,000 copies
    → Requires long reads
    
  • Segmental duplications: >1 kb, >90% identity
    → Extremely difficult even with long reads
    
  • Centromeres: Mega-base tandem repeats
    → Still not fully resolved in human genome!
    
Plant Genomes:
  • Even more repetitive (50-90% repeats)
  • Polyploidy adds another layer
  → Require extensive long-read data
""")

repeat_problem_demo()
```

Run it:
```bash
python3 << 'SCRIPT'
# [paste the above script]
SCRIPT
```

---

## Part 5: Complete Working Example

Let's create a complete, executable script that does everything:

```bash
#!/bin/bash
#
# Complete Genome Assembly Pipeline
# From FASTQ generation to final validation
#

set -euo pipefail

echo "════════════════════════════════════════════════════════════════════════"
echo "  COMPLETE GENOME ASSEMBLY TUTORIAL"
echo "  From scratch to finished assembly"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
THREADS=4
MEMORY=8
COVERAGE=30
GENOME_SIZE=10000
READ_LENGTH=100
INSERT_SIZE=300

# Create directory structure
echo "[Step 1] Creating directory structure..."
mkdir -p tutorial_complete/{reference,reads,qc,trimmed,assembly,results}
cd tutorial_complete

# Generate reference genome
echo "[Step 2] Generating reference genome..."
python3 << 'PYTHON1'
import random
random.seed(42)

def generate_genome(length=10000):
    genome = []
    # Normal region
    for _ in range(3000):
        genome.append(random.choice(['A', 'T', 'G', 'C']))
    # AT-rich
    for _ in range(1000):
        genome.append(random.choice(['A', 'T']))
    # Repeat
    repeat = "ATCGATCGATCG"
    for _ in range(500 // len(repeat)):
        genome.extend(list(repeat))
    # GC-rich
    for _ in range(1000):
        genome.append(random.choice(['G', 'C']))
    # Another repeat
    repeat2 = "GCTAGCTAGCTA"
    for _ in range(500 // len(repeat2)):
        genome.extend(list(repeat2))
    # Normal
    for _ in range(4000):
        genome.append(random.choice(['A', 'T', 'G', 'C']))
    
    return ''.join(genome)

genome = generate_genome(10000)

with open('reference/reference.fasta', 'w') as f:
    f.write('>reference_genome\n')
    for i in range(0, len(genome), 80):
        f.write(genome[i:i+80] + '\n')

print(f"✓ Reference genome created: {len(genome)} bp")
PYTHON1

# Generate reads
echo "[Step 3] Generating paired-end reads (30× coverage)..."
python3 << 'PYTHON2'
import random
import gzip
random.seed(42)

def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))

def generate_quality(length, base_q=35, deg=0.1):
    quals = []
    for i in range(length):
        q = max(2, min(40, int(base_q - (i * deg) + random.gauss(0, 3))))
        quals.append(chr(q + 33))
    return ''.join(quals)

def introduce_errors(seq, rate=0.005):
    seq_list = list(seq)
    bases = ['A', 'T', 'G', 'C']
    for i in range(len(seq_list)):
        if random.random() < rate:
            seq_list[i] = random.choice([b for b in bases if b != seq_list[i]])
    return ''.join(seq_list)

# Load reference
with open('reference/reference.fasta') as f:
    ref = ''.join(line.strip() for line in f if not line.startswith('>'))

# Calculate reads needed
genome_size = len(ref)
read_length = 100
coverage = 30
num_reads = (coverage * genome_size) // (read_length * 2)

print(f"Generating {num_reads} read pairs...")

r1_reads, r2_reads = [], []

for i in range(num_reads):
    insert_size = int(random.gauss(300, 50))
    insert_size = max(200, insert_size)
    
    start = random.randint(0, len(ref) - insert_size)
    fragment = ref[start:start + insert_size]
    
    r1_seq = fragment[:read_length]
    r2_seq = reverse_complement(fragment[-read_length:])
    
    r1_seq = introduce_errors(r1_seq, 0.005)
    r2_seq = introduce_errors(r2_seq, 0.01)
    
    r1_qual = generate_quality(len(r1_seq), 37, 0.1)
    r2_qual = generate_quality(len(r2_seq), 33, 0.15)
    
    header = f"@READ_{i}:1:1:1:{i}:0 1:N:0:ATCG"
    r1_reads.append(f"{header}\n{r1_seq}\n+\n{r1_qual}\n")
    
    header_r2 = header.replace(" 1:N:", " 2:N:")
    r2_reads.append(f"{header_r2}\n{r2_seq}\n+\n{r2_qual}\n")

with gzip.open('reads/sample_R1.fastq.gz', 'wt') as f:
    f.writelines(r1_reads)

with gzip.open('reads/sample_R2.fastq.gz', 'wt') as f:
    f.writelines(r2_reads)

print(f"✓ Generated {num_reads} paired-end reads")
PYTHON2

# Quality control
echo "[Step 4] Running FastQC..."
if command -v fastqc &> /dev/null; then
    fastqc -q -t $THREADS -o qc/ reads/sample_R*.fastq.gz
    echo "✓ FastQC complete"
else
    echo "⚠ FastQC not installed, skipping"
fi

# Trimming
echo "[Step 5] Trimming and filtering reads..."
if command -v fastp &> /dev/null; then
    fastp \
        -i reads/sample_R1.fastq.gz \
        -I reads/sample_R2.fastq.gz \
        -o trimmed/sample_R1.trim.fastq.gz \
        -O trimmed/sample_R2.trim.fastq.gz \
        --detect_adapter_for_pe \
        --qualified_quality_phred 20 \
        --unqualified_percent_limit 40 \
        --length_required 50 \
        --thread $THREADS \
        --html qc/fastp.html \
        --json qc/fastp.json \
        2>&1 | grep -E "(Read|passed|low quality)" || true
    echo "✓ Trimming complete"
else
    echo "⚠ fastp not installed, using raw reads"
    cp reads/sample_R1.fastq.gz trimmed/sample_R1.trim.fastq.gz
    cp reads/sample_R2.fastq.gz trimmed/sample_R2.trim.fastq.gz
fi

# Assembly
echo "[Step 6] Running SPAdes assembly..."
if command -v spades.py &> /dev/null; then
    spades.py \
        -1 trimmed/sample_R1.trim.fastq.gz \
        -2 trimmed/sample_R2.trim.fastq.gz \
        -o assembly/spades \
        --threads $THREADS \
        --memory $MEMORY \
        -k 21,33,55,77 \
        --careful \
        2>&1 | grep -E "(==|NOTICE|Thank you|Warnings)" || true
    echo "✓ Assembly complete"
else
    echo "✗ SPAdes not installed!"
    exit 1
fi

# Quality assessment
echo "[Step 7] Running QUAST..."
if command -v quast.py &> /dev/null; then
    quast.py \
        assembly/spades/contigs.fasta \
        -r reference/reference.fasta \
        -o results/quast \
        --threads $THREADS \
        --min-contig 200 \
        --labels "Assembly" \
        2>&1 | grep -E "(Analyze|metrics|Done)" || true
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  QUAST RESULTS"
    echo "════════════════════════════════════════════════════════════════════════"
    cat results/quast/report.txt | head -20
    echo "✓ Full report: results/quast/report.html"
else
    echo "⚠ QUAST not installed, skipping"
fi

# Detailed analysis
echo ""
echo "[Step 8] Performing detailed analysis..."
python3 << 'PYTHON3'
from Bio import SeqIO
import numpy as np

print("\n════════════════════════════════════════════════════════════════════════")
print("  DETAILED ASSEMBLY ANALYSIS")
print("════════════════════════════════════════════════════════════════════════")

# Load data
try:
    contigs = list(SeqIO.parse('assembly/spades/contigs.fasta', 'fasta'))
    reference = list(SeqIO.parse('reference/reference.fasta', 'fasta'))[0]
except:
    print("✗ Could not load assembly files")
    exit(1)

# Basic stats
lengths = [len(c.seq) for c in contigs]
total_length = sum(lengths)
ref_length = len(reference.seq)

print(f"\n📊 ASSEMBLY STATISTICS:")
print(f"   Contigs: {len(contigs)}")
print(f"   Total length: {total_length:,} bp")
print(f"   Reference length: {ref_length:,} bp")
print(f"   Difference: {total_length - ref_length:+,} bp ({(total_length - ref_length)/ref_length*100:+.2f}%)")

# N50
sorted_lengths = sorted(lengths, reverse=True)
cumsum = 0
n50 = 0
l50 = 0
for i, length in enumerate(sorted_lengths):
    cumsum += length
    if cumsum >= total_length / 2:
        n50 = length
        l50 = i + 1
        break

print(f"\n📏 CONTIGUITY:")
print(f"   Longest contig: {max(lengths):,} bp ({max(lengths)/ref_length*100:.1f}% of genome)")
print(f"   N50: {n50:,} bp")
print(f"   L50: {l50}")

# Coverage analysis
coverages = []
for contig in contigs:
    if '_cov_' in contig.id:
        try:
            cov = float(contig.id.split('_cov_')[1].split('_')[0])
            coverages.append(cov)
        except:
            pass

if coverages:
    print(f"\n📈 COVERAGE:")
    print(f"   Mean: {np.mean(coverages):.1f}×")
    print(f"   Median: {np.median(coverages):.1f}×")
    print(f"   Range: {min(coverages):.1f}× to {max(coverages):.1f}×")

# Alignment check
ref_seq = str(reference.seq)
aligned = 0
for i, contig in enumerate(contigs[:5]):
    contig_seq = str(contig.seq)
    if contig_seq in ref_seq:
        pos = ref_seq.find(contig_seq)
        aligned += 1
        print(f"   ✓ Contig {i+1} ({len(contig_seq)} bp) aligns at position {pos:,}")
    else:
        # Try reverse complement
        rc = str(contig.seq.reverse_complement())
        if rc in ref_seq:
            pos = ref_seq.find(rc)
            aligned += 1
            print(f"   ✓ Contig {i+1} ({len(contig_seq)} bp) aligns (RC) at position {pos:,}")

# Quality score
print(f"\n⭐ QUALITY GRADE:")
score = 0

# Length match (25 points)
length_diff_pct = abs(total_length - ref_length) / ref_length * 100
if length_diff_pct < 1:
    score += 25
elif length_diff_pct < 5:
    score += 20
elif length_diff_pct < 10:
    score += 15
else:
    score += 10

# N50 quality (35 points)
n50_pct = n50 / ref_length
if n50_pct > 0.8:
    score += 35
elif n50_pct > 0.5:
    score += 30
elif n50_pct > 0.3:
    score += 20
else:
    score += 10

# Number of contigs (25 points)
if len(contigs) == 1:
    score += 25
elif len(contigs) <= 5:
    score += 20
elif len(contigs) <= 10:
    score += 15
else:
    score += 10

# Coverage (15 points)
if coverages:
    cv = np.std(coverages) / np.mean(coverages)
    if cv < 0.3:
        score += 15
    elif cv < 0.5:
        score += 10
    else:
        score += 5

print(f"   Length accuracy:  {'✓' if length_diff_pct < 5 else '⚠'}")
print(f"   Contiguity (N50): {'✓' if n50_pct > 0.5 else '⚠'}")
print(f"   Fragmentation:    {'✓' if len(contigs) <= 5 else '⚠'}")
if coverages:
    print(f"   Coverage:         {'✓' if cv < 0.5 else '⚠'}")

print(f"\n   Total Score: {score}/100")

if score >= 90:
    grade = "A+ (Excellent)"
    emoji = "🏆"
elif score >= 80:
    grade = "A  (Very Good)"
    emoji = "🌟"
elif score >= 70:
    grade = "B  (Good)"
    emoji = "👍"
elif score >= 60:
    grade = "C  (Acceptable)"
    emoji = "👌"
else:
    grade = "D  (Needs Improvement)"
    emoji = "⚠️"

print(f"   Grade: {grade} {emoji}")

PYTHON3

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 OUTPUT FILES:"
echo "   reference/reference.fasta        - Reference genome"
echo "   reads/sample_R*.fastq.gz         - Raw reads"
echo "   trimmed/sample_R*.trim.fastq.gz  - Trimmed reads"
echo "   assembly/spades/contigs.fasta    - Final assembly"
echo "   results/quast/report.html        - Quality report"
echo ""
echo "🔍 TO VIEW RESULTS:"
echo "   cat results/quast/report.txt"
echo "   firefox results/quast/report.html"
echo ""
echo "✨ Done! Check the assembly quality above."
echo ""
```

Save this as `complete_tutorial.sh` and run it:

```bash
chmod +x complete_tutorial.sh
./complete_tutorial.sh
```

---

## Part 6: Summary of Key Concepts

### 6.1 Quick Reference Card

```bash
cat > QUICK_REFERENCE.md << 'EOF'
# Genome Assembly Quick Reference

## FASTQ Format
```
@READ_ID
SEQUENCE
+
QUALITY
```
- Quality: Phred+33 encoding
- Q20 = 99% accuracy (minimum for assembly)
- Q30 = 99.9% accuracy (recommended)

## Coverage Calculation
```
Coverage = (Reads × Read_Length) / Genome_Size
```
- Minimum: 20×
- Recommended: 30-50×
- Excellent: 80-100×

## K-mer Selection
```
k_min = read_length / 3
k_max = read_length × 0.7
```
- Smaller k: more sensitive, more ambiguous
- Larger k: more specific, better contiguity

## Assembly Metrics
- **N50**: 50% of assembly in contigs ≥ this length (higher = better)
- **L50**: Number of contigs containing 50% of assembly (lower = better)
- **Contigs**: Total number (lower = better)
- **Genome fraction**: % of reference covered (>95% = good)

## Quality Thresholds

### Excellent Assembly
- N50 > 80% of genome
- L50 = 1-2
- Genome fraction > 99%
- <5 misassemblies

### Good Assembly
- N50 > 50% of genome
- L50 < 5
- Genome fraction > 95%
- <10 misassemblies

### Poor Assembly
- N50 < 20% of genome
- L50 > 20
- Genome fraction < 90%
- >20 misassemblies

## Tool Comparison

| Tool | Speed | Memory | Quality | Use Case |
|------|-------|--------|---------|----------|
| SPAdes | Slow | High | Excellent | Small-medium genomes |
| MEGAHIT | Fast | Low | Good | Large genomes, limited RAM |
| Canu | Slow | Very High | Excellent | Long reads only |

## Common Issues

### Low N50
- Increase coverage
- Use larger k-mers
- Try paired-end if using single-end

### High Contig Count
- Increase coverage
- Improve read quality
- Use longer reads

### Wrong Size
- Too large: contamination or duplication
- Too small: low coverage or poor quality

### Low Mapping Rate
- Adapter contamination
- Wrong species
- Incomplete assembly

## Command Cheat Sheet

```bash
# FastQC
fastqc -t 8 -o qc/ reads.fastq.gz

# fastp (trimming)
fastp -i R1.fq.gz -I R2.fq.gz \
      -o R1.trim.fq.gz -O R2.trim.fq.gz \
      --qualified_quality_phred 20 \
      --length_required 50

# SPAdes
spades.py -1 R1.fq.gz -2 R2.fq.gz \
          -o assembly/ \
          -t 16 -m 100 \
          -k 21,33,55,77 \
          --careful

# QUAST
quast.py contigs.fasta \
         -r reference.fasta \
         -o quast_results/ \
         -t 8

# Read mapping (validation)
bwa index assembly.fasta
bwa mem -t 16 assembly.fasta R1.fq.gz R2.fq.gz | \
    samtools sort -o mapped.bam
samtools flagstat mapped.bam
```

## Troubleshooting Decision Tree

```
Assembly failed?
├─ Out of memory? → Use MEGAHIT or reduce k-mers
├─ Low coverage? → Get more data
├─ Poor quality? → Stricter quality filtering
└─ Repeats? → Use paired-end or long reads

Assembly fragmented?
├─ Low coverage? → Increase to 50×+
├─ Short reads? → Add long-read data
├─ Single-end? → Switch to paired-end
└─ Wrong k-mers? → Use larger k values

Assembly too large?
├─ Check for contamination (BLAST)
├─ Check for duplications (purge_dups)
└─ Verify sample identity

Assembly too small?
├─ Low coverage? → Increase coverage
├─ Over-filtering? → Less strict trimming
└─ Wrong parameters? → Try different assembler
```

## Best Practices

1. **Always run QC** before and after trimming
2. **Use paired-end** for eukaryotes
3. **Check coverage** (30-50× minimum)
4. **Validate results** with multiple metrics
5. **Map reads back** to verify assembly
6. **Compare to reference** if available
7. **Check for contamination** (BLAST/Kraken)
8. **Document everything** (versions, parameters)

## Resources
- SPAdes: https://github.com/ablab/spades
- QUAST: http://quast.sourceforge.net/
- BUSCO: https://busco.ezlab.org/
- Bandage: https://rrwick.github.io/Bandage/
EOF

cat QUICK_REFERENCE.md
```

---

## Final Summary

**What we covered:**

1. ✅ **FASTQ format** - Structure and quality scores
2. ✅ **Paired-end sequencing** - How it works and why it's better
3. ✅ **Quality control** - FastQC algorithm and interpretation
4. ✅ **Read trimming** - fastp algorithm and parameter selection
5. ✅ **De Bruijn graphs** - Core assembly algorithm with examples
6. ✅ **K-mer selection** - Why multiple k-mers and how to choose
7. ✅ **Coverage** - Calculation and requirements
8. ✅ **Assembly metrics** - N50, L50, and quality assessment
9. ✅ **Repeat problems** - Why they're hard and solutions
10. ✅ **Complete workflow** - End-to-end reproducible pipeline

**Key takeaways:**

- **Paired-end >> Single-end** for assembly (10× better N50)
- **Coverage matters**: 30× minimum, 50× recommended
- **K-mers are critical**: Use multiple sizes iteratively
- **Quality control is essential**: Bad data = bad assembly
- **Validation is mandatory**: Use multiple metrics (QUAST, BUSCO, mapping)
- **Repeats are the hardest problem**: Need long reads or deep coverage

**Your dummy data** demonstrates all these concepts with:
- 10kb reference genome
- 30× coverage paired-end reads
- Realistic quality scores and errors
- Repetitive regions
- Complete assembly pipeline

You can now scale this to real projects! 🚀
