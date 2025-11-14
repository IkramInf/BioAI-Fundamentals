# Complete NGS Pipeline Guide: Sequencing Type-Specific Workflows

## Table of Contents
1. [WGS (Whole Genome Sequencing)](#wgs-whole-genome-sequencing)
2. [WES (Whole Exome Sequencing)](#wes-whole-exome-sequencing)
3. [RNA-seq](#rna-seq)
4. [Targeted Sequencing](#targeted-sequencing)
5. [ChIP-seq](#chip-seq)
6. [ATAC-seq](#atac-seq)
7. [Bisulfite-seq](#bisulfite-seq)
8. [scRNA-seq](#scrna-seq)
9. [Metagenomics](#metagenomics)
10. [Quick Comparison Table](#quick-comparison-table)

---

## WGS (Whole Genome Sequencing)

### Pipeline Overview
WGS aims to sequence the entire genome with uniform coverage, identifying all types of genetic variations across coding and non-coding regions.

### Workflow Steps

#### 1. Quality Control
```bash
# FastQC for quality assessment
fastqc sample_R1.fastq.gz sample_R2.fastq.gz

# Trimming adapters and low-quality bases
trimmomatic PE sample_R1.fastq.gz sample_R2.fastq.gz \
  sample_R1_paired.fq.gz sample_R1_unpaired.fq.gz \
  sample_R2_paired.fq.gz sample_R2_unpaired.fq.gz \
  ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 MINLEN:36
```

#### 2. Alignment
```bash
# BWA-MEM is preferred for >70bp reads
bwa mem -t 16 -R '@RG\tID:sample\tSM:sample\tPL:ILLUMINA' \
  reference.fa sample_R1_paired.fq.gz sample_R2_paired.fq.gz | \
  samtools sort -@ 8 -o sample_sorted.bam
```

#### 3. Post-processing
```bash
# Mark duplicates
gatk MarkDuplicates \
  -I sample_sorted.bam \
  -O sample_dedup.bam \
  -M metrics.txt

# Base Quality Score Recalibration (BQSR)
gatk BaseRecalibrator \
  -I sample_dedup.bam \
  -R reference.fa \
  --known-sites dbsnp.vcf \
  -O recal_data.table

gatk ApplyBQSR \
  -I sample_dedup.bam \
  -R reference.fa \
  --bqsr-recal-file recal_data.table \
  -O sample_final.bam
```

#### 4. Coverage Analysis
```bash
# Check coverage uniformity
samtools depth sample_final.bam | \
  awk '{sum+=$3; count++} END {print "Mean coverage:", sum/count}'

# Identify low-coverage regions
bedtools genomecov -ibam sample_final.bam -bga | \
  awk '$4 < 10' > low_coverage_regions.bed
```

#### 5. Variant Calling
```bash
# SNPs and Indels with GATK HaplotypeCaller
gatk HaplotypeCaller \
  -R reference.fa \
  -I sample_final.bam \
  -O sample_variants.vcf

# Structural variants with Manta
configManta.py \
  --bam sample_final.bam \
  --referenceFasta reference.fa \
  --runDir manta_run
manta_run/runWorkflow.py

# CNV calling with CNVkit
cnvkit.py batch sample_final.bam \
  -r reference.cnn \
  -d output_dir/
```

### Practical Tips

**Coverage Requirements:**
- Human WGS: 30-40X for germline, 60-100X for somatic variants
- Lower coverage (10-15X) acceptable for population studies

**Pros:**
- Detects variants in coding and non-coding regions
- Identifies all variant types (SNPs, indels, SVs, CNVs)
- No capture bias or target enrichment artifacts
- Discovers novel/unexpected variants

**Cons:**
- High sequencing cost (~$600-1000 per sample)
- Large data storage requirements (100-200 GB per sample)
- Complex analysis for non-coding variants
- Longer processing time

### Example Use Case
**Clinical scenario:** A patient with suspected genetic disorder where WES was negative. WGS identified a deep intronic variant creating a cryptic splice site in the CFTR gene, explaining the phenotype.

---

## WES (Whole Exome Sequencing)

### Pipeline Overview
WES targets only protein-coding exons (~1-2% of genome) using capture probes, providing cost-effective variant detection in functional regions.

### Workflow Steps

#### 1. Quality Control
Same as WGS, but verify capture efficiency:
```bash
# Check on-target rate
bedtools intersect -a sample_sorted.bam -b exome_targets.bed -bed | \
  wc -l > on_target_reads.txt
```

#### 2. Alignment
```bash
# Use BWA-MEM (identical to WGS)
bwa mem -t 16 -R '@RG\tID:sample\tSM:sample\tPL:ILLUMINA' \
  reference.fa sample_R1.fq.gz sample_R2.fq.gz | \
  samtools sort -@ 8 -o sample_sorted.bam
```

#### 3. Post-processing (WES-specific)
```bash
# Mark duplicates (higher duplication in WES)
gatk MarkDuplicates \
  -I sample_sorted.bam \
  -O sample_dedup.bam \
  -M metrics.txt

# BQSR (same as WGS)
gatk BaseRecalibrator -I sample_dedup.bam -R reference.fa \
  --known-sites dbsnp.vcf -O recal_data.table

gatk ApplyBQSR -I sample_dedup.bam -R reference.fa \
  --bqsr-recal-file recal_data.table -O sample_final.bam
```

#### 4. Target-Specific Coverage
```bash
# Coverage within target regions
bedtools coverage -a exome_targets.bed -b sample_final.bam -hist > coverage_hist.txt

# Find poorly covered exons
bedtools coverage -a exome_targets.bed -b sample_final.bam -mean | \
  awk '$4 < 20' > low_coverage_exons.bed
```

#### 5. Variant Calling with BED Restriction
```bash
# Call variants only in target regions
gatk HaplotypeCaller \
  -R reference.fa \
  -I sample_final.bam \
  -L exome_targets.bed \
  -ip 100 \
  -O sample_variants.vcf

# Filter to target regions + padding
bcftools view -R exome_targets_padded.bed sample_variants.vcf > filtered_variants.vcf
```

### Practical Tips

**Coverage Requirements:**
- Minimum 100X mean coverage recommended
- Aim for >90% of targets at ≥20X coverage

**Target File Considerations:**
- Use manufacturer-provided BED file (Agilent, Twist, IDT)
- Add 100-200bp padding to capture splice sites
- Update BED files periodically (exome definitions change)

**Pros:**
- Cost-effective (~$200-400 per sample)
- Smaller data size (8-12 GB per sample)
- Focused on functional variants
- Higher coverage depth per dollar
- Easier interpretation (coding variants)

**Cons:**
- Misses non-coding regulatory variants
- Capture bias: GC-rich regions under-represented
- Cannot detect large SVs or CNVs reliably
- Target bias between capture kits
- Splice site variants at exon boundaries may be missed

### Example Use Case
**Research cohort:** Screening 500 patients for rare Mendelian disease variants. WES identified compound heterozygous mutations in ABCA4 gene in multiple patients with Stargardt disease, at 1/5th the cost of WGS.

---

## RNA-seq

### Pipeline Overview
RNA-seq sequences expressed transcripts, requiring splice-aware alignment to handle reads spanning exon junctions. Focus is on quantification and differential expression, not variant calling.

### Workflow Steps

#### 1. Quality Control (RNA-specific)
```bash
# FastQC and check for rRNA contamination
fastqc sample_R1.fastq.gz sample_R2.fastq.gz

# Remove rRNA if needed
sortmerna --ref rRNA_databases.fasta \
  --reads sample_R1.fastq.gz --reads sample_R2.fastq.gz \
  --aligned rRNA_reads --other cleaned_reads --fastx

# Adapter trimming
trim_galore --paired sample_R1.fastq.gz sample_R2.fastq.gz
```

#### 2. Splice-Aware Alignment
```bash
# STAR (2-pass mode for novel junctions)
# First pass
STAR --runMode genomeGenerate \
  --genomeDir genome_index \
  --genomeFastaFiles reference.fa \
  --sjdbGTFfile annotations.gtf \
  --sjdbOverhang 99

STAR --genomeDir genome_index \
  --readFilesIn sample_R1.fq.gz sample_R2.fq.gz \
  --readFilesCommand zcat \
  --outFileNamePrefix sample_pass1_ \
  --outSAMtype BAM SortedByCoordinate

# Second pass (using detected junctions)
STAR --genomeDir genome_index \
  --readFilesIn sample_R1.fq.gz sample_R2.fq.gz \
  --readFilesCommand zcat \
  --sjdbFileChrStartEnd sample_pass1_SJ.out.tab \
  --outFileNamePrefix sample_final_ \
  --outSAMtype BAM SortedByCoordinate \
  --quantMode GeneCounts

# Alternative: HISAT2 (faster, less memory)
hisat2-build reference.fa reference_index
hisat2 -p 16 --dta -x reference_index \
  -1 sample_R1.fq.gz -2 sample_R2.fq.gz | \
  samtools sort -@ 8 -o sample_aligned.bam
```

#### 3. Post-processing
```bash
# NO duplicate removal (biological duplicates are informative)
# Index BAM for visualization
samtools index sample_final_Aligned.sortedByCoord.out.bam

# QC with RSeQC
infer_experiment.py -i sample_aligned.bam -r gene_model.bed
read_distribution.py -i sample_aligned.bam -r gene_model.bed
junction_annotation.py -i sample_aligned.bam -r gene_model.bed -o output_prefix
```

#### 4. Quantification
```bash
# Gene-level counts with featureCounts
featureCounts -p -t exon -g gene_id -a annotations.gtf \
  -o gene_counts.txt sample_aligned.bam

# Transcript-level with StringTie
stringtie sample_aligned.bam -G annotations.gtf \
  -o sample_transcripts.gtf -e -B -A gene_abundances.txt

# Alternative: Salmon (pseudo-alignment, faster)
salmon index -t transcriptome.fa -i salmon_index
salmon quant -i salmon_index -l A \
  -1 sample_R1.fq.gz -2 sample_R2.fq.gz \
  -o sample_quant --validateMappings
```

#### 5. Differential Expression Analysis
```R
# DESeq2 in R
library(DESeq2)

# Load count matrix
countData <- read.table("gene_counts.txt", header=TRUE, row.names=1)
colData <- data.frame(condition = c("control", "control", "treated", "treated"))

# Create DESeq object
dds <- DESeqDataSetFromMatrix(countData = countData,
                              colData = colData,
                              design = ~ condition)

# Run analysis
dds <- DESeq(dds)
res <- results(dds, contrast=c("condition", "treated", "control"))

# Volcano plot and heatmaps
plotMA(res)
```

### Practical Tips

**Library Considerations:**
- Stranded libraries preserve strand information (use `--rna-strandness RF` in HISAT2)
- Poly-A selection vs rRNA depletion affects results
- Read length: 75-150bp, paired-end preferred

**Coverage Requirements:**
- Minimum 20-30M reads per sample
- Differential expression: 30-50M reads
- Isoform analysis: 50-100M reads

**Pros:**
- Quantifies gene expression levels
- Detects novel transcripts and isoforms
- Identifies alternative splicing events
- Can detect fusion genes
- Reveals allele-specific expression

**Cons:**
- Cannot detect unexpressed genes
- 3' bias in degraded RNA samples
- PCR duplicates hard to distinguish from high expression
- Reference dependency affects novel transcript detection
- Batch effects common in expression studies

### Example Use Case
**Cancer research:** Comparing tumor vs normal tissue RNA-seq identified upregulation of PD-L1 and downregulation of immune checkpoint genes. Alternative splicing analysis revealed a novel BCR-ABL fusion transcript isoform in CML patients.

---

## Targeted Sequencing

### Pipeline Overview
Targeted sequencing focuses on a small custom gene panel (10s to 100s of genes), achieving ultra-high coverage for sensitive variant detection in specific loci.

### Workflow Steps

#### 1. Quality Control
```bash
# Standard QC
fastqc *.fastq.gz

# Adapter trimming (primers may need removal)
cutadapt -a AGATCGGAAGAGC -A AGATCGGAAGAGC \
  -o sample_R1_trimmed.fq.gz -p sample_R2_trimmed.fq.gz \
  sample_R1.fq.gz sample_R2.fq.gz
```

#### 2. Alignment
```bash
# BWA-MEM or Bowtie2 (similar to WGS)
bwa mem -t 8 -R '@RG\tID:sample\tSM:sample' \
  reference.fa sample_R1_trimmed.fq.gz sample_R2_trimmed.fq.gz | \
  samtools sort -@ 4 -o sample_sorted.bam

# Bowtie2 alternative (good for small panels)
bowtie2 -p 8 --very-sensitive -x reference_index \
  -1 sample_R1_trimmed.fq.gz -2 sample_R2_trimmed.fq.gz | \
  samtools sort -@ 4 -o sample_sorted.bam
```

#### 3. Post-processing
```bash
# Mark duplicates
gatk MarkDuplicates -I sample_sorted.bam -O sample_dedup.bam -M metrics.txt

# BQSR
gatk BaseRecalibrator -I sample_dedup.bam -R reference.fa \
  --known-sites dbsnp.vcf -L target_panel.bed -O recal_data.table
gatk ApplyBQSR -I sample_dedup.bam -R reference.fa \
  --bqsr-recal-file recal_data.table -O sample_final.bam
```

#### 4. Target-Specific Quality Metrics
```bash
# Ultra-deep coverage assessment (expect 1000X+)
samtools depth -b target_panel.bed sample_final.bam | \
  awk '{sum+=$3; count++} END {print "Mean coverage:", sum/count}'

# Per-amplicon coverage (for amplicon-based panels)
bedtools multicov -bams sample_final.bam -bed amplicon_targets.bed > amplicon_coverage.txt

# Check for off-target reads
total_reads=$(samtools view -c sample_final.bam)
on_target=$(samtools view -c -L target_panel.bed sample_final.bam)
echo "On-target percentage: $(echo "scale=2; $on_target*100/$total_reads" | bc)%"
```

#### 5. Variant Calling (High Sensitivity)
```bash
# GATK HaplotypeCaller with low thresholds
gatk HaplotypeCaller \
  -R reference.fa \
  -I sample_final.bam \
  -L target_panel.bed \
  -O sample_variants.vcf \
  --min-base-quality-score 20 \
  --standard-min-confidence-threshold-for-calling 10

# For somatic variants (tumor-only or tumor-normal)
gatk Mutect2 \
  -R reference.fa \
  -I tumor_final.bam \
  -I normal_final.bam \
  -normal normal_sample \
  -L target_panel.bed \
  -O somatic_variants.vcf \
  --germline-resource gnomad.vcf

# Low-frequency variant detection with VarDict
vardict-java -G reference.fa -f 0.01 -N sample \
  -b sample_final.bam -c 1 -S 2 -E 3 target_panel.bed | \
  teststrandbias.R | var2vcf_valid.pl -N sample -E -f 0.01 > variants.vcf
```

### Practical Tips

**Panel Design Considerations:**
- Amplicon-based (AmpliSeq, QIAseq): good for FFPE, higher duplicate rates
- Hybrid capture: more uniform coverage, better for large panels
- Include 50-100bp padding around targets

**Coverage Requirements:**
- Germline variants: 500-1000X minimum
- Somatic variants (cancer): 1000-5000X for low VAF detection
- Liquid biopsy/ctDNA: 10,000-50,000X

**Pros:**
- Ultra-deep coverage enables low-frequency variant detection
- Cost-effective for focused studies ($50-150 per sample)
- Fast turnaround time (hours to days)
- Small data size (1-5 GB per sample)
- Ideal for clinical actionable genes
- Can detect variants down to 0.1-1% VAF

**Cons:**
- Limited to pre-selected genes
- Cannot discover unexpected variants
- Primer/probe design affects some regions
- High duplicate rates in small panels
- Amplicon-based panels have primer bias

### Example Use Case
**Clinical oncology:** A 50-gene cancer hotspot panel detected an EGFR T790M resistance mutation at 2% VAF in liquid biopsy from NSCLC patient progressing on first-line TKI therapy, prompting switch to osimertinib. The ultra-deep coverage (5000X) was critical for detecting this low-frequency resistance clone.

---

## ChIP-seq

### Pipeline Overview
ChIP-seq identifies genome-wide protein-DNA binding sites or histone modifications by sequencing DNA fragments immunoprecipitated with specific antibodies. Focus is on peak calling, not variant detection.

### Workflow Steps

#### 1. Quality Control
```bash
# FastQC
fastqc ChIP_sample.fastq.gz Input_control.fastq.gz

# Adapter trimming
trim_galore ChIP_sample.fastq.gz Input_control.fastq.gz
```

#### 2. Alignment
```bash
# Bowtie2 (standard for ChIP-seq)
bowtie2-build reference.fa reference_index

bowtie2 -p 8 -x reference_index -U ChIP_sample_trimmed.fq.gz | \
  samtools sort -@ 4 -o ChIP_sorted.bam

bowtie2 -p 8 -x reference_index -U Input_control_trimmed.fq.gz | \
  samtools sort -@ 4 -o Input_sorted.bam

# Index BAMs
samtools index ChIP_sorted.bam
samtools index Input_sorted.bam
```

#### 3. Post-processing
```bash
# Remove duplicates (important for ChIP-seq)
picard MarkDuplicates -I ChIP_sorted.bam -O ChIP_dedup.bam \
  -M metrics.txt -REMOVE_DUPLICATES true

picard MarkDuplicates -I Input_sorted.bam -O Input_dedup.bam \
  -M metrics.txt -REMOVE_DUPLICATES true

# Remove low-quality and unmapped reads
samtools view -b -q 30 -F 4 ChIP_dedup.bam > ChIP_filtered.bam
samtools view -b -q 30 -F 4 Input_dedup.bam > Input_filtered.bam

# For paired-end, filter proper pairs
samtools view -b -f 2 -q 30 ChIP_dedup.bam > ChIP_filtered.bam
```

#### 4. ChIP-seq Specific QC
```bash
# Cross-correlation analysis (phantompeakqualtools)
Rscript run_spp.R -c=ChIP_filtered.bam -savp=ChIP_qc.pdf -out=ChIP_metrics.txt

# Fraction of Reads in Peaks (FRiP)
# (Run after peak calling)

# Fingerprint plot (deepTools)
plotFingerprint -b ChIP_filtered.bam Input_filtered.bam \
  --labels ChIP Input -o fingerprint.png
```

#### 5. Peak Calling
```bash
# MACS2 for narrow peaks (transcription factors)
macs2 callpeak -t ChIP_filtered.bam -c Input_filtered.bam \
  -f BAM -g hs -n sample_TF --outdir peaks/ \
  -q 0.05 --call-summits

# MACS2 for broad peaks (histone marks)
macs2 callpeak -t ChIP_H3K27me3.bam -c Input_filtered.bam \
  -f BAM -g hs -n sample_H3K27me3 --outdir peaks/ \
  --broad --broad-cutoff 0.1

# SICER for very broad domains
sicer -t ChIP_filtered.bam -c Input_filtered.bam \
  -s hg38 -w 200 -g 600 -o peaks/
```

#### 6. Downstream Analysis
```bash
# Annotate peaks to nearest genes
bedtools closest -a peaks/sample_TF_peaks.narrowPeak \
  -b genes.bed -d > annotated_peaks.txt

# Motif discovery
findMotifsGenome.pl peaks/sample_TF_summits.bed hg38 motif_output/ \
  -size 200 -mask

# Differential binding (DiffBind in R)
# Create sample sheet and run in R
```

### Practical Tips

**Experimental Design:**
- Always include Input/IgG control
- Use biological replicates (2-3 minimum)
- Single-end 50bp sufficient for most TFs
- Paired-end 100bp better for histone marks

**Sequencing Depth:**
- Transcription factors: 20-40M reads
- Histone marks: 40-60M reads
- Input control: 20-30M reads

**Pros:**
- Identifies genome-wide binding sites
- Reveals regulatory elements
- Can study histone modifications
- Integrates with other epigenomics data
- Identifies motifs in bound regions

**Cons:**
- Requires high-quality antibodies
- Input control critical but often overlooked
- Repetitive regions challenging
- Cannot distinguish direct vs indirect binding
- Requires 10⁶-10⁷ cells per sample
- Peak calling parameters affect results significantly

### Example Use Case
**Transcription factor study:** ChIP-seq for CTCF in normal vs cancer cells revealed loss of binding at TAD boundaries in cancer, correlating with aberrant gene expression. Integration with Hi-C data showed disrupted 3D genome organization. MACS2 identified 45,000 peaks in normal cells but only 32,000 in cancer cells, with differential peaks enriched near oncogenes.

---

## ATAC-seq

### Pipeline Overview
ATAC-seq maps open chromatin regions genome-wide using hyperactive Tn5 transposase. Shares similarities with ChIP-seq but focuses on chromatin accessibility rather than protein binding.

### Workflow Steps

#### 1. Quality Control
```bash
# FastQC
fastqc ATAC_sample_R1.fastq.gz ATAC_sample_R2.fastq.gz

# Adapter trimming (remove Tn5 adapters)
trim_galore --paired --nextera \
  ATAC_sample_R1.fastq.gz ATAC_sample_R2.fastq.gz
```

#### 2. Alignment
```bash
# Bowtie2 with paired-end mode
bowtie2 -p 8 --very-sensitive -X 2000 \
  -x reference_index \
  -1 ATAC_R1_val_1.fq.gz -2 ATAC_R2_val_2.fq.gz | \
  samtools sort -@ 4 -o ATAC_sorted.bam

samtools index ATAC_sorted.bam
```

#### 3. ATAC-seq Specific Post-processing
```bash
# Remove duplicates
picard MarkDuplicates -I ATAC_sorted.bam -O ATAC_dedup.bam \
  -M metrics.txt -REMOVE_DUPLICATES true

# Remove mitochondrial reads (high in ATAC-seq)
samtools idxstats ATAC_dedup.bam | cut -f 1 | grep -v chrM > nuclear_chroms.txt
samtools view -b ATAC_dedup.bam $(cat nuclear_chroms.txt) > ATAC_nuclear.bam

# Filter for high-quality, properly paired reads
samtools view -b -f 2 -q 30 ATAC_nuclear.bam > ATAC_filtered.bam

# ATAC-seq specific: Adjust for Tn5 insertion (shift +4/-5 bp)
alignmentSieve --ATACshift --bam ATAC_filtered.bam -o ATAC_shifted.bam

# Size selection (nucleosome-free fragments)
# Keep fragments <100bp (nucleosome-free)
samtools view -h ATAC_shifted.bam | \
  awk '$9 < 100 && $9 > -100 || $1 ~ /^@/' | \
  samtools view -bS - > ATAC_NFR.bam
```

#### 4. ATAC-seq QC Metrics
```bash
# TSS enrichment score
computeMatrix reference-point --referencePoint TSS \
  -b 2000 -a 2000 -R genes.bed -S ATAC_shifted.bw -o matrix_TSS.gz
plotProfile -m matrix_TSS.gz -o TSS_profile.png

# Fragment size distribution
picard CollectInsertSizeMetrics -I ATAC_shifted.bam -O insert_sizes.txt \
  -H insert_sizes.pdf

# Expected pattern: peak at ~50bp (nucleosome-free), ~200bp (mono-nucleosome)
```

#### 5. Peak Calling
```bash
# MACS2 for ATAC-seq (no control, shift correction)
macs2 callpeak -t ATAC_shifted.bam -f BAMPE -g hs \
  -n ATAC_sample --outdir peaks/ --shift -75 --extsize 150 \
  --nomodel --call-summits --nolambda --keep-dup all -q 0.05

# Alternative: genrich (ATAC-specific)
Genrich -t ATAC_shifted.bam -o peaks.narrowPeak -j -y -r -e chrM
```

#### 6. Footprinting Analysis
```bash
# HINT-ATAC for transcription factor footprints
rgt-hint footprinting --atac-seq --paired-end \
  --organism=hg38 --output-location=footprints/ \
  ATAC_shifted.bam peaks/ATAC_sample_peaks.narrowPeak

# Identify bound motifs
rgt-motifanalysis matching --organism=hg38 \
  --input-files footprints/sample.bed \
  --output-location motif_matching/
```

### Practical Tips

**Library Quality Indicators:**
- TSS enrichment score >7 (good quality)
- Fragment size distribution shows nucleosomal pattern
- <10% mitochondrial reads (indicates good nuclei isolation)
- >40% reads in peaks (FRiP score)

**Sequencing Depth:**
- Minimum 25-50M reads per sample
- Cell type comparisons: 50-75M reads
- Footprinting analysis: 100M+ reads

**Pros:**
- Requires fewer cells (500-50,000 vs millions for ChIP)
- Fast protocol (1 day vs 3-4 days for ChIP)
- No antibodies required
- Identifies regulatory elements genome-wide
- Can infer TF binding via footprinting
- Distinguishes nucleosome positioning

**Cons:**
- High mitochondrial contamination common
- Requires fresh/frozen samples (not FFPE)
- Tn5 bias in some genomic regions
- Cannot identify which protein binds
- Lower specificity than ChIP-seq for individual factors
- Challenging in heterogeneous tissues

### Example Use Case
**Developmental biology:** ATAC-seq across five stages of cardiomyocyte differentiation revealed dynamic chromatin accessibility changes. Early stages showed open chromatin at pluripotency factors (OCT4, NANOG), which closed during differentiation. Cardiac-specific regions (GATA4, NKX2-5) became accessible at day 10-15. Footprinting analysis identified that TBX5 binding preceded cardiac gene expression, suggesting it pioneers chromatin opening.

---

## Bisulfite-seq

### Pipeline Overview
Bisulfite sequencing maps DNA methylation at single-base resolution by converting unmethylated cytosines to uracil (read as thymine), while methylated cytosines remain unchanged. Requires specialized alignment tools.

### Workflow Steps

#### 1. Quality Control
```bash
# FastQC (expect unusual base composition due to C->T conversion)
fastqc BS_sample_R1.fastq.gz BS_sample_R2.fastq.gz

# Adapter trimming with Trim Galore (bisulfite mode)
trim_galore --paired --rrbs \
  BS_sample_R1.fastq.gz BS_sample_R2.fastq.gz
```

#### 2. Bisulfite-Specific Alignment
```bash
# Bismark (most popular bisulfite aligner)
# First, prepare reference
bismark_genome_preparation --path_to_aligner /path/to/bowtie2 reference_genome/

# Align with Bismark
bismark --genome reference_genome/ \
  -1 BS_R1_val_1.fq.gz -2 BS_R2_val_2.fq.gz \
  --multicore 4 --output_dir bismark_output/

# Alternative: BSMAP
bsmap -a BS_R1.fq.gz -b BS_R2.fq.gz \
  -d reference.fa -o BS_aligned.bam -p 8 -v 5
```

#### 3. Post-processing
```bash
# Remove duplicates (critical in bisulfite-seq)
deduplicate_bismark --paired --bam bismark_output/sample_pe.bam

# Sort BAM
samtools sort bismark_output/sample_pe.deduplicated.bam \
  -o sample_sorted.bam
```

#### 4. Methylation Extraction
```bash
# Extract methylation calls
bismark_methylation_extractor --paired-end --comprehensive \
  --merge_non_CpG --bedGraph --counts \
  --cytosine_report --genome_folder reference_genome/ \
  sample_sorted.bam

# Output files:
# - CpG_context_sample.txt (CpG methylation)
# - CHG_context_sample.txt (CHG methylation)
# - CHH_context_sample.txt (CHH methylation)
# - sample.bedGraph (genome browser visualization)
# - sample.bismark.cov.gz (coverage file)
```

#### 5. Bisulfite-Specific QC
```bash
# Conversion rate check (using lambda phage spike-in or CHH context)
# CHH methylation should be <2% (indicates complete conversion)
awk '$4+$5 > 0 {meth+=$4; unmeth+=$5} END {print "CHH methylation:", meth/(meth+unmeth)*100"%"}' CHH_context_sample.txt

# Coverage distribution
awk '{print $5+$6}' sample.bismark.cov.gz | \
  sort | uniq -c > coverage_distribution.txt

# M-bias plot (position-specific methylation bias)
bismark2report --alignment_report sample_PE_report.txt \
  --mbias_report sample.M-bias.txt
```

#### 6. Differential Methylation Analysis
```bash
# DSS (in R) for differentially methylated regions (DMRs)
library(DSS)

# Load bismark coverage files
bsseq_data <- read.bismark(c("control.cov", "treatment.cov"),
                           sampleNames = c("Control", "Treatment"),
                           rmZeroCov = TRUE)

# Call DMRs
dmlTest <- DMLtest(bsseq_data, group1 = "Control", group2 = "Treatment")
dmrs <- callDMR(dmlTest, p.threshold = 0.05, minlen = 50, minCG = 3)

# methylKit alternative
library(methylKit)
file.list <- list("control.cov", "treatment.cov")
myobj <- methRead(file.list, sample.id = c("C", "T"),
                  assembly = "hg38", treatment = c(0,1),
                  pipeline = "bismarkCoverage")

# Get differentially methylated bases
myDiff <- calculateDiffMeth(myobj)
myDiff.hyper <- getMethylDiff(myDiff, difference = 25, qvalue = 0.01, type = "hyper")

# Annotate DMRs
annotate.WithGenicParts(myDiff.hyper, gene.obj)
```

#### 7. Visualization
```bash
# Create bigWig for genome browser
LC_COLLATE=C sort -k1,1 -k2,2n sample.bedGraph > sample_sorted.bedGraph
bedGraphToBigWig sample_sorted.bedGraph chrom.sizes sample.bw

# Methylation heatmap in R
library(methylKit)
getCorrelation(myobj, plot = TRUE)
```

### Practical Tips

**Library Preparation Considerations:**
- WGBS (Whole Genome): 28 CpGs, post-bisulfite adapter tagging (PBAT) better for low input
- RRBS (Reduced Representation): MspI digestion enriches CpG islands, costs 1/10th of WGBS
- Targeted bisulfite-seq: capture specific regions after bisulfite conversion

**Quality Metrics:**
- Bisulfite conversion rate >99% (check CHH methylation)
- Mapping efficiency 40-70% (lower than regular sequencing)
- Duplicate rate should be <20%
- M-bias: check for position-dependent methylation artifacts

**Coverage Requirements:**
- WGBS: 10-30X coverage (higher for allele-specific methylation)
- RRBS: 20-50X coverage
- Minimum 10X coverage required for reliable methylation calls
- At least 5 CpGs per DMR

**Pros:**
- Single-base resolution methylation mapping
- Quantitative (% methylation at each CpG)
- Gold standard for DNA methylation
- Detects 5mC and distinguishes from 5hmC (with additional steps)
- Can detect non-CpG methylation

**Cons:**
- DNA degradation from bisulfite treatment (~90% loss)
- Lower mapping efficiency due to reduced complexity
- Cannot distinguish 5mC from 5hmC without oxidation step
- Expensive for whole genome (~$1000-2000 per sample)
- Requires high input DNA (>1μg for WGBS)
- Complex bioinformatics (specialized aligners required)
- Incomplete conversion leads to false positives

### Example Use Case
**Cancer epigenetics:** WGBS of colon cancer vs normal tissue revealed widespread hypomethylation in intergenic regions and hypermethylation of tumor suppressor promoters. Specific findings:
- MLH1 promoter showed 85% methylation in tumor vs 5% in normal
- Global methylation decreased from 72% to 58%
- Identified 2,847 differentially methylated regions (DMRs)
- DMRs at enhancers correlated with altered gene expression in RNA-seq
- RRBS validation in 50 additional samples confirmed findings at 1/10th cost

---

## scRNA-seq

### Pipeline Overview
Single-cell RNA-seq profiles transcriptomes of individual cells, requiring specialized handling of cellular barcodes and UMIs (Unique Molecular Identifiers) to distinguish cells and correct for PCR amplification bias.

### Workflow Steps

#### 1. Demultiplexing and Quality Control
```bash
# Cell Ranger (10x Genomics platform)
cellranger count --id=sample_run \
  --transcriptome=/path/to/refdata-gex-GRCh38 \
  --fastqs=/path/to/fastqs \
  --sample=sample_name \
  --expect-cells=5000 \
  --localcores=16

# STARsolo (alternative, open-source)
STAR --runMode genomeGenerate \
  --genomeDir genome_index \
  --genomeFastaFiles reference.fa \
  --sjdbGTFfile annotations.gtf

STAR --genomeDir genome_index \
  --readFilesIn sample_R2.fastq.gz sample_R1.fastq.gz \
  --readFilesCommand zcat \
  --soloType CB_UMI_Simple \
  --soloCBwhitelist 3M-february-2018.txt \
  --soloUMIlen 12 \
  --soloCBstart 1 --soloCBlen 16 \
  --outFileNamePrefix sample_ \
  --outSAMtype BAM SortedByCoordinate \
  --soloFeatures Gene GeneFull \
  --soloOutFileNames Solo.out/ genes.tsv barcodes.tsv matrix.mtx
```

#### 2. Cell Barcode Processing
```bash
# Cell Ranger automatically handles:
# - Barcode error correction
# - Cell calling (distinguishing real cells from empty droplets)
# - UMI deduplication

# For STARsolo, check output
# Solo.out/Gene/filtered/ contains cell-filtered matrix
# Solo.out/Gene/raw/ contains all barcodes

# Manually inspect knee plot for cell calling
# (already done by Cell Ranger)
```

#### 3. Quality Control in R (Seurat)
```R
library(Seurat)
library(ggplot2)

# Load 10x data
sample.data <- Read10X(data.dir = "filtered_feature_bc_matrix/")
sample <- CreateSeuratObject(counts = sample.data, 
                             project = "scRNA", 
                             min.cells = 3, 
                             min.features = 200)

# Calculate QC metrics
sample[["percent.mt"]] <- PercentageFeatureSet(sample, pattern = "^MT-")

# Visualize QC metrics
VlnPlot(sample, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
        ncol = 3)

# Filter low-quality cells
sample <- subset(sample, subset = nFeature_RNA > 200 & 
                                  nFeature_RNA < 5000 & 
                                  percent.mt < 15)

# Typical filters:
# - nFeature_RNA: 200-5000 genes per cell (removes empty droplets and doublets)
# - percent.mt: <15% (high indicates dying cells)
# - nCount_RNA: >500 UMIs (ensures sufficient sequencing depth)
```

#### 4. Normalization and Feature Selection
```R
# Normalize data (accounts for differences in sequencing depth)
sample <- NormalizeData(sample, normalization.method = "LogNormalize", 
                        scale.factor = 10000)

# Identify highly variable features
sample <- FindVariableFeatures(sample, selection.method = "vst", 
                               nfeatures = 2000)

# Scale data (regress out unwanted variation)
sample <- ScaleData(sample, vars.to.regress = c("percent.mt", "nCount_RNA"))
```

#### 5. Dimensionality Reduction
```R
# PCA
sample <- RunPCA(sample, features = VariableFeatures(object = sample))

# Determine number of PCs
ElbowPlot(sample, ndims = 50)

# UMAP for visualization
sample <- RunUMAP(sample, dims = 1:30)

# Alternative: tSNE
sample <- RunTSNE(sample, dims = 1:30)
```

#### 6. Clustering
```R
# Find neighbors and clusters
sample <- FindNeighbors(sample, dims = 1:30)
sample <- FindClusters(sample, resolution = 0.5)

# Visualize clusters
DimPlot(sample, reduction = "umap", label = TRUE)

# Resolution parameter affects granularity:
# - 0.4-0.6: fewer, broader clusters
# - 0.8-1.2: more, finer clusters
```

#### 7. Cell Type Annotation
```R
# Find marker genes for each cluster
cluster.markers <- FindAllMarkers(sample, only.pos = TRUE, 
                                  min.pct = 0.25, 
                                  logfc.threshold = 0.25)

# Top markers per cluster
top10 <- cluster.markers %>% 
  group_by(cluster) %>% 
  top_n(n = 10, wt = avg_log2FC)

# Heatmap of marker genes
DoHeatmap(sample, features = top10$gene)

# Annotate based on known markers
# Example: T cells (CD3D, CD3E), B cells (CD79A, MS4A1), 
#          Monocytes (CD14, FCGR3A)

# Automatic annotation with SingleR
library(SingleR)
ref <- celldex::HumanPrimaryCellAtlasData()
predictions <- SingleR(test = GetAssayData(sample), 
                       ref = ref, 
                       labels = ref$label.main)
sample$celltype <- predictions$labels
```

#### 8. Differential Expression Analysis
```R
# Compare two conditions (e.g., disease vs control)
Idents(sample) <- "condition"
de.genes <- FindMarkers(sample, ident.1 = "disease", ident.2 = "control")

# Pseudobulk analysis (more robust for condition comparisons)
library(DESeq2)
pseudobulk <- AggregateExpression(sample, 
                                  group.by = c("celltype", "condition"),
                                  return.seurat = FALSE)

# Export for DESeq2
# ... standard DESeq2 workflow
```

#### 9. Trajectory Analysis (Optional)
```R
# Monocle3 for pseudotime/trajectory
library(monocle3)

cds <- as.cell_data_set(sample)
cds <- cluster_cells(cds)
cds <- learn_graph(cds)
cds <- order_cells(cds)

plot_cells(cds, color_cells_by = "pseudotime")
```

### Practical Tips

**Platform Considerations:**
- 10x Genomics: Most common, 3' or 5' gene expression, 5,000-10,000 cells/run
- Drop-seq/inDrop: Open-source alternatives
- Smart-seq2/3: Full-length transcripts, fewer cells but deeper per cell
- CITE-seq: Combines RNA + surface protein (antibodies)

**Quality Metrics:**
- Median genes per cell: 1,000-3,000 (cell type dependent)
- Median UMIs per cell: 2,000-10,000
- Doublet rate: <5% (use DoubletFinder or Scrublet)
- Ambient RNA: Check with SoupX or CellBender

**Sequencing Depth:**
- Standard scRNA-seq: 20,000-50,000 reads per cell
- Deep profiling: 100,000+ reads per cell
- Total: 200-500M reads for 5,000-10,000 cells

**Pros:**
- Resolves cellular heterogeneity
- Identifies rare cell populations
- Reveals cell state transitions
- No need for cell sorting/isolation
- Discovers novel cell types
- Characterizes developmental trajectories

**Cons:**
- High cost ($500-2000 per sample)
- Captures only ~10-20% of transcriptome per cell
- 3' bias (10x) misses isoforms
- Doublets confound analysis
- Batch effects between runs
- Requires fresh cells (not FFPE)
- Complex data analysis
- Dropout events (zero inflation)

### Example Use Case
**Tumor microenvironment:** scRNA-seq of melanoma tumor (8,500 cells) revealed:
- 12 distinct cell populations including malignant cells, T cells, B cells, macrophages, fibroblasts, endothelial cells
- CD8+ T cells split into exhausted (high PD-1, LAG3) vs cytotoxic (high GZMB, PRF1) subsets
- Tumor cells showed heterogeneous expression of MITF (melanocyte lineage marker)
- Trajectory analysis revealed malignant cells transitioning between proliferative and invasive states
- Ligand-receptor analysis (CellPhoneDB) showed tumor cells suppressing T cells via PD-L1/PD-1 interaction
- Integrated with bulk RNA-seq from 50 patients to validate that exhausted T cell signature correlated with poor response to immunotherapy

---

## Metagenomics

### Pipeline Overview
Metagenomic sequencing analyzes DNA from environmental samples containing mixed species. Unlike other NGS types, it focuses on taxonomic classification and functional profiling rather than alignment to a single reference genome.

### Workflow Steps

#### 1. Quality Control
```bash
# FastQC
fastqc sample_R1.fastq.gz sample_R2.fastq.gz

# Remove adapters and low-quality reads
trimmomatic PE sample_R1.fastq.gz sample_R2.fastq.gz \
  sample_R1_paired.fq.gz sample_R1_unpaired.fq.gz \
  sample_R2_paired.fq.gz sample_R2_unpaired.fq.gz \
  ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:20 MINLEN:50

# Remove host contamination (e.g., human reads from gut microbiome)
bowtie2 -p 8 -x human_genome_index \
  -1 sample_R1_paired.fq.gz -2 sample_R2_paired.fq.gz \
  --un-conc-gz sample_nonhuman_R%.fq.gz \
  -S host_aligned.sam
```

#### 2. Taxonomic Classification
```bash
# Kraken2 (fast, k-mer based)
kraken2 --db kraken2_db --threads 16 --paired \
  --report sample_kraken_report.txt \
  --output sample_kraken_output.txt \
  sample_nonhuman_R1.fq.gz sample_nonhuman_R2.fq.gz

# Bracken (abundance estimation from Kraken2)
bracken -d kraken2_db -i sample_kraken_report.txt \
  -o sample_bracken_species.txt -r 150 -l S -t 10

# MetaPhlAn4 (marker gene based, more accurate)
metaphlan sample_nonhuman_R1.fq.gz,sample_nonhuman_R2.fq.gz \
  --input_type fastq --nproc 16 --bowtie2out sample.bowtie2.bz2 \
  -o sample_metaphlan_profile.txt

# Kaiju (protein-level classification for divergent species)
kaiju -t nodes.dmp -f kaiju_db.fmi -i sample_nonhuman_R1.fq.gz \
  -j sample_nonhuman_R2.fq.gz -o sample_kaiju.out -z 16

kaiju2table -t nodes.dmp -n names.dmp -r species \
  -o sample_kaiju_summary.txt sample_kaiju.out
```

#### 3. Assembly-Based Analysis
```bash
# MEGAHIT (fast, memory-efficient assembler)
megahit -1 sample_nonhuman_R1.fq.gz -2 sample_nonhuman_R2.fq.gz \
  -o assembly_output --min-contig-len 500 -t 16

# SPAdes (more accurate but slower)
metaspades.py -1 sample_nonhuman_R1.fq.gz -2 sample_nonhuman_R2.fq.gz \
  -o spades_assembly -t 16 -m 100

# Assess assembly quality
quast.py assembly_output/final.contigs.fa -o quast_results

# Binning contigs into MAGs (Metagenome-Assembled Genomes)
# Map reads back to assembly
bowtie2-build assembly_output/final.contigs.fa contigs_index
bowtie2 -p 16 -x contigs_index \
  -1 sample_nonhuman_R1.fq.gz -2 sample_nonhuman_R2.fq.gz | \
  samtools sort -@ 8 -o mapped_sorted.bam

# MetaBAT2 binning
jgi_summarize_bam_contig_depths --outputDepth depth.txt mapped_sorted.bam
metabat2 -i assembly_output/final.contigs.fa -a depth.txt -o bins/bin -m 1500

# Alternative: MaxBin2
run_MaxBin.pl -contig assembly_output/final.contigs.fa \
  -reads sample_nonhuman_R1.fq.gz -reads2 sample_nonhuman_R2.fq.gz \
  -out maxbin_bins/bin -thread 16

# Check bin quality with CheckM
checkm lineage_wf -t 16 -x fa bins/ checkm_output/
checkm qa checkm_output/lineage.ms checkm_output/ -o 2 -f bin_quality.txt
```

#### 4. Functional Annotation
```bash
# Annotate MAGs with Prokka
for bin in bins/*.fa; do
  prokka --outdir annotations/$(basename $bin .fa) \
    --prefix $(basename $bin .fa) --metagenome $bin
done

# HUMAnN3 for functional profiling (metabolic pathways)
humann --input sample_nonhuman_R1.fq.gz \
  --output humann_output --threads 16

# Outputs:
# - Gene families (UniRef90)
# - Pathways (MetaCyc)
# - Pathway coverage and abundance

# DIAMOND for fast protein annotation
diamond blastx -d nr_db.dmnd -q assembly_output/final.contigs.fa \
  -o diamond_results.txt -f 6 -p 16 --top 5

# eggNOG-mapper for functional annotation
emapper.py -i assembly_output/final.contigs.fa \
  --itype CDS -o eggnog_results --cpu 16
```

#### 5. Visualization and Statistics
```R
# Krona visualization
ktImportTaxonomy -o krona.html sample_kraken_output.txt

# Alpha diversity in R
library(vegan)
library(phyloseq)

# Load abundance table
otu_table <- read.table("sample_bracken_species.txt", header=TRUE, row.names=1)

# Calculate diversity indices
shannon <- diversity(otu_table, index = "shannon")
simpson <- diversity(otu_table, index = "simpson")
richness <- specnumber(otu_table)

# Beta diversity (comparing multiple samples)
# Create phyloseq object
physeq <- phyloseq(otu_table(otu_mat, taxa_are_rows=TRUE),
                   sample_data(metadata))

# Ordination
ord <- ordinate(physeq, method="PCoA", distance="bray")
plot_ordination(physeq, ord, color="condition")

# Differential abundance with DESeq2
library(DESeq2)
diagdds <- phyloseq_to_deseq2(physeq, ~ condition)
diagdds <- DESeq(diagdds)
res <- results(diagdds)
```

#### 6. Strain-Level Analysis
```bash
# StrainPhlAn for strain tracking
sample2markers.py -i sample.bowtie2.bz2 \
  -o markers/ -n 8

# Extract markers for species of interest
extract_markers.py -c s__Escherichia_coli \
  -o reference_markers/

# Build phylogenetic tree
strainphlan -s markers/*.pkl \
  -m reference_markers/ \
  -o strainphlan_output/ \
  -n 8 --phylophlan_mode accurate
```

### Practical Tips

**Sample Types:**
- Gut microbiome: Remove human DNA (>90% of reads can be host)
- Soil/water: High diversity, deeper sequencing needed
- Mock communities: Use as positive controls

**Sequencing Strategy:**
- Shotgun metagenomics: 10-50M reads per sample for profiling
- Deep sequencing: 100M+ reads for assembly and MAG recovery
- Paired-end 150bp standard
- Consider long-read sequencing (PacBio/Nanopore) for better assembly

**Database Considerations:**
- Kraken2 DB size: 8GB (miniKraken) to 100GB+ (full RefSeq)
- Update databases regularly (microbial genomes constantly added)
- Custom databases for specific environments

**Pros:**
- Culture-independent (detects unculturable organisms)
- Discovers novel species
- Provides functional capacity of community
- No prior knowledge of species required
- Can reconstruct complete genomes (MAGs)
- Detects viruses, plasmids, mobile elements

**Cons:**
- Computationally intensive
- Reference database dependent (novel organisms missed)
- Cannot distinguish live vs dead organisms
- DNA extraction bias affects results
- Expensive for deep sequencing ($300-1000 per sample)
- Chimeric sequences in assembly
- Horizontal gene transfer complicates taxonomy
- Relative abundance, not absolute counts

### Example Use Case
**Gut microbiome and disease:** Metagenomic analysis of IBD patients (n=50) vs healthy controls (n=50) revealed:

**Taxonomic findings:**
- Reduced alpha diversity in IBD (Shannon index: 2.8 vs 3.9)
- Decreased *Faecalibacterium prausnitzii* (butyrate producer)
- Increased *Escherichia coli* and *Ruminococcus gnavus*
- Novel *Clostridiales* species detected in 20% of IBD patients

**Functional findings:**
- HUMAnN3 showed reduced butyrate synthesis pathways
- Increased oxidative stress response genes
- Enrichment of adhesion/invasion genes in *E. coli* strains

**MAG analysis:**
- Recovered 47 high-quality MAGs (>90% complete, <5% contamination)
- One novel *Alistipes* species found exclusively in IBD
- Strain-level analysis showed patient-specific *F. prausnitzii* strains

**Integration:**
- Correlated microbiome composition with metabolomics (SCFA levels)
- Machine learning classifier (Random Forest) using top 20 species achieved 85% accuracy for IBD diagnosis

---

## Quick Comparison Table

| Feature | WGS | WES | RNA-seq | Targeted | ChIP-seq | ATAC-seq | Bisulfite-seq | scRNA-seq | Metagenomics |
|---------|-----|-----|---------|----------|----------|----------|---------------|-----------|--------------|
| **Primary aligner** | BWA-MEM | BWA-MEM | STAR/HISAT2 | BWA/Bowtie2 | Bowtie2 | Bowtie2 | Bismark | Cell Ranger/STARsolo | Kraken2/MetaPhlAn |
| **Remove duplicates?** | Yes | Yes | **No** | Yes | **Yes** | **Yes** | **Yes** | Auto (UMI) | Depends |
| **Typical coverage** | 30-40X | 100X | 30M reads | 1000-5000X | 20-40M reads | 50M reads | 10-30X | 20-50K reads/cell | 10-100M reads |
| **Data size (per sample)** | 100-200 GB | 8-12 GB | 10-20 GB | 1-5 GB | 5-10 GB | 5-10 GB | 80-150 GB | 50-100 GB | 20-50 GB |
| **Cost per sample** | $600-1000 | $200-400 | $150-300 | $50-150 | $300-500 | $200-400 | $1000-2000 | $500-2000 | $300-1000 |
| **Key QC metric** | Coverage uniformity | On-target % | Mapping % | On-target % | FRiP, cross-correlation | TSS enrichment, FRiP | Conversion rate | Genes/cell, %MT | Host contamination |
| **Output format** | VCF (variants) | VCF (variants) | Counts matrix | VCF (variants) | BED (peaks) | BED (peaks) | Methylation calls | Counts matrix | Taxonomy table |
| **Downstream analysis** | Variant annotation | Variant annotation | DESeq2/EdgeR | Variant annotation | Motif discovery | Footprinting | DMR calling | Seurat/Scanpy | Diversity analysis |
| **Special considerations** | Large SVs, CNVs | BED file critical | Strand specificity | Ultra-high depth | Input control required | Mitochondrial reads | DNA degradation | Doublet detection | Database dependency |
| **Cell/input requirement** | 1-10 μg DNA | 50-100 ng DNA | 0.1-1 μg RNA | 10-50 ng DNA | 10⁶-10⁷ cells | 50,000 cells | 1-5 μg DNA | 500-10,000 cells | Variable |

---

## General Pipeline Principles Across All Types

### 1. Quality Control is Universal
Every pipeline starts with:
- FastQC assessment
- Adapter trimming
- Quality filtering
- Post-alignment QC specific to data type

### 2. Alignment Strategy Depends on Biology
- **DNA variants (WGS/WES/Targeted):** Use BWA-MEM for accurate local alignment
- **Transcripts (RNA-seq):** Require splice-aware aligners (STAR, HISAT2)
- **Epigenetics (ChIP/ATAC):** Bowtie2 sufficient, focus on peak calling
- **Modified DNA (Bisulfite-seq):** Specialized aligners handle C→T conversion
- **Single-cell:** Barcode/UMI handling before alignment
- **Mixed species:** Classification, not alignment

### 3. Post-Processing Varies
- **Duplicate removal:** Critical for DNA sequencing, harmful for RNA-seq
- **Base quality recalibration:** Important for variant calling, unnecessary for expression/peaks
- **Target restriction:** Apply BED files for WES/targeted to limit false positives

### 4. Output Types Differ
- **Variants:** VCF files (WGS, WES, Targeted, RNA-seq for RNA editing)
- **Quantification:** Count matrices (RNA-seq, scRNA-seq)
- **Regions:** BED files (ChIP-seq, ATAC-seq)
- **Methylation:** Custom formats (Bisulfite-seq)
- **Taxonomic profiles:** Abundance tables (Metagenomics)

### 5. Storage and Compute Requirements
Scale exponentially:
- **Small:** Targeted panels (1-5 GB, hours)
- **Medium:** WES, RNA-seq (10-20 GB, hours to day)
- **Large:** WGS (100-200 GB, days)
- **Very large:** scRNA-seq, WGBS, metagenomics (50-200 GB, days to weeks)

---

## Choosing the Right Sequencing Type

**For germline variant discovery:**
- Suspected Mendelian disease: Start with **WES**, escalate to **WGS** if negative
- Pharmacogenomics: **Targeted panel** (ADME genes)
- Population genetics: Low-coverage **WGS** (10-15X)

**For cancer genomics:**
- Hotspot mutations: **Targeted panel** (fast, sensitive)
- Comprehensive profiling: **WGS** (SNVs, CNVs, SVs, mutational signatures)
- Transcriptome changes: **RNA-seq** (fusion genes, expression)
- Liquid biopsy: **Ultra-deep targeted** (ctDNA detection)

**For gene expression:**
- Bulk tissue: **RNA-seq** (DEG analysis)
- Cell heterogeneity: **scRNA-seq** (cell types, states)
- Spatial context: Spatial transcriptomics (not covered here)

**For epigenetics:**
- Specific protein binding: **ChIP-seq**
- Open chromatin landscape: **ATAC-seq**
- DNA methylation: **Bisulfite-seq** (WGBS or RRBS)

**For microbiome:**
- Community composition: **16S rRNA sequencing** (not covered, targeted amplicon)
- Functional capacity: **Shotgun metagenomics**
- Strain resolution: Deep **shotgun metagenomics** + assembly

---

## Key Takeaways

1. **No universal pipeline:** Each sequencing type requires specialized tools and parameters

2. **Alignment tools matter:** Splice awareness, bisulfite handling, and barcode processing are type-specific

3. **Post-processing is not one-size-fits-all:** Duplicate removal helps DNA but hurts RNA

4. **Quality metrics differ:** Coverage uniformity (WGS), on-target % (WES), TSS enrichment (ATAC-seq), conversion rate (Bisulfite-seq)

5. **Cost-accuracy tradeoffs:** Targeted panels are cheap but limited; WGS is comprehensive but expensive

6. **Integration enhances insights:** Combine WGS + RNA-seq, or ATAC-seq + ChIP-seq for mechanistic understanding

7. **Stay updated:** Tools evolve rapidly; always check for latest versions and best practices in literature
