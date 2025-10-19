# NGS Analysis Pipeline Documentation

## Complete Guide for Prokaryotic and Eukaryotic Genome Analysis

---

## Table of Contents
1. [Quality Control](#1-quality-control)
2. [Read Trimming and Filtering](#2-read-trimming-and-filtering)
3. [Alignment to Reference Genome](#3-alignment-to-reference-genome)
4. [Post-Alignment Processing](#4-post-alignment-processing)
5. [Variant Calling](#5-variant-calling)
6. [Variant Filtering](#6-variant-filtering)
7. [Variant Annotation](#7-variant-annotation)
8. [Downstream Analysis](#8-downstream-analysis)

---

## 1. Quality Control

Quality control is the first critical step to assess raw sequencing data quality before proceeding with analysis.

### Popular Tools

#### FastQC
**Purpose:** Comprehensive quality assessment of raw sequencing reads

**Input Files:**
- Raw FASTQ files (`.fastq`, `.fq`, `.fastq.gz`)

**Output Files:**
- HTML report (`.html`)
- Quality metrics file (`.zip`)

**Sample Command:**
```bash
# Single-end reads
fastqc sample_R1.fastq.gz -o qc_output/

# Paired-end reads
fastqc sample_R1.fastq.gz sample_R2.fastq.gz -o qc_output/ -t 4
```

**Pros:**
- Easy to use and interpret
- Generates visual HTML reports
- Fast processing
- Industry standard

**Cons:**
- Limited to quality assessment only
- No direct remediation features
- Can flag issues that aren't problematic for all analyses

**Best for:** Quick quality assessment for all organism types

---

#### MultiQC
**Purpose:** Aggregate multiple QC reports into single summary

**Input Files:**
- Output directories from various QC tools (FastQC, etc.)

**Output Files:**
- Consolidated HTML report
- Data directory with metrics

**Sample Command:**
```bash
multiqc qc_output/ -o multiqc_results/
```

**Pros:**
- Consolidates multiple samples
- Comparative analysis across samples
- Supports many bioinformatics tools

**Cons:**
- Requires other tools' outputs
- Can be overwhelming for small projects

**Best for:** Projects with multiple samples

---

#### Qualimap
**Purpose:** Quality control of alignment and sequencing data

**Input Files:**
- BAM files (post-alignment)
- BED files (optional, for targeted regions)

**Output Files:**
- HTML report
- PDF report
- Raw statistics

**Sample Command:**
```bash
qualimap bamqc -bam aligned_reads.bam -outdir qualimap_output/

# For RNA-seq
qualimap rnaseq -bam aligned_reads.bam -gtf genes.gtf -outdir qualimap_output/
```

**Pros:**
- Detailed alignment statistics
- Coverage analysis
- Works on aligned data

**Cons:**
- Slower than FastQC
- Requires aligned reads

**Best for:** Post-alignment quality assessment

---

### Organism-Specific Considerations

**Prokaryotes:**
- Typically simpler QC due to smaller genome size
- Look for contamination from host DNA
- Check for plasmid sequences

**Eukaryotes:**
- More complex genomes require careful QC
- Assess for adapter contamination
- Check for RNA contamination in DNA-seq
- Consider repeat content

---

## 2. Read Trimming and Filtering

Remove low-quality bases, adapters, and contaminants to improve downstream analysis accuracy.

### Popular Tools

#### Trimmomatic
**Purpose:** Flexible trimming tool for Illumina sequence data

**Input Files:**
- Raw FASTQ files (`.fastq`, `.fastq.gz`)
- Adapter sequences file (`.fa`)

**Output Files:**
- Trimmed paired reads (R1 and R2)
- Unpaired reads (singletons)

**Sample Command:**
```bash
# Paired-end
trimmomatic PE -threads 4 \
  sample_R1.fastq.gz sample_R2.fastq.gz \
  sample_R1_paired.fastq.gz sample_R1_unpaired.fastq.gz \
  sample_R2_paired.fastq.gz sample_R2_unpaired.fastq.gz \
  ILLUMINACLIP:adapters.fa:2:30:10 \
  LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36

# Single-end
trimmomatic SE -threads 4 \
  sample.fastq.gz sample_trimmed.fastq.gz \
  ILLUMINACLIP:adapters.fa:2:30:10 \
  LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
```

**Pros:**
- Highly configurable
- Handles paired-end data well
- Fast processing
- Widely used and tested

**Cons:**
- Complex parameter syntax
- May be too aggressive with default settings
- Requires adapter file

**Best for:** Standard Illumina data for both prokaryotes and eukaryotes

---

#### Cutadapt
**Purpose:** Remove adapter sequences and quality trimming

**Input Files:**
- Raw FASTQ files

**Output Files:**
- Trimmed FASTQ files
- Trimming report

**Sample Command:**
```bash
# Paired-end with adapter removal
cutadapt -a AGATCGGAAGAGC -A AGATCGGAAGAGC \
  -o sample_R1_trimmed.fastq.gz -p sample_R2_trimmed.fastq.gz \
  -m 36 -q 20 -j 4 \
  sample_R1.fastq.gz sample_R2.fastq.gz

# Single-end
cutadapt -a AGATCGGAAGAGC -m 36 -q 20 -j 4 \
  -o sample_trimmed.fastq.gz sample.fastq.gz
```

**Pros:**
- Excellent adapter detection
- Simpler syntax than Trimmomatic
- Good documentation
- Automatic adapter detection option

**Cons:**
- Slower than some alternatives
- Less feature-rich for quality trimming

**Best for:** Adapter-heavy samples, automatic adapter detection

---

#### fastp
**Purpose:** All-in-one preprocessing tool

**Input Files:**
- Raw FASTQ files

**Output Files:**
- Trimmed FASTQ files
- HTML report
- JSON report

**Sample Command:**
```bash
# Paired-end
fastp -i sample_R1.fastq.gz -I sample_R2.fastq.gz \
  -o sample_R1_clean.fastq.gz -O sample_R2_clean.fastq.gz \
  -h fastp_report.html -j fastp_report.json \
  -w 4 --detect_adapter_for_pe

# Single-end
fastp -i sample.fastq.gz -o sample_clean.fastq.gz \
  -h fastp_report.html -j fastp_report.json \
  -w 4
```

**Pros:**
- Very fast (faster than Trimmomatic)
- Built-in QC reporting
- Automatic adapter detection
- All-in-one solution

**Cons:**
- Less control over specific parameters
- Relatively newer (less established)

**Best for:** Quick preprocessing with automatic settings, both organisms

---

### Organism-Specific Considerations

**Prokaryotes:**
- Generally require less aggressive trimming
- Shorter read requirements acceptable (MINLEN:30)
- Focus on adapter contamination

**Eukaryotes:**
- May need more stringent quality filtering
- Consider polyA tail removal for RNA-seq
- Longer minimum read length (MINLEN:50 or higher)
- More sensitive to adapter contamination

---

## 3. Alignment to Reference Genome

Map cleaned reads to a reference genome to identify their genomic origins.

### Popular Tools

#### BWA (Burrows-Wheeler Aligner)
**Purpose:** Fast and accurate alignment for short reads

**Input Files:**
- Reference genome (`.fasta`, `.fa`)
- Trimmed FASTQ files

**Output Files:**
- SAM/BAM alignment file

**Sample Commands:**
```bash
# Index reference genome (one-time step)
bwa index reference.fasta

# Align paired-end reads (BWA-MEM for reads >70bp)
bwa mem -t 8 -R "@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA" \
  reference.fasta \
  sample_R1_trimmed.fastq.gz sample_R2_trimmed.fastq.gz \
  > sample_aligned.sam

# For shorter reads (BWA-ALN)
bwa aln -t 8 reference.fasta sample_R1_trimmed.fastq.gz > sample_R1.sai
bwa aln -t 8 reference.fasta sample_R2_trimmed.fastq.gz > sample_R2.sai
bwa sampe reference.fasta sample_R1.sai sample_R2.sai \
  sample_R1_trimmed.fastq.gz sample_R2_trimmed.fastq.gz > sample_aligned.sam

# Convert SAM to BAM
samtools view -bS sample_aligned.sam > sample_aligned.bam
```

**Pros:**
- Very fast and memory-efficient
- Excellent for short reads
- Industry standard
- Good for prokaryotes

**Cons:**
- Not optimal for long reads (>1kb)
- Less accurate with indels compared to newer aligners
- Not splice-aware (unsuitable for RNA-seq)

**Best for:** DNA-seq for prokaryotes and eukaryotes

---

#### Bowtie2
**Purpose:** Fast gapped read aligner for short reads

**Input Files:**
- Reference genome
- Trimmed FASTQ files

**Output Files:**
- SAM alignment file

**Sample Commands:**
```bash
# Build index (one-time step)
bowtie2-build reference.fasta reference_index

# Align paired-end reads
bowtie2 -x reference_index \
  -1 sample_R1_trimmed.fastq.gz -2 sample_R2_trimmed.fastq.gz \
  -S sample_aligned.sam -p 8 --very-sensitive

# Single-end alignment
bowtie2 -x reference_index -U sample_trimmed.fastq.gz \
  -S sample_aligned.sam -p 8 --very-sensitive
```

**Pros:**
- Very fast
- Memory efficient
- Multiple sensitivity presets
- Good for small genomes

**Cons:**
- Not splice-aware
- Less accurate than BWA-MEM for longer reads

**Best for:** Prokaryotic genomes, ChIP-seq, ATAC-seq

---

#### STAR (Spliced Transcripts Alignment to a Reference)
**Purpose:** RNA-seq aligner with splice junction detection

**Input Files:**
- Reference genome
- Gene annotation (GTF/GFF)
- Trimmed FASTQ files

**Output Files:**
- BAM alignment file
- Splice junction file
- Gene counts (optional)

**Sample Commands:**
```bash
# Generate genome index (one-time step)
STAR --runMode genomeGenerate \
  --genomeDir star_index/ \
  --genomeFastaFiles reference.fasta \
  --sjdbGTFfile genes.gtf \
  --sjdbOverhang 99 \
  --runThreadN 8

# Align RNA-seq reads
STAR --genomeDir star_index/ \
  --readFilesIn sample_R1_trimmed.fastq.gz sample_R2_trimmed.fastq.gz \
  --readFilesCommand zcat \
  --outFileNamePrefix sample_ \
  --outSAMtype BAM SortedByCoordinate \
  --outSAMunmapped Within \
  --quantMode GeneCounts \
  --runThreadN 8
```

**Pros:**
- Excellent splice junction detection
- Very fast
- Can generate gene counts directly
- Gold standard for RNA-seq

**Cons:**
- High memory requirement (30GB+ for human genome)
- Overkill for prokaryotes
- Complex indexing

**Best for:** Eukaryotic RNA-seq exclusively

---

#### HISAT2
**Purpose:** Fast and sensitive alignment with splice awareness

**Input Files:**
- Reference genome
- Trimmed FASTQ files
- Known splice sites (optional)

**Output Files:**
- SAM alignment file

**Sample Commands:**
```bash
# Build index
hisat2-build reference.fasta hisat2_index

# Extract splice sites and exons from GTF (optional but recommended)
hisat2_extract_splice_sites.py genes.gtf > splicesites.txt
hisat2_extract_exons.py genes.gtf > exons.txt

# Build index with known splice sites
hisat2-build --ss splicesites.txt --exon exons.txt \
  reference.fasta hisat2_index

# Align RNA-seq reads
hisat2 -x hisat2_index \
  -1 sample_R1_trimmed.fastq.gz -2 sample_R2_trimmed.fastq.gz \
  -S sample_aligned.sam -p 8 --dta
```

**Pros:**
- Lower memory than STAR
- Fast alignment
- Splice-aware
- Good for both DNA and RNA

**Cons:**
- Slightly less accurate than STAR for complex splicing
- Still memory-intensive for large genomes

**Best for:** Eukaryotic RNA-seq with limited computational resources

---

#### Minimap2
**Purpose:** Long-read and short-read aligner

**Input Files:**
- Reference genome
- FASTQ/FASTA reads (short or long)

**Output Files:**
- SAM/BAM alignment file

**Sample Commands:**
```bash
# Index reference (optional for speed)
minimap2 -d reference.mmi reference.fasta

# Align short reads (similar to BWA-MEM)
minimap2 -ax sr reference.mmi sample_R1.fastq.gz sample_R2.fastq.gz > aligned.sam

# Align long reads (PacBio/Nanopore)
minimap2 -ax map-pb reference.mmi long_reads.fastq.gz > aligned.sam
minimap2 -ax map-ont reference.mmi long_reads.fastq.gz > aligned.sam

# RNA-seq (splice-aware)
minimap2 -ax splice reference.mmi reads.fastq.gz > aligned.sam
```

**Pros:**
- Extremely versatile (handles all read types)
- Very fast
- Low memory
- Future-proof

**Cons:**
- Less established for short reads
- May have slightly lower accuracy than specialized tools

**Best for:** Long-read sequencing, multi-platform projects

---

### Organism-Specific Considerations

**Prokaryotes:**
- Use BWA-MEM or Bowtie2 for DNA-seq
- No splice-aware aligner needed (no introns)
- Faster alignment due to smaller genomes
- Lower computational requirements

**Eukaryotes:**
- Use BWA-MEM for DNA-seq (WGS, WES)
- Use STAR or HISAT2 for RNA-seq
- Consider genome complexity and repeats
- Higher memory and time requirements
- Splice-aware alignment essential for RNA-seq

---

## 4. Post-Alignment Processing

Clean and prepare aligned reads for variant calling through sorting, deduplication, and quality assessment.

### Popular Tools

#### SAMtools
**Purpose:** Swiss-army knife for SAM/BAM file manipulation

**Input Files:**
- SAM/BAM files

**Output Files:**
- Sorted BAM files
- Indexed BAM files (`.bai`)
- Statistics files

**Sample Commands:**
```bash
# Convert SAM to BAM
samtools view -bS sample_aligned.sam > sample_aligned.bam

# Sort BAM file
samtools sort sample_aligned.bam -o sample_sorted.bam -@ 4

# Index BAM file
samtools index sample_sorted.bam

# Generate alignment statistics
samtools flagstat sample_sorted.bam > alignment_stats.txt
samtools idxstats sample_sorted.bam > chr_stats.txt
samtools stats sample_sorted.bam > detailed_stats.txt

# Filter by mapping quality
samtools view -q 20 -b sample_sorted.bam > sample_filtered.bam

# Remove unmapped reads
samtools view -F 4 -b sample_sorted.bam > sample_mapped.bam
```

**Pros:**
- Essential standard tool
- Fast and efficient
- Comprehensive functionality
- Well-documented

**Cons:**
- Command-line only
- Steep learning curve for beginners

**Best for:** All BAM file operations (essential for all pipelines)

---

#### Picard Tools
**Purpose:** Comprehensive toolkit for BAM manipulation and QC

**Input Files:**
- BAM files

**Output Files:**
- Processed BAM files
- QC metrics files

**Sample Commands:**
```bash
# Mark duplicates
java -jar picard.jar MarkDuplicates \
  INPUT=sample_sorted.bam \
  OUTPUT=sample_dedup.bam \
  METRICS_FILE=duplicate_metrics.txt \
  REMOVE_DUPLICATES=false \
  CREATE_INDEX=true

# Collect alignment metrics
java -jar picard.jar CollectAlignmentSummaryMetrics \
  INPUT=sample_dedup.bam \
  OUTPUT=alignment_metrics.txt \
  REFERENCE_SEQUENCE=reference.fasta

# Collect insert size metrics (paired-end only)
java -jar picard.jar CollectInsertSizeMetrics \
  INPUT=sample_dedup.bam \
  OUTPUT=insert_size_metrics.txt \
  HISTOGRAM_FILE=insert_size_histogram.pdf

# Add or replace read groups
java -jar picard.jar AddOrReplaceReadGroups \
  INPUT=sample_sorted.bam \
  OUTPUT=sample_rg.bam \
  RGID=sample1 \
  RGLB=lib1 \
  RGPL=ILLUMINA \
  RGPU=unit1 \
  RGSM=sample1
```

**Pros:**
- Comprehensive quality metrics
- Industry standard
- Integrates well with GATK
- Detailed documentation

**Cons:**
- Slower than alternatives
- Java-based (higher memory usage)
- Verbose syntax

**Best for:** GATK-based pipelines, comprehensive QC

---

#### sambamba
**Purpose:** Fast SAM/BAM manipulation tool

**Input Files:**
- BAM files

**Output Files:**
- Processed BAM files

**Sample Commands:**
```bash
# Sort BAM (faster than samtools)
sambamba sort -t 8 -o sample_sorted.bam sample_aligned.bam

# Mark duplicates (faster than Picard)
sambamba markdup -t 8 sample_sorted.bam sample_dedup.bam

# Filter by mapping quality and flags
sambamba view -F "mapping_quality >= 20 and not duplicate" \
  -f bam -t 8 sample_dedup.bam > sample_filtered.bam

# Index BAM
sambamba index -t 8 sample_sorted.bam
```

**Pros:**
- Much faster than SAMtools/Picard
- Multi-threaded operations
- Simple syntax
- Low memory usage

**Cons:**
- Less comprehensive than Picard
- Smaller user community

**Best for:** Large datasets requiring speed, both organisms

---

#### GATK BaseRecalibrator
**Purpose:** Recalibrate base quality scores

**Input Files:**
- BAM files
- Reference genome
- Known variants VCF (dbSNP, etc.)

**Output Files:**
- Recalibration table
- Recalibrated BAM

**Sample Commands:**
```bash
# Generate recalibration table
gatk BaseRecalibrator \
  -I sample_dedup.bam \
  -R reference.fasta \
  --known-sites dbsnp.vcf.gz \
  --known-sites gold_standard_indels.vcf.gz \
  -O recal_data.table

# Apply recalibration
gatk ApplyBQSR \
  -I sample_dedup.bam \
  -R reference.fasta \
  --bqsr-recal-file recal_data.table \
  -O sample_recal.bam
```

**Pros:**
- Improves variant calling accuracy
- Standard in clinical pipelines
- Reduces false positives

**Cons:**
- Requires known variants database
- Computationally intensive
- Not applicable to non-model organisms

**Best for:** Eukaryotic clinical/research pipelines with known variants

---

### Standard Post-Alignment Workflow

```bash
# Complete post-alignment processing pipeline

# 1. Sort BAM
samtools sort -@ 8 sample_aligned.bam -o sample_sorted.bam

# 2. Mark duplicates
java -jar picard.jar MarkDuplicates \
  INPUT=sample_sorted.bam \
  OUTPUT=sample_dedup.bam \
  METRICS_FILE=dup_metrics.txt \
  CREATE_INDEX=true

# 3. Add read groups (if not done during alignment)
java -jar picard.jar AddOrReplaceReadGroups \
  INPUT=sample_dedup.bam \
  OUTPUT=sample_rg.bam \
  RGID=1 RGLB=lib1 RGPL=ILLUMINA RGPU=unit1 RGSM=sample1 \
  CREATE_INDEX=true

# 4. Base quality recalibration (optional, for eukaryotes with known variants)
gatk BaseRecalibrator \
  -I sample_rg.bam \
  -R reference.fasta \
  --known-sites known_variants.vcf.gz \
  -O recal_data.table

gatk ApplyBQSR \
  -I sample_rg.bam \
  -R reference.fasta \
  --bqsr-recal-file recal_data.table \
  -O sample_final.bam

# 5. Generate QC metrics
samtools flagstat sample_final.bam > flagstat.txt
samtools idxstats sample_final.bam > idxstats.txt
```

---

### Organism-Specific Considerations

**Prokaryotes:**
- Simpler processing (fewer steps)
- Duplicate marking less critical (lower duplication rates)
- Base recalibration usually skipped (no known variant databases)
- Faster processing due to smaller genomes

**Eukaryotes:**
- All processing steps recommended
- Duplicate marking essential (PCR duplicates common)
- Base recalibration improves accuracy significantly
- Read groups essential for multi-sample calling
- Longer processing time

---

## 5. Variant Calling

Identify genetic variants (SNPs, indels, structural variants) from aligned reads.

### Popular Tools

#### GATK HaplotypeCaller
**Purpose:** Gold standard for germline variant calling in eukaryotes

**Input Files:**
- Recalibrated BAM files
- Reference genome
- Target intervals (optional)

**Output Files:**
- VCF/gVCF files

**Sample Commands:**
```bash
# Single sample calling
gatk HaplotypeCaller \
  -R reference.fasta \
  -I sample_final.bam \
  -O sample_variants.vcf.gz \
  --native-pair-hmm-threads 4

# Generate gVCF for joint calling
gatk HaplotypeCaller \
  -R reference.fasta \
  -I sample_final.bam \
  -O sample_variants.g.vcf.gz \
  -ERC GVCF

# Joint genotyping (multiple samples)
gatk CombineGVCFs \
  -R reference.fasta \
  --variant sample1.g.vcf.gz \
  --variant sample2.g.vcf.gz \
  -O cohort.g.vcf.gz

gatk GenotypeGVCFs \
  -R reference.fasta \
  -V cohort.g.vcf.gz \
  -O cohort_variants.vcf.gz
```

**Pros:**
- Highest accuracy for diploid organisms
- Excellent indel detection
- Best practice for human genomics
- Joint calling capability
- Well-maintained and documented

**Cons:**
- Very slow
- High computational requirements
- Not ideal for haploid organisms
- Complex workflow

**Best for:** Eukaryotic diploid genomes (human, mouse, plants)

---

#### BCFtools mpileup/call
**Purpose:** Fast variant caller for all organisms

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- VCF files

**Sample Commands:**
```bash
# Generate pileup and call variants
bcftools mpileup -f reference.fasta sample_final.bam | \
  bcftools call -mv -Oz -o sample_variants.vcf.gz

# Multi-sample calling
bcftools mpileup -f reference.fasta sample1.bam sample2.bam | \
  bcftools call -mv -Oz -o cohort_variants.vcf.gz

# For haploid organisms (prokaryotes)
bcftools mpileup -f reference.fasta sample.bam | \
  bcftools call -mv --ploidy 1 -Oz -o variants.vcf.gz

# Index VCF
bcftools index variants.vcf.gz
```

**Pros:**
- Very fast
- Low memory usage
- Simple workflow
- Handles both haploid and diploid
- Good for prokaryotes

**Cons:**
- Lower accuracy than GATK for complex regions
- Less sophisticated indel calling
- Simpler statistical model

**Best for:** Prokaryotes, preliminary analysis, speed-critical projects

---

#### FreeBayes
**Purpose:** Bayesian variant caller for all organisms

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- VCF files

**Sample Commands:**
```bash
# Single sample
freebayes -f reference.fasta sample_final.bam > sample_variants.vcf

# Multiple samples (population calling)
freebayes -f reference.fasta \
  sample1.bam sample2.bam sample3.bam > cohort_variants.vcf

# Haploid calling (prokaryotes)
freebayes -f reference.fasta --ploidy 1 sample.bam > variants.vcf

# With quality filters
freebayes -f reference.fasta \
  --min-mapping-quality 20 \
  --min-base-quality 20 \
  --min-alternate-fraction 0.2 \
  sample.bam > variants.vcf
```

**Pros:**
- Works well for both haploid and diploid
- Population-aware calling
- No pre-processing required
- Handles complex variants

**Cons:**
- Slower than BCFtools
- Can produce many false positives without filtering
- Memory-intensive for large genomes

**Best for:** Population studies, prokaryotes, complex variants

---

#### DeepVariant
**Purpose:** Deep learning-based variant caller

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- VCF files

**Sample Commands:**
```bash
# Using Docker
docker run \
  -v "${PWD}:/input" \
  google/deepvariant:latest \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/input/reference.fasta \
  --reads=/input/sample_final.bam \
  --output_vcf=/input/sample_variants.vcf.gz \
  --num_shards=4

# For different data types: WGS, WES, PACBIO, HYBRID
# Change --model_type accordingly
```

**Pros:**
- State-of-the-art accuracy
- No parameter tuning needed
- Handles various sequencing platforms
- Reduced false positives

**Cons:**
- Requires GPU for speed (CPU is very slow)
- Large computational footprint
- Black box approach
- Trained primarily on eukaryotic data

**Best for:** High-accuracy eukaryotic variant calling with GPU access

---

#### Snippy
**Purpose:** Rapid bacterial variant calling pipeline

**Input Files:**
- FASTQ files
- Reference genome

**Output Files:**
- VCF files
- Annotated variants
- Consensus sequences

**Sample Commands:**
```bash
# Single isolate
snippy --outdir snippy_output \
  --ref reference.gbk \
  --R1 sample_R1.fastq.gz \
  --R2 sample_R2.fastq.gz \
  --cpus 8

# Multiple isolates (core genome alignment)
snippy-multi input_samples.tab --ref reference.gbk --cpus 8 > runme.sh
sh ./runme.sh

snippy-core --ref reference.gbk snippy_*
```

**Pros:**
- All-in-one pipeline for bacteria
- Fast and easy to use
- Includes annotation
- Core genome SNP analysis

**Cons:**
- Prokaryote-specific
- Less control over parameters
- Not suitable for eukaryotes

**Best for:** Bacterial comparative genomics and outbreak analysis

---

#### Strelka2
**Purpose:** Fast and accurate variant caller for germline and somatic variants

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- VCF files (SNVs and indels separate)

**Sample Commands:**
```bash
# Configure germline workflow
configureStrelkaGermlineWorkflow.py \
  --bam sample_final.bam \
  --referenceFasta reference.fasta \
  --runDir strelka_output

# Run workflow
strelka_output/runWorkflow.py -m local -j 8

# For somatic calling (tumor-normal pairs)
configureStrelkaSomaticWorkflow.py \
  --normalBam normal.bam \
  --tumorBam tumor.bam \
  --referenceFasta reference.fasta \
  --runDir strelka_somatic

strelka_somatic/runWorkflow.py -m local -j 8
```

**Pros:**
- Very fast
- Good accuracy
- Designed for both germline and somatic
- Handles tumor-normal pairs

**Cons:**
- Less accurate than GATK for germline
- Requires workflow setup

**Best for:** Cancer genomics, fast germline calling in eukaryotes

---

### Organism-Specific Considerations

**Prokaryotes:**
- Use BCFtools, FreeBayes, or Snippy
- Set ploidy to 1 (haploid)
- Faster calling due to smaller genomes
- Simpler variant landscapes
- Focus on SNPs more than indels

**Eukaryotes:**
- Use GATK HaplotypeCaller (gold standard) or DeepVariant
- Set ploidy to 2 (diploid) or higher (polyploid)
- Longer processing time
- More complex variants
- Joint calling recommended for cohorts
- Consider somatic vs germline variants

---

## 6. Variant Filtering

Remove false positive variants and low-quality calls to generate high-confidence variant sets.

### Popular Tools

#### GATK VariantFiltration
**Purpose:** Apply hard filters to variant calls

**Input Files:**
- Raw VCF files
- Reference genome

**Output Files:**
- Filtered VCF files

**Sample Commands:**
```bash
# Hard filtering for SNPs
gatk VariantFiltration \
  -R reference.fasta \
  -V raw_variants.vcf.gz \
  -O filtered_snps.vcf.gz \
  --filter-name "QD_filter" --filter-expression "QD < 2.0" \
  --filter-name "FS_filter" --filter-expression "FS > 60.0" \
  --filter-name "MQ_filter" --filter-expression "MQ < 40.0" \
  --filter-name "SOR_filter" --filter-expression "SOR > 3.0" \
  --filter-name "MQRankSum_filter" --filter-expression "MQRankSum < -12.5" \
  --filter-name "ReadPosRankSum_filter" --filter-expression "ReadPosRankSum < -8.0"

# Hard filtering for Indels
gatk VariantFiltration \
  -R reference.fasta \
  -V raw_indels.vcf.gz \
  -O filtered_indels.vcf.gz \
  --filter-name "QD_filter" --filter-expression "QD < 2.0" \
  --filter-name "FS_filter" --filter-expression "FS > 200.0" \
  --filter-name "SOR_filter" --filter-expression "SOR > 10.0" \
  --filter-name "ReadPosRankSum_filter" --filter-expression "ReadPosRankSum < -20.0"

# Select only PASS variants
gatk SelectVariants \
  -R reference.fasta \
  -V filtered_variants.vcf.gz \
  -O pass_only.vcf.gz \
  --exclude-filtered
```

**Pros:**
- Precise control over filters
- Well-documented filter recommendations
- Integrates with GATK workflow
- Transparent filtering

**Cons:**
- Requires manual threshold setting
- Less sophisticated than VQSR
- May filter true positives

**Best for:** Small datasets, non-model organisms without training data

---

#### GATK VQSR (Variant Quality Score Recalibration)
**Purpose:** Machine learning-based variant filtering

**Input Files:**
- Raw VCF files
- Training/truth datasets (HapMap, 1000G, etc.)
- Reference genome

**Output Files:**
- Recalibrated VCF files

**Sample Commands:**
```bash
# Build recalibration model for SNPs
gatk VariantRecalibrator \
  -R reference.fasta \
  -V raw_variants.vcf.gz \
  --resource:hapmap,known=false,training=true,truth=true,prior=15.0 hapmap.vcf.gz \
  --resource:omni,known=false,training=true,truth=true,prior=12.0 omni.vcf.gz \
  --resource:1000G,known=false,training=true,truth=false,prior=10.0 1000G.vcf.gz \
  --resource:dbsnp,known=true,training=false,truth=false,prior=2.0 dbsnp.vcf.gz \
  -an QD -an MQ -an MQRankSum -an ReadPosRankSum -an FS -an SOR \
  -mode SNP \
  -O snp_recal.recal \
  --tranches-file snp_recal.tranches \
  --rscript-file snp_recal.plots.R

# Apply recalibration
gatk ApplyVQSR \
  -R reference.fasta \
  -V raw_variants.vcf.gz \
  -O filtered_snps.vcf.gz \
  --recal-file snp_recal.recal \
  --tranches-file snp_recal.tranches \
  --truth-sensitivity-filter-level 99.0 \
  -mode SNP

# Repeat for INDELs
# (similar commands with -mode INDEL and different resources)
```

**Pros:**
- Superior accuracy for model organisms
- Adaptive filtering
- Maximizes sensitivity and specificity
- Industry standard for human genomics

**Cons:**
- Requires large datasets (>30 samples recommended)
- Needs high-quality training sets
- Only applicable to model organisms
- Complex workflow

**Best for:** Large eukaryotic cohorts (human, mouse) with training data

---

#### BCFtools filter
**Purpose:** Flexible VCF filtering tool

**Input Files:**
- VCF files

**Output Files:**
- Filtered VCF files

**Sample Commands:**
```bash
# Filter by quality score
bcftools filter -i 'QUAL>30' input.vcf.gz -Oz -o filtered.vcf.gz

# Filter by depth and quality
bcftools filter -i 'QUAL>30 && DP>10 && DP<200' input.vcf.gz -Oz -o filtered.vcf.gz

# Filter by allele frequency (for population data)
bcftools filter -i 'AF>0.05' input.vcf.gz -Oz -o filtered.vcf.gz

# Complex filtering
bcftools filter \
  -i 'QUAL>30 && DP>10 && MQ>40 && (GT="1/1" || GT="0/1")' \
  input.vcf.gz -Oz -o filtered.vcf.gz

# Remove variants near indels
bcftools filter -g 10 input.vcf.gz -Oz -o filtered.vcf.gz

# Mark low-quality variants (soft filter)
bcftools filter -s "LowQual" -e 'QUAL<30' input.vcf.gz -Oz -o filtered.vcf.gz
```

**Pros:**
- Very flexible and fast
- Simple syntax
- Works with any VCF
- Both hard and soft filtering

**Cons:**
- Requires knowledge of VCF fields
- Manual threshold selection
- No machine learning

**Best for:** Quick filtering, prokaryotes, custom workflows

---

#### VCFtools
**Purpose:** Comprehensive VCF manipulation and filtering

**Input Files:**
- VCF files

**Output Files:**
- Filtered VCF files
- Statistics files

**Sample Commands:**
```bash
# Filter by quality and depth
vcftools --gzvcf input.vcf.gz \
  --minQ 30 \
  --min-meanDP 10 \
  --max-meanDP 200 \
  --recode --recode-INFO-all \
  --out filtered

# Filter by missing data
vcftools --gzvcf input.vcf.gz \
  --max-missing 0.9 \
  --recode --recode-INFO-all \
  --out filtered

# Filter by minor allele frequency
vcftools --gzvcf input.vcf.gz \
  --maf 0.05 \
  --recode --recode-INFO-all \
  --out filtered

# Remove specific individuals
vcftools --gzvcf input.vcf.gz \
  --remove-indv sample1 \
  --remove-indv sample2 \
  --recode --recode-INFO-all \
  --out filtered

# Generate statistics
vcftools --gzvcf input.vcf.gz --freq --out allele_freq
vcftools --gzvcf input.vcf.gz --depth --out depth_stats
vcftools --gzvcf input.vcf.gz --site-mean-depth --out site_depth
```

**Pros:**
- User-friendly
- Many built-in statistics
- Good documentation
- Population genetics focused

**Cons:**
- Slower than BCFtools
- Less flexible filtering expressions
- Awkward output format

**Best for:** Population genomics, eukaryotic studies

---

### Recommended Filtering Criteria

#### For Prokaryotes (Haploid):
```bash
bcftools filter \
  -i 'QUAL>=30 && DP>=10 && MQ>=30 && AF>=0.8' \
  input.vcf.gz -Oz -o filtered.vcf.gz
```
- QUAL ≥ 30 (variant quality)
- DP ≥ 10 (read depth)
- MQ ≥ 30 (mapping quality)
- AF ≥ 0.8 (allele frequency, for haploid)

#### For Eukaryotes (Diploid):
```bash
# SNPs
bcftools filter \
  -i 'QUAL>=30 && DP>=10 && DP<=200 && MQ>=40 && QD>=2.0 && FS<=60.0 && SOR<=3.0' \
  snps.vcf.gz -Oz -o filtered_snps.vcf.gz

# Indels
bcftools filter \
  -i 'QUAL>=30 && DP>=10 && DP<=200 && QD>=2.0 && FS<=200.0 && SOR<=10.0' \
  indels.vcf.gz -Oz -o filtered_indels.vcf.gz
```

---

### Organism-Specific Considerations

**Prokaryotes:**
- Simpler filtering (haploid)
- Focus on high allele frequency (AF > 0.8)
- Lower depth acceptable (DP > 10)
- Stricter allele frequency filters
- Watch for mixed populations

**Eukaryotes:**
- More complex filtering (heterozygosity)
- Allow intermediate AF (0.2-0.8 for heterozygotes)
- Higher depth recommended (DP > 20)
- Separate SNP and indel filtering
- Consider VQSR for large cohorts

---

## 7. Variant Annotation

Add biological context to variants including functional effects, population frequencies, and clinical significance.

### Popular Tools

#### SnpEff
**Purpose:** Genetic variant annotation and effect prediction

**Input Files:**
- VCF files
- Genome database (downloaded or custom)

**Output Files:**
- Annotated VCF files
- HTML summary report
- Statistics files

**Sample Commands:**
```bash
# Download pre-built database
java -jar snpEff.jar download GRCh38.99

# Annotate variants
java -jar snpEff.jar -v GRCh38.99 input.vcf.gz > annotated.vcf

# For prokaryotes (custom database)
# First, build database from GenBank file
java -jar snpEff.jar build -genbank -v bacteria_genome

# Then annotate
java -jar snpEff.jar -v bacteria_genome input.vcf.gz > annotated.vcf

# With statistics
java -jar snpEff.jar -v -stats annotation_stats.html \
  GRCh38.99 input.vcf.gz > annotated.vcf

# Generate HTML report and VCF
java -Xmx8g -jar snpEff.jar -v \
  -stats snpeff_summary.html \
  -csvStats snpeff_stats.csv \
  GRCh38.99 filtered.vcf.gz > annotated.vcf
```

**Pros:**
- Fast annotation
- Extensive pre-built databases
- Predicts variant effects
- Easy to use
- Good for both prokaryotes and eukaryotes
- Detailed HTML reports

**Cons:**
- Effect predictions can be inaccurate
- Limited clinical annotations
- May miss rare transcripts

**Best for:** Initial functional annotation for all organisms

---

#### VEP (Variant Effect Predictor)
**Purpose:** Comprehensive variant annotation with multiple data sources

**Input Files:**
- VCF files
- Cache files or database connection

**Output Files:**
- Annotated VCF/TSV files
- HTML statistics

**Sample Commands:**
```bash
# Download cache
vep_install -a cf -s homo_sapiens -y GRCh38

# Basic annotation
vep -i input.vcf.gz \
  -o annotated.vcf \
  --cache \
  --species homo_sapiens \
  --assembly GRCh38 \
  --vcf \
  --fork 4

# Comprehensive annotation
vep -i input.vcf.gz \
  -o annotated.vcf \
  --cache \
  --everything \
  --assembly GRCh38 \
  --vcf \
  --fork 4 \
  --plugin CADD,whole_genome_SNVs.tsv.gz \
  --plugin dbNSFP,dbNSFP.gz,ALL \
  --plugin SpliceAI,snv=spliceai_scores.raw.snv.hg38.vcf.gz

# Output as tab-delimited
vep -i input.vcf.gz \
  -o annotated.txt \
  --cache \
  --tab \
  --fields "Location,Allele,Gene,Feature,Consequence,SYMBOL,SIFT,PolyPhen" \
  --fork 4
```

**Pros:**
- Most comprehensive annotations
- Multiple prediction algorithms
- Clinical annotations (ClinVar, etc.)
- Regular updates
- Plugin system for custom annotations
- Industry standard

**Cons:**
- Slower than SnpEff
- Large cache files
- Complex setup
- Primarily eukaryote-focused
- Requires significant storage

**Best for:** Clinical applications, comprehensive eukaryotic annotation

---

#### ANNOVAR
**Purpose:** Functional annotation of genetic variants

**Input Files:**
- VCF or custom format files
- Annotation databases

**Output Files:**
- Annotated text files
- VCF files (optional)

**Sample Commands:**
```bash
# Download databases
annotate_variation.pl -buildver hg38 -downdb -webfrom annovar refGene humandb/
annotate_variation.pl -buildver hg38 -downdb -webfrom annovar clinvar_20210501 humandb/
annotate_variation.pl -buildver hg38 -downdb -webfrom annovar gnomad30_genome humandb/

# Convert VCF to ANNOVAR format
convert2annovar.pl -format vcf4 input.vcf.gz > input.avinput

# Annotate with multiple databases
table_annovar.pl input.avinput humandb/ \
  -buildver hg38 \
  -out annotated \
  -remove \
  -protocol refGene,cytoBand,gnomad30_genome,clinvar_20210501,dbnsfp42a \
  -operation g,r,f,f,f \
  -nastring . \
  -vcfinput input.vcf.gz

# Generate VCF output
table_annovar.pl input.vcf.gz humandb/ \
  -buildver hg38 \
  -out annotated \
  -remove \
  -protocol refGene,gnomad30_genome \
  -operation g,f \
  -vcfinput
```

**Pros:**
- Fast annotation
- Many database options
- Flexible output formats
- Good for filtering variants
- Customizable

**Cons:**
- Less user-friendly interface
- Requires manual database management
- Less comprehensive than VEP
- Outdated documentation

**Best for:** Custom annotation pipelines, filtering workflows

---

#### Prokka (Prokaryote-specific)
**Purpose:** Rapid prokaryotic genome annotation

**Input Files:**
- FASTA genome sequences

**Output Files:**
- GFF3, GBK, FNA, FAA files
- Annotated features

**Sample Commands:**
```bash
# Basic annotation
prokka --outdir prokka_output \
  --prefix sample \
  --cpus 8 \
  genome.fasta

# With custom parameters
prokka --outdir prokka_output \
  --prefix sample \
  --genus Escherichia \
  --species coli \
  --strain K12 \
  --cpus 8 \
  --rfam \
  genome.fasta

# Use annotations for variant interpretation
# Prokka output can be used with SnpEff for variant annotation
```

**Pros:**
- Very fast
- Prokaryote-optimized
- Multiple output formats
- Includes RNA genes
- Easy to use

**Cons:**
- Prokaryotes only
- Not for variant annotation directly
- Less comprehensive than manual curation

**Best for:** Rapid bacterial genome annotation

---

#### bcftools annotate
**Purpose:** Add or remove annotations from VCF files

**Input Files:**
- VCF files
- Annotation files (VCF, BED, TSV)

**Output Files:**
- Annotated VCF files

**Sample Commands:**
```bash
# Add INFO from another VCF
bcftools annotate -a dbsnp.vcf.gz \
  -c ID,INFO/CAF \
  -Oz -o annotated.vcf.gz \
  input.vcf.gz

# Add annotations from BED file
bcftools annotate -a regions.bed.gz \
  -c CHROM,FROM,TO,REGION \
  -h <(echo '##INFO=<ID=REGION,Number=1,Type=String,Description="Region name">') \
  -Oz -o annotated.vcf.gz \
  input.vcf.gz

# Add custom annotations from TSV
bcftools annotate -a annotations.tsv.gz \
  -c CHROM,POS,REF,ALT,INFO/CUSTOM \
  -h <(echo '##INFO=<ID=CUSTOM,Number=1,Type=String,Description="Custom annotation">') \
  -Oz -o annotated.vcf.gz \
  input.vcf.gz

# Remove annotations
bcftools annotate -x INFO/DP,FORMAT/GQ \
  -Oz -o cleaned.vcf.gz \
  input.vcf.gz
```

**Pros:**
- Very fast
- Flexible annotation sources
- Simple syntax
- Works with any organism

**Cons:**
- Requires pre-formatted annotation files
- No effect prediction
- Manual database preparation

**Best for:** Adding custom annotations, combining VCF files

---

### Recommended Annotation Workflow

#### For Eukaryotes:
```bash
# 1. Functional annotation with VEP
vep -i filtered.vcf.gz \
  -o vep_annotated.vcf \
  --cache --everything \
  --assembly GRCh38 \
  --vcf --fork 4

# 2. Add population frequencies
bcftools annotate -a gnomad.vcf.gz \
  -c INFO/AF,INFO/AF_popmax \
  vep_annotated.vcf -Oz -o final_annotated.vcf.gz

# 3. Extract relevant fields
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/CSQ\n' \
  final_annotated.vcf.gz > variants_table.txt
```

#### For Prokaryotes:
```bash
# 1. Annotate genome with Prokka (one-time)
prokka --outdir annotation --prefix ref genome.fasta

# 2. Build SnpEff database from Prokka output
java -jar snpEff.jar build -genbank -v ref

# 3. Annotate variants
java -jar snpEff.jar -v ref filtered.vcf.gz > annotated.vcf

# 4. Extract high-impact variants
java -jar SnpSift.jar filter \
  "(ANN[*].IMPACT = 'HIGH')" \
  annotated.vcf > high_impact.vcf
```

---

### Key Annotation Fields

**Functional Effects:**
- Gene name and ID
- Transcript ID
- Consequence (missense, nonsense, synonymous, etc.)
- Amino acid change
- Impact (HIGH, MODERATE, LOW, MODIFIER)

**Population Data:**
- Allele frequency (AF) in populations
- gnomAD frequencies
- 1000 Genomes data

**Clinical Significance:**
- ClinVar annotations
- COSMIC (cancer variants)
- OMIM (genetic disorders)

**Prediction Scores:**
- SIFT (deleterious prediction)
- PolyPhen (pathogenicity prediction)
- CADD score (deleteriousness)
- GERP (conservation)

---

### Organism-Specific Considerations

**Prokaryotes:**
- Use SnpEff with custom databases
- Annotate genome first with Prokka
- Focus on gene disruption and AMR genes
- Simpler annotation (no splicing)
- Consider operon structure

**Eukaryotes:**
- Use VEP or SnpEff with pre-built databases
- Include clinical annotations
- Consider splice variants
- Multiple transcript isoforms
- Regulatory regions important
- Population frequencies critical

---

## 8. Downstream Analysis

Interpret annotated variants and perform biological analysis to extract meaningful insights.

### Variant Prioritization

#### SnpSift
**Purpose:** Filter and manipulate annotated variants

**Input Files:**
- Annotated VCF files

**Output Files:**
- Filtered VCF files
- TSV reports

**Sample Commands:**
```bash
# Filter high-impact variants
java -jar SnpSift.jar filter \
  "(ANN[*].IMPACT = 'HIGH')" \
  annotated.vcf > high_impact.vcf

# Filter by gene list
java -jar SnpSift.jar filter \
  "(ANN[*].GENE in SET[0])" \
  -s genes_of_interest.txt \
  annotated.vcf > filtered.vcf

# Extract variants with SIFT deleterious
java -jar SnpSift.jar filter \
  "(ANN[*].EFFECT has 'missense_variant') & (dbNSFP_SIFT_pred = 'D')" \
  annotated.vcf > deleterious.vcf

# Convert to tab-delimited
java -jar SnpSift.jar extractFields \
  annotated.vcf CHROM POS REF ALT "ANN[0].GENE" "ANN[0].EFFECT" \
  "ANN[0].IMPACT" "ANN[0].HGVS_P" > variants.txt

# Filter by allele frequency (rare variants)
java -jar SnpSift.jar filter \
  "(gnomAD_AF < 0.01) | (na gnomAD_AF)" \
  annotated.vcf > rare_variants.vcf
```

**Pros:**
- Powerful filtering expressions
- Works seamlessly with SnpEff
- Flexible queries
- Good documentation

**Cons:**
- Requires knowledge of VCF structure
- Java-based (memory intensive)

**Best for:** Prioritizing variants after SnpEff annotation

---

### Comparative Genomics (Prokaryotes)

#### Roary
**Purpose:** Pan-genome analysis for prokaryotes

**Input Files:**
- GFF3 files from multiple strains

**Output Files:**
- Gene presence/absence matrix
- Core/accessory genome files
- Phylogenetic trees

**Sample Commands:**
```bash
# Run pan-genome analysis
roary -e -n -v -p 8 -f roary_output *.gff

# With minimum identity threshold
roary -e -n -v -p 8 -i 90 -f roary_output *.gff

# Output:
# - gene_presence_absence.csv (matrix)
# - core_gene_alignment.aln (core genes)
# - accessory_binary_genes.fa.newick (tree)
```

**Pros:**
- Fast pan-genome analysis
- Identifies core and accessory genes
- Generates alignments
- Prokaryote-optimized

**Cons:**
- Prokaryotes only
- Requires high-quality annotations
- Memory-intensive for large datasets

**Best for:** Bacterial comparative genomics

---

#### Mauve
**Purpose:** Multiple genome alignment and visualization

**Input Files:**
- FASTA genome sequences

**Output Files:**
- Alignment files
- Graphical visualizations

**Sample Commands:**
```bash
# Progressive Mauve alignment
progressiveMauve --output=alignment.xmfa \
  genome1.fasta genome2.fasta genome3.fasta

# With GUI
# Launch Mauve and use File > Align with progressiveMauve
```

**Pros:**
- Visual alignment representation
- Identifies rearrangements
- Good for closely related genomes
- User-friendly GUI

**Cons:**
- Limited to small numbers of genomes
- Computationally intensive
- Not suitable for large-scale studies

**Best for:** Visualizing genome rearrangements in bacteria

---

### Phylogenetic Analysis

#### IQ-TREE
**Purpose:** Fast maximum likelihood phylogenetic inference

**Input Files:**
- Multiple sequence alignment (FASTA, PHYLIP, NEXUS)

**Output Files:**
- Phylogenetic tree (Newick format)
- Log files

**Sample Commands:**
```bash
# Basic tree inference
iqtree -s alignment.fasta -m MFP -bb 1000 -nt 8

# With specific model
iqtree -s alignment.fasta -m GTR+G -bb 1000 -nt 8

# Model selection only
iqtree -s alignment.fasta -m TESTONLY -nt 8

# Output:
# - alignment.fasta.treefile (best tree)
# - alignment.fasta.iqtree (log)
```

**Pros:**
- Very fast
- Automatic model selection
- Bootstrap support
- Well-maintained

**Cons:**
- Requires aligned sequences
- Command-line only

**Best for:** Phylogenetic trees from SNP/gene alignments

---

#### RAxML-NG
**Purpose:** Maximum likelihood phylogenetic inference

**Input Files:**
- Multiple sequence alignment

**Output Files:**
- Phylogenetic trees
- Bootstrap trees

**Sample Commands:**
```bash
# Check alignment
raxml-ng --check --msa alignment.fasta --model GTR+G

# Infer tree
raxml-ng --msa alignment.fasta --model GTR+G --threads 8 --prefix output

# With bootstrap
raxml-ng --all --msa alignment.fasta --model GTR+G \
  --bs-trees 100 --threads 8 --prefix output
```

**Pros:**
- Highly accurate
- Good for large datasets
- Parallel processing

**Cons:**
- Slower than IQ-TREE
- More complex syntax

**Best for:** Large phylogenetic analyses

---

### Population Genetics

#### PLINK
**Purpose:** Genome-wide association and population analysis

**Input Files:**
- VCF files or PLINK binary format

**Output Files:**
- Various analysis outputs (PCA, association, LD)

**Sample Commands:**
```bash
# Convert VCF to PLINK format
plink --vcf variants.vcf.gz --make-bed --out dataset

# Principal Component Analysis
plink --bfile dataset --pca 10 --out pca_results

# Calculate allele frequencies
plink --bfile dataset --freq --out allele_freq

# Linkage disequilibrium analysis
plink --bfile dataset --ld-window 99999 --ld-window-kb 1000 \
  --ld-window-r2 0.2 --r2 --out ld_results

# Genome-wide association study (GWAS)
plink --bfile dataset --assoc --adjust --out gwas_results

# Hardy-Weinberg equilibrium
plink --bfile dataset --hardy --out hwe_results
```

**Pros:**
- Industry standard for GWAS
- Fast and efficient
- Comprehensive analyses
- Large user community

**Cons:**
- Steep learning curve
- Primarily for eukaryotes
- Complex file formats

**Best for:** Population genomics, GWAS in eukaryotes

---

#### VCFtools (Statistical Analysis)
**Purpose:** Population genetics statistics

**Input Files:**
- VCF files

**Output Files:**
- Various statistics files

**Sample Commands:**
```bash
# Calculate Tajima's D
vcftools --gzvcf variants.vcf.gz --TajimaD 10000 --out tajima

# Calculate nucleotide diversity (pi)
vcftools --gzvcf variants.vcf.gz --window-pi 10000 --out pi_diversity

# Calculate Fst between populations
vcftools --gzvcf variants.vcf.gz \
  --weir-fst-pop pop1.txt \
  --weir-fst-pop pop2.txt \
  --out fst_results

# Calculate heterozygosity
vcftools --gzvcf variants.vcf.gz --het --out heterozygosity

# Calculate LD
vcftools --gzvcf variants.vcf.gz --geno-r2 --out ld_stats

# Identity by state (IBS) matrix
vcftools --gzvcf variants.vcf.gz --relatedness2 --out ibs_matrix
```

**Pros:**
- Many population statistics
- Easy to use
- Good documentation

**Cons:**
- Slower than PLINK
- Limited to specific analyses

**Best for:** Population genetics statistics

---

### Functional Enrichment Analysis

#### Gene Ontology (GO) Enrichment

**Using Python (goatools):**
```bash
# Install goatools
pip install goatools

# Python script for GO enrichment
python go_enrichment.py \
  --study genes_of_interest.txt \
  --population all_genes.txt \
  --association gene2go.txt \
  --obo go-basic.obo \
  --method fdr_bh \
  --pval 0.05 \
  --outfile go_enrichment_results.txt
```

**Using R (clusterProfiler):**
```r
library(clusterProfiler)
library(org.Hs.eg.db)

# Prepare gene list
genes <- read.table("genes_of_interest.txt")$V1

# GO enrichment
ego <- enrichGO(gene = genes,
                OrgDb = org.Hs.eg.db,
                ont = "BP",
                pAdjustMethod = "BH",
                pvalueCutoff = 0.05,
                qvalueCutoff = 0.05,
                readable = TRUE)

# Visualize results
dotplot(ego, showCategory=20)
barplot(ego, showCategory=20)

# Save results
write.csv(as.data.frame(ego), "go_enrichment_results.csv")
```

**Best for:** Understanding biological processes affected by variants

---

### Structural Variant Analysis

#### Delly
**Purpose:** Structural variant discovery

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- VCF files with SVs

**Sample Commands:**
```bash
# Call SVs
delly call -g reference.fasta -o sv_calls.bcf sample.bam

# Convert to VCF
bcftools view sv_calls.bcf > sv_calls.vcf

# Genotype across multiple samples
delly call -g reference.fasta -v sv_calls.bcf -o genotyped.bcf \
  sample1.bam sample2.bam sample3.bam

# Filter SVs
delly filter -f germline -o filtered.bcf genotyped.bcf
```

**Pros:**
- Detects multiple SV types
- Works with paired-end data
- Fast processing

**Cons:**
- Requires good coverage
- Less accurate than long-read methods

**Best for:** Eukaryotic structural variant detection

---

#### Manta
**Purpose:** Structural variant and indel caller

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- VCF files

**Sample Commands:**
```bash
# Configure workflow
configManta.py \
  --bam sample.bam \
  --referenceFasta reference.fasta \
  --runDir manta_output

# Run workflow
manta_output/runWorkflow.py -m local -j 8
```

**Pros:**
- Fast and accurate
- Good for germline and somatic SVs
- Comprehensive SV types

**Cons:**
- Complex output
- Requires paired-end data

**Best for:** Clinical SV detection in eukaryotes

---

### Visualization Tools

#### IGV (Integrative Genomics Viewer)
**Purpose:** Interactive visualization of genomic data

**Input Files:**
- BAM files
- VCF files
- BED files
- Reference genome

**Usage:**
```bash
# Launch IGV (GUI application)
igv

# Load files through GUI:
# 1. Load reference genome
# 2. File > Load from File > Select BAM/VCF
# 3. Navigate to regions of interest
# 4. Visualize alignments, variants, coverage

# Command-line batch script
igv -b batch_script.txt

# Example batch script:
# new
# genome hg38
# load sample.bam
# load variants.vcf
# goto chr1:1,000,000-1,100,000
# snapshot region.png
# exit
```

**Pros:**
- Excellent visualization
- Interactive exploration
- Supports many formats
- Free and widely used

**Cons:**
- GUI-based (not for pipelines)
- Memory-intensive for large files
- Requires indexed files

**Best for:** Manual variant inspection, publication figures

---

#### JBrowse2
**Purpose:** Web-based genome browser

**Input Files:**
- BAM, VCF, BED, GFF files
- Reference genome

**Setup:**
```bash
# Install JBrowse CLI
npm install -g @jbrowse/cli

# Create new instance
jbrowse create output_folder

# Add reference genome
jbrowse add-assembly reference.fasta --out output_folder

# Add tracks
jbrowse add-track sample.bam --out output_folder
jbrowse add-track variants.vcf.gz --out output_folder

# Serve locally
jbrowse serve output_folder
```

**Pros:**
- Web-based (shareable)
- Modern interface
- Supports many data types
- Good for collaboration

**Cons:**
- Setup complexity
- Requires web server
- Learning curve

**Best for:** Sharing data, collaborative projects

---

### Copy Number Variation Analysis

#### CNVkit (Eukaryotes)
**Purpose:** Copy number variation detection from sequencing data

**Input Files:**
- BAM files
- Reference genome

**Output Files:**
- CNV calls
- Plots

**Sample Commands:**
```bash
# Batch workflow with reference
cnvkit.py batch tumor.bam --normal normal.bam \
  --targets targets.bed \
  --fasta reference.fasta \
  --output-dir cnvkit_output/

# Without matched normal (reference from pooled normals)
cnvkit.py batch tumor.bam \
  --reference pooled_reference.cnn \
  --output-dir cnvkit_output/

# Visualize results
cnvkit.py scatter tumor.cnr -s tumor.cns -o scatter_plot.pdf
cnvkit.py diagram tumor.cnr -s tumor.cns -o diagram.pdf
```

**Pros:**
- Works with targeted and WGS data
- No matched normal required
- Good visualization
- Widely used in cancer genomics

**Cons:**
- Eukaryotes only
- Complex interpretation
- Requires adequate coverage

**Best for:** Cancer genomics, CNV detection in eukaryotes

---

### Antimicrobial Resistance Detection (Prokaryotes)

#### ABRicate
**Purpose:** Mass screening of contigs for antimicrobial resistance genes

**Input Files:**
- FASTA assemblies or contigs

**Output Files:**
- TSV reports with AMR genes

**Sample Commands:**
```bash
# Screen for AMR genes (multiple databases)
abricate --db ncbi genome.fasta > amr_results.txt
abricate --db card genome.fasta > card_results.txt
abricate --db resfinder genome.fasta > resfinder_results.txt

# Summary across multiple samples
abricate --db ncbi sample1.fasta sample2.fasta sample3.fasta > all_results.txt
abricate --summary all_results.txt > summary_matrix.txt

# Available databases: ncbi, card, resfinder, argannot, megares, vfdb
```

**Pros:**
- Very fast
- Multiple databases
- Simple output
- Easy to interpret

**Cons:**
- Assembly-based (not variant-based)
- May miss point mutations
- Prokaryotes only

**Best for:** Rapid AMR screening in bacteria

---

#### ARIBA
**Purpose:** Antimicrobial resistance identification from reads

**Input Files:**
- FASTQ files
- Reference database

**Output Files:**
- Detailed AMR reports

**Sample Commands:**
```bash
# Download and prepare database
ariba getref card card_ref
ariba prepareref -f card_ref.fa -m card_ref.tsv card_db

# Run ARIBA
ariba run card_db reads_1.fastq.gz reads_2.fastq.gz ariba_output

# Summarize multiple samples
ariba summary summary_output ariba_output1/report.tsv ariba_output2/report.tsv
```

**Pros:**
- Detects point mutations
- Works directly from reads
- Accurate AMR prediction
- Multiple resistance mechanisms

**Cons:**
- Slower than ABRicate
- Requires good coverage
- Complex output

**Best for:** Comprehensive AMR detection including mutations

---

### Metabolic Pathway Analysis

#### KEGG Pathway Enrichment

**Using R (pathview):**
```r
library(pathview)
library(KEGGREST)

# Load gene list with fold changes
gene_data <- read.table("gene_expression.txt", header=TRUE)

# KEGG pathway visualization
pathview(gene.data = gene_data,
         pathway.id = "00010",  # Glycolysis pathway
         species = "hsa",        # Human
         out.suffix = "glycolysis")

# Multiple pathways
pathways <- c("00010", "00020", "00030")
for(pathway in pathways) {
  pathview(gene.data = gene_data,
           pathway.id = pathway,
           species = "hsa")
}
```

**Best for:** Understanding metabolic changes

---

### Quality Control of Final Results

#### MultiQC (Comprehensive Report)
**Purpose:** Aggregate all QC metrics into single report

**Input Files:**
- Outputs from all analysis tools

**Output Files:**
- HTML summary report

**Sample Commands:**
```bash
# Aggregate all QC reports
multiqc analysis_directory/ -o final_report/

# With custom configuration
multiqc analysis_directory/ \
  -o final_report/ \
  -c multiqc_config.yaml \
  --title "NGS Analysis Project" \
  --comment "Comprehensive analysis report"
```

**Pros:**
- Comprehensive overview
- Beautiful visualizations
- Supports 100+ tools
- Easy to share

**Cons:**
- Requires compatible tool outputs
- Can be overwhelming

**Best for:** Final project reports, QC overview

---

### Statistical Analysis and Reporting

#### R/Bioconductor Workflow
**Purpose:** Custom statistical analysis

**Example workflow:**
```r
library(VariantAnnotation)
library(ggplot2)
library(dplyr)

# Read VCF file
vcf <- readVcf("annotated.vcf.gz", "hg38")

# Extract information
info_data <- info(vcf)
qual_data <- qual(vcf)

# Basic statistics
summary(qual_data)
table(info_data$EFFECT)

# Variant type distribution
variant_types <- data.frame(
  Type = info_data$VARIANT_TYPE,
  Quality = qual_data
)

ggplot(variant_types, aes(x=Type, y=Quality)) +
  geom_boxplot() +
  theme_bw() +
  labs(title="Variant Quality by Type")

# Transition/Transversion ratio
ti_tv <- calculateTiTv(vcf)
print(ti_tv)

# Export filtered variants
filtered_vcf <- vcf[qual_data > 30]
writeVcf(filtered_vcf, "high_quality.vcf")
```

**Best for:** Custom analyses, publication-quality figures

---

## Complete Analysis Workflows

### Workflow 1: Prokaryotic DNA Sequencing

```bash
#!/bin/bash
# Complete prokaryotic variant calling pipeline

# 1. Quality Control
fastqc reads_R1.fastq.gz reads_R2.fastq.gz -o qc/

# 2. Read Trimming
fastp -i reads_R1.fastq.gz -I reads_R2.fastq.gz \
  -o clean_R1.fastq.gz -O clean_R2.fastq.gz \
  -h fastp_report.html -w 4

# 3. Alignment
bwa mem -t 8 reference.fasta clean_R1.fastq.gz clean_R2.fastq.gz | \
  samtools sort -@ 4 -o aligned.bam -

# 4. Post-alignment processing
samtools index aligned.bam
samtools flagstat aligned.bam > alignment_stats.txt

# 5. Variant Calling (haploid)
bcftools mpileup -f reference.fasta aligned.bam | \
  bcftools call -mv --ploidy 1 -Oz -o variants.vcf.gz
bcftools index variants.vcf.gz

# 6. Variant Filtering
bcftools filter -i 'QUAL>=30 && DP>=10 && MQ>=30' \
  variants.vcf.gz -Oz -o filtered.vcf.gz

# 7. Variant Annotation
java -jar snpEff.jar -v bacteria_genome filtered.vcf.gz > annotated.vcf

# 8. Extract high-impact variants
java -jar SnpSift.jar filter "(ANN[*].IMPACT = 'HIGH')" \
  annotated.vcf > high_impact.vcf

# 9. AMR detection
abricate --db card assembly.fasta > amr_genes.txt

# 10. Generate final report
multiqc . -o final_report/
```

---

### Workflow 2: Eukaryotic DNA Sequencing (GATK Best Practices)

```bash
#!/bin/bash
# Complete eukaryotic variant calling pipeline (GATK)

# 1. Quality Control
fastqc reads_R1.fastq.gz reads_R2.fastq.gz -o qc/
multiqc qc/ -o qc/

# 2. Read Trimming
trimmomatic PE -threads 8 \
  reads_R1.fastq.gz reads_R2.fastq.gz \
  clean_R1_paired.fastq.gz clean_R1_unpaired.fastq.gz \
  clean_R2_paired.fastq.gz clean_R2_unpaired.fastq.gz \
  ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 \
  SLIDINGWINDOW:4:15 MINLEN:36

# 3. Alignment with read groups
bwa mem -t 8 -R "@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA\tLB:lib1" \
  reference.fasta clean_R1_paired.fastq.gz clean_R2_paired.fastq.gz | \
  samtools sort -@ 4 -o sorted.bam -

# 4. Mark Duplicates
gatk MarkDuplicates \
  -I sorted.bam \
  -O dedup.bam \
  -M dup_metrics.txt \
  --CREATE_INDEX true

# 5. Base Quality Recalibration
gatk BaseRecalibrator \
  -I dedup.bam \
  -R reference.fasta \
  --known-sites dbsnp.vcf.gz \
  -O recal_data.table

gatk ApplyBQSR \
  -I dedup.bam \
  -R reference.fasta \
  --bqsr-recal-file recal_data.table \
  -O recal.bam

# 6. Variant Calling
gatk HaplotypeCaller \
  -R reference.fasta \
  -I recal.bam \
  -O raw_variants.vcf.gz

# 7. Variant Filtering (Hard filters)
# Select SNPs
gatk SelectVariants \
  -R reference.fasta \
  -V raw_variants.vcf.gz \
  --select-type-to-include SNP \
  -O raw_snps.vcf.gz

# Filter SNPs
gatk VariantFiltration \
  -R reference.fasta \
  -V raw_snps.vcf.gz \
  -O filtered_snps.vcf.gz \
  --filter-name "QD_filter" --filter-expression "QD < 2.0" \
  --filter-name "FS_filter" --filter-expression "FS > 60.0" \
  --filter-name "MQ_filter" --filter-expression "MQ < 40.0" \
  --filter-name "SOR_filter" --filter-expression "SOR > 3.0"

# Select INDELs
gatk SelectVariants \
  -R reference.fasta \
  -V raw_variants.vcf.gz \
  --select-type-to-include INDEL \
  -O raw_indels.vcf.gz

# Filter INDELs
gatk VariantFiltration \
  -R reference.fasta \
  -V raw_indels.vcf.gz \
  -O filtered_indels.vcf.gz \
  --filter-name "QD_filter" --filter-expression "QD < 2.0" \
  --filter-name "FS_filter" --filter-expression "FS > 200.0" \
  --filter-name "SOR_filter" --filter-expression "SOR > 10.0"

# Merge filtered variants
gatk MergeVcfs \
  -I filtered_snps.vcf.gz \
  -I filtered_indels.vcf.gz \
  -O filtered_variants.vcf.gz

# Select PASS only
gatk SelectVariants \
  -R reference.fasta \
  -V filtered_variants.vcf.gz \
  -O final_variants.vcf.gz \
  --exclude-filtered

# 8. Variant Annotation
vep -i final_variants.vcf.gz \
  -o annotated.vcf \
  --cache --everything \
  --assembly GRCh38 \
  --vcf --fork 4

# 9. Prioritize variants
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%QUAL\t%INFO/CSQ\n' \
  annotated.vcf > variants_table.txt

# 10. Generate final report
multiqc . -o final_report/
```

---

### Workflow 3: Eukaryotic RNA-seq Analysis

```bash
#!/bin/bash
# RNA-seq variant calling workflow

# 1. Quality Control
fastqc reads_R1.fastq.gz reads_R2.fastq.gz -o qc/

# 2. Read Trimming
fastp -i reads_R1.fastq.gz -I reads_R2.fastq.gz \
  -o clean_R1.fastq.gz -O clean_R2.fastq.gz \
  --detect_adapter_for_pe \
  -h fastp_report.html

# 3. Alignment with STAR
STAR --genomeDir star_index/ \
  --readFilesIn clean_R1.fastq.gz clean_R2.fastq.gz \
  --readFilesCommand zcat \
  --outFileNamePrefix sample_ \
  --outSAMtype BAM SortedByCoordinate \
  --outSAMunmapped Within \
  --outSAMattributes NH HI AS NM MD \
  --quantMode GeneCounts \
  --runThreadN 8

# 4. Mark Duplicates
gatk MarkDuplicates \
  -I sample_Aligned.sortedByCoord.out.bam \
  -O dedup.bam \
  -M dup_metrics.txt \
  --CREATE_INDEX true

# 5. Split reads at junctions (RNA-seq specific)
gatk SplitNCigarReads \
  -R reference.fasta \
  -I dedup.bam \
  -O split.bam

# 6. Variant Calling (RNA-seq mode)
gatk HaplotypeCaller \
  -R reference.fasta \
  -I split.bam \
  -O variants.vcf.gz \
  --dont-use-soft-clipped-bases \
  --standard-min-confidence-threshold-for-calling 20

# 7. Variant Filtering (more stringent for RNA-seq)
gatk VariantFiltration \
  -R reference.fasta \
  -V variants.vcf.gz \
  -O filtered.vcf.gz \
  --filter-name "FS_filter" --filter-expression "FS > 30.0" \
  --filter-name "QD_filter" --filter-expression "QD < 2.0"

# 8. Annotation
vep -i filtered.vcf.gz \
  -o annotated.vcf \
  --cache --everything \
  --vcf --fork 4

# 9. Gene expression analysis
# Use sample_ReadsPerGene.out.tab from STAR for DE analysis
```

---

## Tool Comparison Summary

### Quick Reference Table

| **Category** | **Prokaryotes** | **Eukaryotes** | **Speed** | **Accuracy** |
|--------------|-----------------|----------------|-----------|--------------|
| **QC** | FastQC | FastQC | Fast | N/A |
| **Trimming** | fastp | Trimmomatic/fastp | Fast | Good |
| **DNA Alignment** | BWA-MEM/Bowtie2 | BWA-MEM | Fast | Excellent |
| **RNA Alignment** | N/A | STAR/HISAT2 | Medium | Excellent |
| **Deduplication** | sambamba | Picard | Medium | N/A |
| **Variant Calling** | BCFtools/FreeBayes | GATK/DeepVariant | Slow | Excellent |
| **Filtering** | BCFtools filter | GATK VQSR | Fast | Good |
| **Annotation** | SnpEff | VEP/SnpEff | Medium | Comprehensive |

---

## Best Practices Summary

### For Prokaryotes:
1. **Use simple, fast tools** - BCFtools, BWA-MEM, fastp
2. **Set ploidy to 1** for variant calling
3. **Focus on gene disruption** and AMR genes
4. **Use Snippy** for streamlined bacterial analysis
5. **Perform pan-genome analysis** with Roary for comparative studies
6. **Screen for AMR genes** with ABRicate or ARIBA
7. **Lower computational requirements** - can run on standard machines

### For Eukaryotes:
1. **Follow GATK Best Practices** for highest accuracy
2. **Use splice-aware aligners** (STAR/HISAT2) for RNA-seq
3. **Always perform BQSR** when known variants available
4. **Use VQSR instead of hard filters** for large cohorts
5. **Annotate with VEP** for clinical applications
6. **Consider population frequencies** when prioritizing variants
7. **Requires significant computational resources** - HPC recommended

### General Best Practices:
1. **Always perform QC** before and after each major step
2. **Keep detailed logs** of all commands and parameters
3. **Validate** critical findings with IGV or alternative methods
4. **Use version control** for scripts and document software versions
5. **Follow file naming conventions** for reproducibility
6. **Backup raw data** and maintain organized directory structure
7. **Document** any deviations from standard protocols

---

## Common Pitfalls to Avoid

### Data Quality Issues:
- **Skipping QC steps** - Always check quality before proceeding
- **Insufficient read depth** - Ensure adequate coverage (30x+ for variants)
- **Adapter contamination** - Trim adapters thoroughly
- **Low mapping quality** - Filter reads with MQ < 20

### Analysis Errors:
- **Wrong ploidy setting** - Set correctly for organism (1 for prokaryotes, 2 for diploid eukaryotes)
- **Missing read groups** - Add RG tags during alignment for GATK
- **Incorrect reference genome** - Use exact same reference throughout
- **Mixing genome versions** - Keep consistent (e.g., all hg38 or all hg19)

### Interpretation Issues:
- **Over-reliance on predictions** - SIFT/PolyPhen are predictions, not facts
- **Ignoring population frequencies** - Common variants unlikely to be pathogenic
- **Not validating findings** - Confirm with Sanger sequencing or alternative methods
- **Batch effects** - Account for technical variation in multi-sample studies

---

## Recommended Computational Requirements

### Minimum (Small Projects):
- **CPU:** 4-8 cores
- **RAM:** 16-32 GB
- **Storage:** 500 GB SSD
- **Suitable for:** Single bacterial genomes, small targeted sequencing

### Standard (Most Projects):
- **CPU:** 16-32 cores
- **RAM:** 64-128 GB
- **Storage:** 2-5 TB
- **Suitable for:** Multiple samples, eukaryotic WES, RNA-seq

### High-Performance (Large Projects):
- **CPU:** 64+ cores
- **RAM:** 256+ GB
- **Storage:** 10+ TB
- **Suitable for:** WGS cohorts, population studies, cancer genomics

---

## Additional Resources

### Documentation:
- **GATK Best Practices:** https://gatk.broadinstitute.org/
- **BWA Manual:** http://bio-bwa.sourceforge.net/
- **SAMtools Documentation:** http://www.htslib.org/
- **VEP Documentation:** https://www.ensembl.org/vep
- **SnpEff Manual:** https://pcingola.github.io/SnpEff/

### Databases:
- **dbSNP:** https://www.ncbi.nlm.nih.gov/snp/
- **gnomAD:** https://gnomad.broadinstitute.org/
- **ClinVar:** https://www.ncbi.nlm.nih.gov/clinvar/
- **COSMIC:** https://cancer.sanger.ac.uk/cosmic
- **CARD (AMR):** https://card.mcmaster.ca/

### Training:
- **GATK Workshops:** Broad Institute tutorials
- **Bioinformatics.org:** Free courses
- **Coursera/edX:** NGS analysis courses
- **Galaxy Project:** Web-based training

---

## Conclusion

This documentation provides a comprehensive guide to NGS analysis for both prokaryotic and eukaryotic organisms. The key to successful analysis is:

1. **Understand your organism** - Choose appropriate tools and parameters
2. **Maintain data quality** - QC at every step
3. **Use established workflows** - Follow best practices
4. **Validate results** - Don't trust blindly
5. **Document everything** - Ensure reproducibility

Remember that bioinformatics is both art and science - parameters may need adjustment based on your specific data and research questions. Always critically evaluate results and consult the literature for the latest methods and recommendations.
