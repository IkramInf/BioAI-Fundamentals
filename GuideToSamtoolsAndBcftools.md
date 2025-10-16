# Comprehensive Guide to SAMtools and BCFtools

## Overview

**SAMtools** and **BCFtools** are companion tools from the same suite for manipulating high-throughput sequencing data. SAMtools focuses on sequence alignment files (SAM/BAM/CRAM), while BCFtools specializes in variant call files (VCF/BCF). They share similar command structures and philosophies.

---

## Table of Contents

1. [File Format Conversion & Compression](#1-file-format-conversion--compression)
2. [File Indexing](#2-file-indexing)
3. [File Viewing & Statistics](#3-file-viewing--statistics)
4. [Filtering & Subsetting](#4-filtering--subsetting)
5. [Sorting & Merging](#5-sorting--merging)
6. [Variant Calling](#6-variant-calling)
7. [Normalization & Annotation](#7-normalization--annotation)
8. [Quality Control & Coverage](#8-quality-control--coverage)
9. [Consensus & Reference Operations](#9-consensus--reference-operations)
10. [Unique Commands by Tool](#10-unique-commands-by-tool)

---

## 1. File Format Conversion & Compression

### SAMtools: SAM ↔ BAM ↔ CRAM Conversion

**SAM to BAM (compressed binary):**

```bash
samtools view -bS input.sam -o output.bam
# -b: output BAM format
# -S: input is SAM (auto-detected in newer versions)
```

**BAM to SAM (human-readable):**

```bash
samtools view -h input.bam -o output.sam
# -h: include header
```

**BAM to CRAM (reference-based compression, smaller size):**

```bash
samtools view -C -T reference.fasta input.bam -o output.cram
# -C: output CRAM format
# -T: reference genome (required for CRAM)
```

**CRAM to BAM:**

```bash
samtools view -b -T reference.fasta input.cram -o output.bam
```

### BCFtools: VCF ↔ BCF Conversion

**VCF to BCF (compressed binary):**

```bash
bcftools view -O b input.vcf.gz -o output.bcf
# -O b: output BCF format (b=BCF, z=compressed VCF, u=uncompressed BCF, v=uncompressed VCF)
```

**BCF to VCF:**

```bash
bcftools view -O z input.bcf -o output.vcf.gz
```

**Compress and index VCF:**

```bash
bgzip input.vcf          # creates input.vcf.gz
bcftools index input.vcf.gz
```

### Comparison & Recommendations

- **SAMtools**: More optimized for alignment format conversions. CRAM provides ~50% size reduction over BAM.
- **BCFtools**: BCF format is ~30% smaller than compressed VCF and faster to process.
- **Best Practice**: Use CRAM for long-term storage; BCF for intermediate variant processing.

---

## 2. File Indexing

### SAMtools Index

**Index BAM file:**

```bash
samtools index input.bam              # creates input.bam.bai
samtools index -b input.bam output.bai  # specify output name
```

**Index CRAM file:**

```bash
samtools index input.cram             # creates input.cram.crai
```

**Build CSI index (for very large chromosomes >512 Mbp):**

```bash
samtools index -c input.bam           # creates .csi index instead of .bai
```

### BCFtools Index

**Index VCF/BCF:**

```bash
bcftools index input.vcf.gz           # creates .csi index by default
bcftools index -t input.vcf.gz        # creates .tbi (tabix) index
bcftools index --threads 4 input.vcf.gz
```

### Comparison

- **SAMtools**: BAI index (default) works for chromosomes <512 Mbp; use CSI for larger.
- **BCFtools**: CSI index (default) works for any chromosome size; TBI for backward compatibility.
- **Performance**: Both are comparable; CSI is more flexible.

---

## 3. File Viewing & Statistics

### SAMtools View & Stats

**View alignments in a region:**

```bash
samtools view input.bam chr1:1000-2000
samtools view -h input.bam chr1:1000-2000  # with header
```

**Count alignments:**

```bash
samtools view -c input.bam                # total reads
samtools view -c -F 4 input.bam           # mapped reads only
samtools view -c -f 4 input.bam           # unmapped reads only
```

**Alignment statistics:**

```bash
samtools flagstat input.bam
# Output: total reads, mapped, paired, properly paired, etc.

samtools stats input.bam > stats.txt
# Detailed statistics including quality scores, insert sizes

samtools idxstats input.bam
# Per-chromosome mapping statistics
```

**Coverage information:**

```bash
samtools depth input.bam                  # per-base depth
samtools depth -r chr1:1000-2000 input.bam  # region-specific
samtools coverage input.bam               # coverage summary per chromosome
```

### BCFtools View & Stats

**View variants:**

```bash
bcftools view input.vcf.gz                # all variants
bcftools view -H input.vcf.gz             # without header
bcftools view -r chr1:1000-2000 input.vcf.gz  # specific region
```

**Count variants:**

```bash
bcftools view -H input.vcf.gz | wc -l    # total variants
```

**Variant statistics:**

```bash
bcftools stats input.vcf.gz > stats.txt
# Detailed statistics: ts/tv ratio, indel distribution, quality scores

bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' input.vcf.gz
# Custom format output
```

### Comparison

- **SAMtools**: Superior for alignment QC with `flagstat`, `stats`, and `coverage` commands.
- **BCFtools**: Better for variant-level statistics (ts/tv ratio, allele frequencies).
- **Accuracy**: Both are highly accurate for their respective data types.

---

## 4. Filtering & Subsetting

### SAMtools Filter

**Filter by mapping quality:**

```bash
samtools view -q 30 input.bam -o high_quality.bam
# -q: minimum mapping quality
```

**Filter by flags:**

```bash
samtools view -F 4 input.bam -o mapped.bam        # exclude unmapped
samtools view -f 2 input.bam -o properly_paired.bam  # properly paired only
samtools view -F 1804 input.bam -o filtered.bam   # exclude unmapped, secondary, PCR duplicates
```

**Extract specific chromosomes:**

```bash
samtools view -b input.bam chr1 chr2 -o subset.bam
```

**Subsample reads:**

```bash
samtools view -s 0.5 input.bam -o half_reads.bam  # 50% random subsample
samtools view -s 42.1 input.bam -o subset.bam     # seed 42, 10% of reads
```

### BCFtools Filter

**Filter by quality:**

```bash
bcftools view -i 'QUAL>30' input.vcf.gz -o filtered.vcf.gz
bcftools view -e 'QUAL<30' input.vcf.gz -o excluded.vcf.gz
```

**Filter by type:**

```bash
bcftools view -v snps input.vcf.gz -o snps.vcf.gz      # SNPs only
bcftools view -v indels input.vcf.gz -o indels.vcf.gz  # indels only
```

**Filter by allele frequency:**

```bash
bcftools view -i 'AF>0.01' input.vcf.gz -o common.vcf.gz
bcftools view -i 'INFO/DP>10 & QUAL>30' input.vcf.gz -o high_conf.vcf.gz
```

**Filter by genotype:**

```bash
bcftools view -i 'GT="1/1"' input.vcf.gz -o homozygous.vcf.gz
bcftools view -g ^miss input.vcf.gz -o no_missing.vcf.gz
```

**Region-based filtering:**

```bash
bcftools view -r chr1,chr2 input.vcf.gz -o subset.vcf.gz
bcftools view -R regions.bed input.vcf.gz -o filtered.vcf.gz  # from BED file
bcftools view -t ^chr1:1000-2000 input.vcf.gz -o excluded.vcf.gz  # exclude region
```

### Comparison

- **SAMtools**: Optimized for read-level filtering (quality, flags, sampling).
- **BCFtools**: More powerful for variant filtering with complex expressions.
- **Flexibility**: BCFtools has more sophisticated filtering language.

---

## 5. Sorting & Merging

### SAMtools Sort & Merge

**Sort by coordinate:**

```bash
samtools sort input.bam -o sorted.bam
samtools sort -@ 8 -m 2G input.bam -o sorted.bam
# -@: threads
# -m: memory per thread
```

**Sort by read name:**

```bash
samtools sort -n input.bam -o name_sorted.bam
```

**Merge multiple BAM files:**

```bash
samtools merge output.bam input1.bam input2.bam input3.bam
samtools merge -@ 8 output.bam *.bam
```

**Merge with region:**

```bash
samtools merge -R chr1:1000-2000 output.bam input1.bam input2.bam
```

### BCFtools Sort & Merge

**Sort VCF file:**

```bash
bcftools sort input.vcf.gz -o sorted.vcf.gz
bcftools sort -O b input.vcf.gz -o sorted.bcf
```

**Merge multiple VCF files (same samples, different regions):**

```bash
bcftools concat file1.vcf.gz file2.vcf.gz -o merged.vcf.gz
bcftools concat -a file*.vcf.gz -o merged.vcf.gz  # allow overlaps
```

**Merge multiple VCF files (different samples, same regions):**

```bash
bcftools merge sample1.vcf.gz sample2.vcf.gz -o merged.vcf.gz
bcftools merge -m none input*.vcf.gz -o merged.vcf.gz  # only matching positions
```

### Comparison

- **SAMtools sort**: Highly optimized with multi-threading; can sort by name or coordinate.
- **BCFtools sort**: Coordinate-based sorting for variants.
- **Merging**: `samtools merge` for combining alignment files; `bcftools concat` for regions, `bcftools merge` for samples.
- **Performance**: SAMtools sort is faster with multi-threading support.

---

## 6. Variant Calling

### SAMtools mpileup (Legacy)

**Generate pileup:**

```bash
samtools mpileup -f reference.fasta input.bam > output.pileup
```

**Call variants (old method, deprecated):**

```bash
samtools mpileup -uf reference.fasta input.bam | bcftools call -mv -o output.vcf
```

### BCFtools mpileup & call (Recommended)

**Call variants (modern approach):**

```bash
bcftools mpileup -f reference.fasta input.bam | bcftools call -mv -Oz -o output.vcf.gz
# -m: multiallelic caller (recommended)
# -v: output variants only
# -O z: compressed VCF output
```

**Multi-sample calling:**

```bash
bcftools mpileup -f reference.fasta sample1.bam sample2.bam | \
  bcftools call -mv -Oz -o multi_sample.vcf.gz
```

**Advanced calling with quality filters:**

```bash
bcftools mpileup -Ou -f reference.fasta input.bam | \
  bcftools call -mv -Ou | \
  bcftools filter -s LowQual -e 'QUAL<20 || DP<10' -Oz -o filtered.vcf.gz
# -Ou: uncompressed BCF for piping (faster)
# -s: soft filter (marks variants, doesn't remove)
```

**Consensus caller (alternative):**

```bash
bcftools mpileup -f reference.fasta input.bam | \
  bcftools call -c -Oz -o consensus.vcf.gz
# -c: consensus caller (older, faster but less accurate)
```

### Comparison

- **BCFtools mpileup**: Replaced `samtools mpileup` for variant calling.
- **Multiallelic caller (-m)**: More accurate than consensus caller (-c).
- **Recommendation**: Use BCFtools for all variant calling; SAMtools for alignment operations.

---

## 7. Normalization & Annotation

### BCFtools norm (Unique to BCFtools)

**Left-align and normalize indels:**

```bash
bcftools norm -f reference.fasta input.vcf.gz -o normalized.vcf.gz
# Essential for comparing variants from different callers
```

**Split multiallelic sites:**

```bash
bcftools norm -m- input.vcf.gz -o split.vcf.gz
# -m-: split multiallelic sites into biallelic records
```

**Join biallelic sites:**

```bash
bcftools norm -m+ input.vcf.gz -o joined.vcf.gz
```

**Remove duplicates:**

```bash
bcftools norm -d all input.vcf.gz -o deduped.vcf.gz
```

**Complete normalization pipeline:**

```bash
bcftools norm -f reference.fasta -m-both -d all input.vcf.gz -o clean.vcf.gz
```

### BCFtools annotate

**Add or remove INFO fields:**

```bash
bcftools annotate -x INFO/DP input.vcf.gz -o no_dp.vcf.gz  # remove field
bcftools annotate -a annotations.vcf.gz -c INFO input.vcf.gz -o annotated.vcf.gz  # add
```

**Add sample information:**

```bash
bcftools annotate -s sample_name input.vcf.gz -o named.vcf.gz
```

**Rename chromosomes:**

```bash
bcftools annotate --rename-chrs chr_name_mapping.txt input.vcf.gz -o renamed.vcf.gz
```

### Comparison

- **BCFtools norm**: No SAMtools equivalent; essential for variant standardization.
- **Accuracy**: Left-alignment critical for indel comparison across datasets.
- **Unique**: BCFtools annotation framework has no SAMtools parallel.

---

## 8. Quality Control & Coverage

### SAMtools QC Tools

**Flag statistics:**

```bash
samtools flagstat input.bam
# Shows: total reads, duplicates, mapped, paired, singletons
```

**Detailed statistics:**

```bash
samtools stats input.bam > stats.txt
# Includes: insert size distribution, GC content, error rates
```

**Per-base coverage:**

```bash
samtools depth input.bam > depth.txt
samtools depth -a input.bam > depth_all.txt  # include zero coverage
```

**Coverage summary:**

```bash
samtools coverage input.bam
# Per-chromosome: coverage, mean depth, %covered
```

**Pileup format (detailed base-level view):**

```bash
samtools mpileup input.bam > pileup.txt
# Shows base calls at each position
```

### BCFtools QC Tools

**Variant statistics:**

```bash
bcftools stats input.vcf.gz > vcf_stats.txt
# Includes: ts/tv ratio, indel stats, quality distribution
```

**Compare VCF files:**

```bash
bcftools stats -c input1.vcf.gz input2.vcf.gz > comparison.txt
```

**Custom queries:**

```bash
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%QUAL\t[%GT]\n' input.vcf.gz
# Extract specific fields
```

**ROH (Runs of Homozygosity) detection:**

```bash
bcftools roh input.vcf.gz -o roh.txt
```

### Comparison

- **SAMtools**: Comprehensive alignment QC; better for coverage analysis.
- **BCFtools**: Specialized variant QC; unique ROH detection.
- **Recommendation**: Use both in pipeline; SAMtools pre-calling, BCFtools post-calling.

---

## 9. Consensus & Reference Operations

### SAMtools Operations

**Generate consensus FASTA:**

```bash
samtools consensus input.bam -f reference.fasta -o consensus.fasta
# Newer samtools versions
```

**Extract FASTA from reference:**

```bash
samtools faidx reference.fasta
samtools faidx reference.fasta chr1:1000-2000 > region.fasta
```

**Create reference index:**

```bash
samtools faidx reference.fasta  # creates .fai index
```

### BCFtools consensus

**Generate consensus from VCF:**

```bash
bcftools consensus -f reference.fasta input.vcf.gz > consensus.fasta
# Applies variants to reference
```

**Generate consensus for specific sample:**

```bash
bcftools consensus -f reference.fasta -s sample_name input.vcf.gz > sample_consensus.fasta
```

**Apply only high-quality variants:**

```bash
bcftools view -i 'QUAL>30' input.vcf.gz | \
  bcftools consensus -f reference.fasta > high_qual_consensus.fasta
```

**Mask low-quality regions:**

```bash
bcftools consensus -f reference.fasta -m mask.bed input.vcf.gz > masked_consensus.fasta
```

### BCFtools gtcheck (Genotype checking)

**Check sample concordance:**

```bash
bcftools gtcheck -g genotypes.vcf.gz input.vcf.gz
# Useful for detecting sample swaps
```

### Comparison

- **SAMtools consensus**: From alignments directly.
- **BCFtools consensus**: From variant calls; more flexible with filtering.
- **Use case**: BCFtools consensus preferred for incorporating validated variants.

---

## 10. Unique Commands by Tool

### Unique to SAMtools

#### fixmate

**Fix mate-pair information:**

```bash
samtools sort -n input.bam | samtools fixmate -m - - | \
  samtools sort - | samtools markdup - output.bam
# Required before markdup
```

#### markdup

**Mark or remove PCR duplicates:**

```bash
samtools markdup input.bam output.bam
samtools markdup -r input.bam deduped.bam  # remove duplicates
samtools markdup -s input.bam deduped.bam  # report statistics
```

#### collate

**Collate reads by name:**

```bash
samtools collate -o output.bam input.bam prefix
# Faster than full name sort for some applications
```

#### fastq/fasta

**Convert BAM to FASTQ:**

```bash
samtools fastq input.bam -1 R1.fastq -2 R2.fastq -s singles.fastq
samtools fasta input.bam > output.fasta
```

#### split

**Split BAM by read group:**

```bash
samtools split -f '%*_%!.bam' input.bam
```

#### addreplacerg

**Add or replace read groups:**

```bash
samtools addreplacerg -r '@RG\tID:sample1\tSM:sample1' input.bam -o output.bam
```

#### calmd

**Calculate MD tag:**

```bash
samtools calmd -b input.bam reference.fasta > output.bam
# Adds mismatch positions in MD tag
```

#### reheader

**Replace header:**

```bash
samtools reheader new_header.sam input.bam > output.bam
```

#### cat

**Concatenate BAM files (no sorting):**

```bash
samtools cat -o output.bam input1.bam input2.bam
```

#### quickcheck

**Validate BAM integrity:**

```bash
samtools quickcheck input.bam && echo "OK" || echo "FAIL"
samtools quickcheck -v *.bam
```

### Unique to BCFtools

#### isec

**Find intersections between VCF files:**

```bash
bcftools isec file1.vcf.gz file2.vcf.gz -p isec_output
# Creates separate files for unique and shared variants
```

**Extract private variants:**

```bash
bcftools isec -C file1.vcf.gz file2.vcf.gz > unique_to_file1.vcf
```

#### plugin framework

**Various plugins for specialized tasks:**

**Fill tags (calculate AF, AC, AN):**

```bash
bcftools +fill-tags input.vcf.gz -o tagged.vcf.gz
```

**Impute missing genotypes:**

```bash
bcftools +impute-info input.vcf.gz -o imputed.vcf.gz
```

**Set genotypes:**

```bash
bcftools +setGT input.vcf.gz -- -t q -n . -i 'FMT/DP<10' -o filtered.vcf.gz
# Set low-depth genotypes to missing
```

**Split multiallelic sites:**

```bash
bcftools +split-vep input.vcf.gz -o split.vcf.gz
```

**Calculate trio information:**

```bash
bcftools +trio-dnm2 -p proband,father,mother input.vcf.gz -o denovo.vcf.gz
```

#### reheader

**Replace samples or contigs:**

```bash
bcftools reheader -s sample_names.txt input.vcf.gz -o renamed.vcf.gz
```

#### convert

**Convert between formats:**

```bash
bcftools convert --gvcf2vcf input.gvcf.gz -o output.vcf.gz
bcftools convert --haplegendsample2vcf legend.gz samples > output.vcf
```

#### +fixploidy

**Fix ploidy issues:**

```bash
bcftools +fixploidy input.vcf.gz -- -p ploidy.txt -o fixed.vcf.gz
```

#### +prune

**LD-based pruning:**

```bash
bcftools +prune -l 0.8 -w 1000kb input.vcf.gz -o pruned.vcf.gz
```

#### +contrast

**Identify private alleles between groups:**

```bash
bcftools +contrast -s case.txt -c control.txt input.vcf.gz -o contrast.vcf.gz
```

---

## Performance Comparison Summary

| Task | SAMtools | BCFtools | Recommendation |
|------|----------|----------|----------------|
| Format conversion | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Equal performance |
| Sorting | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | SAMtools (multi-threading) |
| Filtering | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | BCFtools (more flexible) |
| QC/Statistics | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | SAMtools (alignment QC) |
| Variant calling | N/A | ⭐⭐⭐⭐⭐ | BCFtools only |
| Normalization | N/A | ⭐⭐⭐⭐⭐ | BCFtools only |
| Consensus generation | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | BCFtools (from variants) |
| Set operations | N/A | ⭐⭐⭐⭐⭐ | BCFtools only |

---

## Best Practices

### Typical Alignment Processing Pipeline

```bash
# 1. Sort by coordinate
samtools sort -@ 8 -m 2G raw.bam -o sorted.bam

# 2. Mark duplicates
samtools markdup sorted.bam deduped.bam

# 3. Index
samtools index deduped.bam

# 4. QC
samtools flagstat deduped.bam > flagstat.txt
samtools stats deduped.bam > stats.txt
samtools coverage deduped.bam > coverage.txt
```

### Typical Variant Processing Pipeline

```bash
# 1. Call variants
bcftools mpileup -Ou -f reference.fasta input.bam | \
  bcftools call -mv -Ou | \
  bcftools filter -s LowQual -e 'QUAL<20 || DP<10' -Oz -o raw.vcf.gz

# 2. Index
bcftools index raw.vcf.gz

# 3. Normalize
bcftools norm -f reference.fasta -m-both -Oz -o normalized.vcf.gz raw.vcf.gz
bcftools index normalized.vcf.gz

# 4. Filter
bcftools view -i 'QUAL>30 && INFO/DP>10' normalized.vcf.gz -Oz -o filtered.vcf.gz

# 5. Statistics
bcftools stats filtered.vcf.gz > vcf_stats.txt

# 6. Generate consensus
bcftools consensus -f reference.fasta filtered.vcf.gz > consensus.fasta
```

---

## Installation & Version Check

```bash
# Install via conda
conda install -c bioconda samtools bcftools

# Check versions
samtools --version
bcftools --version

# Get help
samtools help
bcftools help
samtools view --help
bcftools view --help
```

---

## Key Takeaways

1. **Use SAMtools for**: Alignment manipulation, format conversion, coverage analysis, duplicate marking
2. **Use BCFtools for**: Variant calling, VCF manipulation, variant filtering, normalization, set operations
3. **Both work together**: Alignments (SAMtools) → Variants (BCFtools) → Consensus (BCFtools)
4. **File formats**: Prefer CRAM for storage, BAM for processing, BCF for variant processing
5. **Always index**: Both tools require indexed files for region-based queries
6. **Multi-threading**: Use `-@ threads` for better performance
7. **Piping**: Both support piping with `-O u` (uncompressed output) for efficiency

This documentation covers the most common use cases. Both tools have additional advanced features available through `--help` and their respective manual pages.
