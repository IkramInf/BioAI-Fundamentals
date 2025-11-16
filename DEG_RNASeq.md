# Differential Gene Expression Analysis Using RNA-Seq: A Complete Tutorial

**Author:** Bioinformatics Engineering Team  
**Last Updated:** November 2025  
**Target Audience:** Bioinformatics engineers and computational biologists

---

## Table of Contents
1. [Introduction](#introduction)
2. [RNA-Seq vs Microarray](#rna-seq-vs-microarray)
3. [Theoretical Background](#theoretical-background)
4. [Tools Selection & Rationale](#tools-selection)
5. [Practical Tutorial](#practical-tutorial)
6. [Results Interpretation](#results-interpretation)
7. [Quality Control & Troubleshooting](#quality-control)
8. [Conclusion](#conclusion)

---

## Introduction

This tutorial walks through a complete differential gene expression (DGE) analysis using RNA-seq count data. We'll analyze a real dataset comparing treated vs control samples, covering everything from raw counts to publication-ready results.

**What you'll learn:**
- How to properly handle RNA-seq count data
- Statistical methods for DGE analysis
- Parameter selection and optimization
- Quality control checkpoints
- Result interpretation and validation

**Prerequisites:**
- Basic R programming
- Understanding of statistical concepts (p-values, FDR)
- Familiarity with genomics terminology

---

## RNA-Seq vs Microarray: Key Differences

Before diving in, understand why RNA-seq analysis differs fundamentally from microarray analysis.

### Microarray Data Characteristics
- **Continuous intensity values** (fluorescence)
- **Normally distributed** (or log-normal after transformation)
- **Limited dynamic range** (saturation at high expression)
- **Background noise** requires normalization
- **Analysis approach:** Linear models, t-tests on log-transformed data

### RNA-Seq Count Data Characteristics
- **Discrete count values** (integers representing reads)
- **Non-normally distributed** (follows negative binomial distribution)
- **Large dynamic range** (0 to millions of reads)
- **Overdispersed** (variance > mean, unlike Poisson)
- **Analysis approach:** Generalized linear models (GLMs) with proper distribution assumptions

### Why This Matters for Analysis

**Key Difference #1: Distribution Assumptions**
```
Microarray: log2(intensity) ~ Normal(μ, σ²)
RNA-Seq: counts ~ NegativeBinomial(μ, dispersion)
```

Using standard t-tests on RNA-seq counts is statistically invalid because:
1. Counts are discrete, not continuous
2. Variance scales with mean (heteroscedastic)
3. Low counts have different uncertainty than high counts

**Key Difference #2: Zero Inflation**
RNA-seq data contains many true zeros (genes not expressed), unlike microarray where every probe gives a signal. This requires specialized handling.

**Key Difference #3: Sequencing Depth Effects**
Each sample may have different total read counts (library size). A gene with 100 counts in a 10M read library is different from 100 counts in a 50M read library. Microarrays don't have this issue because all samples are measured on the same chip.

---

## Theoretical Background

### The Count Data Problem

RNA-seq counts follow a **negative binomial distribution** because:

1. **Mean-variance relationship:** For gene *i* in sample *j*:
   ```
   Var(K_ij) = μ_ij + α_i * μ_ij²
   ```
   where α_i is the dispersion parameter. This overdispersion (variance > mean) comes from biological variability between replicates.

2. **Why not Poisson?** Poisson assumes variance = mean, but real RNA-seq data shows variance >> mean due to:
   - Biological variability between samples
   - Technical variability in library prep
   - Variation in cell populations

### Statistical Model for DGE

Most modern DGE tools use a **generalized linear model (GLM)** framework:

```
log(μ_ij) = Σ x_jr * β_ir
```

Where:
- μ_ij: expected count for gene i in sample j
- x_jr: design matrix (experimental conditions)
- β_ir: regression coefficients (log fold changes)

### Multiple Testing Problem

With ~20,000 genes tested, we expect ~1,000 false positives at p < 0.05. We correct using:

- **FDR (False Discovery Rate):** Controls the proportion of false positives among discoveries. More powerful than Bonferroni.
- **Benjamini-Hochberg procedure:** Most common FDR method, balances false positives vs sensitivity.

**Rule of thumb:** Use FDR < 0.05 for exploratory analysis, FDR < 0.01 for confirmatory studies.

---

## Tools Selection & Rationale

### Why DESeq2?

We'll use **DESeq2** for this tutorial. Here's why:

**Pros:**
- Gold standard in the field (>30,000 citations)
- Robust dispersion estimation via shrinkage
- Handles complex experimental designs
- Built-in outlier detection
- Conservative (lower false positive rate)
- Excellent documentation and community support

**Cons:**
- Slower than edgeR for very large datasets (>500 samples)
- Requires raw integer counts (no normalized input)
- Memory intensive for datasets with >50,000 features

### Alternative Tools Comparison

| Tool | Best For | Dispersion Method | Speed | Main Limitation |
|------|----------|-------------------|-------|-----------------|
| **DESeq2** | Most studies, robust results | Shrinkage estimation | Moderate | Memory usage |
| **edgeR** | Large datasets, simple designs | Tagwise dispersion | Fast | Less robust with small n |
| **limma-voom** | Microarray-like workflow | Precision weights | Fast | Requires good quality data |
| **NOISeq** | No replicates | Non-parametric | Slow | Lower statistical power |

**When to choose alternatives:**
- **edgeR:** >500 samples, need speed, willing to accept slightly higher FDR
- **limma-voom:** Converting from microarray pipeline, need speed
- **NOISeq:** Exploratory analysis without replicates (but seriously, get replicates)

---

## Practical Tutorial

### Use Case: Drug Treatment Response Study

**Experimental Design:**
- 6 control samples (untreated cells)
- 6 treated samples (drug X, 24h)
- Goal: Identify genes responding to treatment
- Organism: *Homo sapiens* (human)

### Step 1: Environment Setup

```r
# Install packages (run once)
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("DESeq2", "apeglm", "org.Hs.eg.db", 
                       "AnnotationDbi", "pheatmap", "RColorBrewer"))

# Load libraries
library(DESeq2)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(AnnotationDbi)
library(org.Hs.eg.db)

# Set working directory
setwd("~/rna_seq_analysis")

# Set seed for reproducibility
set.seed(42)
```

### Step 2: Load Real Count Data

We'll use a subset of the airway dataset (glucocorticoid treatment of airway smooth muscle cells).

```r
# Load example data
library(airway)
data(airway)

# Extract count matrix and metadata
countData <- assay(airway)
colData <- colData(airway)

# Simplify for this tutorial: focus on treated vs untreated
colData <- colData[colData$cell == "N61311", ]
countData <- countData[, rownames(colData)]

# Rename conditions for clarity
colData$condition <- factor(colData$dex, levels = c("untrt", "trt"))

# Inspect the data
head(countData[, 1:4])
```

**Expected output:**
```
                N61311_untrt_1 N61311_trt_1 N61311_untrt_2 N61311_trt_2
ENSG00000000003           679          448           257          515
ENSG00000000005             0            0             0            0
ENSG00000000419           467          515           322          524
ENSG00000000457           260          211           233          207
```

**What we have:**
- Rows = genes (Ensembl IDs)
- Columns = samples
- Values = raw read counts (integers)

### Step 3: Create DESeq2 Object

```r
# Create DESeqDataSet object
dds <- DESeqDataSetFromMatrix(
    countData = countData,
    colData = colData,
    design = ~ condition
)

# Check object
dds
```

**Parameter explanation: design formula**

The `design = ~ condition` formula specifies our statistical model.

- **Simple designs:** `~ condition` (two-group comparison)
- **Paired samples:** `~ patient + condition` (accounts for patient-specific effects)
- **Multiple factors:** `~ batch + condition` (controls for batch effects)
- **Interactions:** `~ genotype + treatment + genotype:treatment`

**Why this matters:** The design formula determines:
1. Which comparisons are tested
2. What covariates are controlled for
3. The statistical power of your analysis

**Common mistake:** Using `~ 1` (intercept-only) by mistake, which tests nothing.

### Step 4: Pre-filtering

```r
# Remove genes with very low counts
# Keep genes with at least 10 reads total across all samples
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep, ]

# Check how many genes remain
nrow(dds)
```

**Why pre-filter?**

**Pros:**
- Reduces memory usage (important for large datasets)
- Speeds up computation
- Reduces multiple testing burden (more power)
- Removes genes with no information content

**Cons:**
- Arbitrary threshold may remove true low-expression genes
- Could bias results if filtering is too aggressive

**Parameter choice: Why ≥10 reads?**

This is conservative. Alternatives:
- **≥1 read in ≥3 samples:** More lenient, keeps more genes
- **≥5 reads in ≥50% samples:** More stringent
- **CPM-based:** `rowSums(cpm(counts(dds)) > 1) >= 3`

**Rule of thumb:** Filter genes with counts too low to be statistically meaningful. For 6 samples, ≥10 total reads means average ≥1.67 reads/sample, which is reasonable.

### Step 5: Run DESeq2 Pipeline

```r
# Run the full DESeq2 analysis
dds <- DESeq(dds)
```

**What happens inside DESeq():**

1. **Estimation of size factors** (normalization for library size)
   - Uses median-of-ratios method
   - Accounts for sequencing depth differences
   - More robust than simple total count normalization

2. **Estimation of dispersion** (gene-wise variability)
   - Gene-wise dispersion estimates
   - Shrinkage toward fitted trend
   - Handles variance heterogeneity

3. **Negative binomial GLM fitting**
   - Tests for differential expression
   - Wald test by default (or LRT if specified)

**Alternative: Likelihood Ratio Test (LRT)**
```r
# For testing multiple conditions or time courses
dds_lrt <- DESeq(dds, test = "LRT", reduced = ~ 1)
```

**When to use LRT:**
- Comparing nested models
- Testing overall effect across multiple levels
- Time course experiments

**When to use Wald test (default):**
- Two-group comparisons
- Need fold change estimates
- Most common scenarios

### Step 6: Extract Results

```r
# Get results for treated vs untreated
res <- results(dds, contrast = c("condition", "trt", "untrt"))

# View results structure
head(res)
```

**Parameter explanation: contrast**

```r
contrast = c("condition", "trt", "untrt")
```

This specifies: (treatment) - (control), so:
- **Positive log2FC:** Gene upregulated in treatment
- **Negative log2FC:** Gene downregulated in treatment

**Alternative specification methods:**
```r
# By name (if you have >2 groups)
results(dds, name = "condition_trt_vs_untrt")

# By coefficient number
results(dds, coef = 2)
```

### Step 7: Log Fold Change Shrinkage

```r
# Apply shrinkage for better effect size estimates
library(apeglm)
res_shrink <- lfcShrink(dds, 
                        coef = "condition_trt_vs_untrt",
                        type = "apeglm")

# Compare before and after
par(mfrow = c(1, 2))
plotMA(res, main = "Before shrinkage", ylim = c(-5, 5))
plotMA(res_shrink, main = "After shrinkage", ylim = c(-5, 5))
```

**Why shrinkage?**

Low-count genes have unreliable fold change estimates. Shrinkage pulls extreme fold changes toward zero for genes with high uncertainty.

**Benefits:**
- More accurate effect size estimates
- Better for ranking genes
- Reduces false positives from noisy low-count genes

**Shrinkage methods:**

| Method | Speed | Accuracy | When to Use |
|--------|-------|----------|-------------|
| **apeglm** | Slow | Best | Publication, need accurate FC |
| **ashr** | Fast | Good | Large datasets |
| **normal** | Fastest | Decent | Quick exploration |

**Important:** Shrinkage only affects log2FC, not p-values. Use for visualization and ranking, but original results for significance testing.

### Step 8: Add Gene Annotations

```r
# Add gene symbols and descriptions
res_shrink$symbol <- mapIds(org.Hs.eg.db,
                            keys = row.names(res_shrink),
                            column = "SYMBOL",
                            keytype = "ENSEMBL",
                            multiVals = "first")

res_shrink$entrez <- mapIds(org.Hs.eg.db,
                            keys = row.names(res_shrink),
                            column = "ENTREZID",
                            keytype = "ENSEMBL",
                            multiVals = "first")

res_shrink$name <- mapIds(org.Hs.eg.db,
                          keys = row.names(res_shrink),
                          column = "GENENAME",
                          keytype = "ENSEMBL",
                          multiVals = "first")
```

### Step 9: Filter Significant Genes

```r
# Define significance thresholds
padj_cutoff <- 0.05
lfc_cutoff <- 1  # |log2FC| > 1 means >2-fold change

# Filter significant genes
sig_genes <- subset(res_shrink, 
                    padj < padj_cutoff & abs(log2FoldChange) > lfc_cutoff)

# Sort by adjusted p-value
sig_genes_sorted <- sig_genes[order(sig_genes$padj), ]

# View top results
head(as.data.frame(sig_genes_sorted))

# Count up/downregulated genes
summary(sig_genes$log2FoldChange > 0)
```

**Parameter choice: Thresholds**

**FDR (padj) threshold:**
- **0.05:** Standard for exploratory analysis
- **0.01:** More stringent for confirmatory studies
- **0.10:** Acceptable for hypothesis generation

**Log2FC threshold:**
- **0.5:** Small changes (1.4-fold)
- **1.0:** Moderate changes (2-fold) - **recommended**
- **1.5:** Large changes (2.8-fold) - very stringent

**Why both?**

Statistical significance (p-value) ≠ biological significance (fold change).

Example: A gene changing from 10,000 to 10,200 reads might be significant (p < 0.001) but biologically meaningless (1.02-fold). Conversely, a gene changing 5-fold with high variance might not reach significance but could be biologically important.

### Step 10: Export Results

```r
# Create results directory
dir.create("results", showWarnings = FALSE)

# Export all results
write.csv(as.data.frame(res_shrink), 
          file = "results/all_genes_results.csv",
          row.names = TRUE)

# Export significant genes only
write.csv(as.data.frame(sig_genes_sorted),
          file = "results/significant_genes.csv",
          row.names = TRUE)

# Create summary statistics
summary_stats <- data.frame(
    Total_genes = nrow(res_shrink),
    Significant = sum(res_shrink$padj < padj_cutoff, na.rm = TRUE),
    Upregulated = sum(res_shrink$padj < padj_cutoff & 
                     res_shrink$log2FoldChange > lfc_cutoff, na.rm = TRUE),
    Downregulated = sum(res_shrink$padj < padj_cutoff & 
                       res_shrink$log2FoldChange < -lfc_cutoff, na.rm = TRUE)
)

write.csv(summary_stats, 
          file = "results/analysis_summary.csv",
          row.names = FALSE)

print(summary_stats)
```

---

## Quality Control & Troubleshooting

### QC Step 1: Check Sample Clustering

```r
# Variance stabilizing transformation for visualization
vsd <- vst(dds, blind = FALSE)

# Sample distance heatmap
sampleDists <- dist(t(assay(vsd)))
sampleDistMatrix <- as.matrix(sampleDists)

colors <- colorRampPalette(rev(brewer.pal(9, "Blues")))(255)
pheatmap(sampleDistMatrix,
         clustering_distance_rows = sampleDists,
         clustering_distance_cols = sampleDists,
         col = colors,
         main = "Sample-to-Sample Distances")
```

**What to look for:**
- ✅ **Good:** Samples cluster by treatment group
- ⚠️ **Warning:** One sample doesn't cluster with its group (potential outlier)
- ❌ **Bad:** Random clustering (batch effects, sample swaps)

### QC Step 2: PCA Plot

```r
# Principal Component Analysis
pcaData <- plotPCA(vsd, intgroup = "condition", returnData = TRUE)
percentVar <- round(100 * attr(pcaData, "percentVar"))

ggplot(pcaData, aes(x = PC1, y = PC2, color = condition, label = name)) +
    geom_point(size = 4) +
    geom_text(hjust = 0.5, vjust = -1) +
    xlab(paste0("PC1: ", percentVar[1], "% variance")) +
    ylab(paste0("PC2: ", percentVar[2], "% variance")) +
    theme_bw() +
    ggtitle("PCA: Samples by Treatment Condition")
```

**Interpretation:**
- **PC1 should separate treatment groups** (if treatment has major effect)
- **PC2 might show batch or biological variation**
- **Outliers:** Samples far from their group need investigation

**Troubleshooting:**
```r
# If PC1 doesn't separate by treatment:
# 1. Check if you have batch effects
pcaData_batch <- plotPCA(vsd, intgroup = c("condition", "batch"), 
                         returnData = TRUE)

# 2. Check if treatment effect is subtle (look at PC2, PC3)
plotPCA(vsd, intgroup = "condition", ntop = 500)  # Use top 500 variable genes
```

### QC Step 3: Dispersion Plot

```r
# Check dispersion estimates
plotDispEsts(dds, main = "Dispersion Estimates")
```

**What to look for:**
- Black points: Gene-wise estimates
- Red line: Fitted trend
- Blue points: Final shrunk estimates

✅ **Good pattern:**
- Dispersion decreases with mean expression
- Most blue points close to red trend
- Few outliers

❌ **Bad pattern:**
- No clear trend (may need more replicates)
- Many outliers far from trend (data quality issues)

### QC Step 4: P-value Distribution

```r
# Check p-value distribution
hist(res$pvalue[res$baseMean > 1], 
     breaks = 50, 
     col = "grey",
     main = "P-value Distribution",
     xlab = "p-value")
```

**Expected:** 
- Uniform distribution with peak near 0
- Peak represents true positives

**Problems:**
- U-shaped (peak at 0 and 1): Possible batch effects
- Completely uniform: No signal, increase sample size

### QC Step 5: Check Cook's Distance

```r
# Check for outliers flagged by Cook's distance
W <- res$stat
maxCooks <- apply(assays(dds)[["cooks"]], 1, max)
idx <- !is.na(W)
plot(rank(W[idx]), maxCooks[idx], 
     xlab = "Rank of Wald statistic", 
     ylab = "Maximum Cook's distance",
     main = "Cook's Distance vs Wald Statistic")
abline(h = qf(0.99, 2, ncol(dds) - 2))
```

**Interpretation:** Points above the line are potential outliers automatically filtered by DESeq2.

---

## Results Interpretation

### Visualization 1: Volcano Plot

```r
# Create volcano plot
volcano_data <- as.data.frame(res_shrink)
volcano_data$significant <- ifelse(volcano_data$padj < 0.05 & 
                                   abs(volcano_data$log2FoldChange) > 1,
                                   "Significant", "Not Significant")

ggplot(volcano_data, aes(x = log2FoldChange, y = -log10(padj), 
                         color = significant)) +
    geom_point(alpha = 0.6, size = 1.5) +
    scale_color_manual(values = c("grey", "red")) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
    xlim(-6, 6) +
    ylim(0, max(-log10(volcano_data$padj), na.rm = TRUE) + 5) +
    labs(title = "Volcano Plot: Treated vs Untreated",
         x = "Log2 Fold Change",
         y = "-Log10 Adjusted P-value") +
    theme_bw()
```

### Visualization 2: Heatmap of Top Genes

```r
# Select top 50 significant genes
top_genes <- head(order(res_shrink$padj), 50)

# Create heatmap
mat <- assay(vsd)[top_genes, ]
mat <- mat - rowMeans(mat)  # Center rows

pheatmap(mat, 
         annotation_col = as.data.frame(colData(dds)[, "condition", drop = FALSE]),
         show_rownames = TRUE,
         cluster_cols = TRUE,
         cluster_rows = TRUE,
         main = "Top 50 Differentially Expressed Genes",
         fontsize_row = 6)
```

### Visualization 3: Individual Gene Plots

```r
# Plot counts for top gene
top_gene <- rownames(res_shrink)[which.min(res_shrink$padj)]

plotCounts(dds, gene = top_gene, intgroup = "condition", 
           returnData = FALSE,
           main = paste("Expression of", res_shrink[top_gene, "symbol"]))
```

### Biological Interpretation

**Key questions to ask:**

1. **Do the top genes make biological sense?**
   - Are known pathway genes enriched?
   - Do fold changes match expected biology?

2. **Are there unexpected findings?**
   - Novel genes worth following up
   - Contradictions with literature

3. **What's next?**
   - Pathway enrichment analysis (GSEA, GO)
   - Validate top hits with qRT-PCR
   - Functional experiments

---

## Advanced: Handling Complex Designs

### Batch Effect Correction

```r
# If you have batch information
design(dds) <- ~ batch + condition

# Re-run DESeq
dds <- DESeq(dds)

# Results will now control for batch
res_batch <- results(dds, contrast = c("condition", "trt", "untrt"))
```

### Paired Sample Analysis

```r
# For paired samples (e.g., before/after treatment in same patient)
design(dds) <- ~ patient + condition

# Patient effects are controlled, focus on treatment
res_paired <- results(dds, contrast = c("condition", "trt", "untrt"))
```

**Why this matters:** Paired designs have more statistical power because they account for patient-specific baseline differences.

---

## Common Pitfalls & Solutions

### Problem 1: Small Sample Size

**Symptoms:** 
- Wide confidence intervals
- Few significant genes despite clear PCA separation

**Solutions:**
```r
# Use independent filtering (automatic in DESeq2)
# Be more conservative with thresholds
res_conservative <- results(dds, alpha = 0.01)  # Stricter FDR

# Use biological replicates, not technical replicates
# Minimum: 3 replicates per group
# Better: 5-6 replicates per group
# Ideal: 10+ replicates for complex designs
```

### Problem 2: High Dispersion

**Symptoms:**
- Dispersion plot shows many outliers
- Few significant genes

**Solutions:**
```r
# Check for outlier samples
# Remove problematic samples
# Consider if biological variability is expected (e.g., patient samples)
```

### Problem 3: Batch Effects

**Symptoms:**
- Samples cluster by batch in PCA, not by condition

**Solutions:**
```r
# Add batch to design
design(dds) <- ~ batch + condition

# Or use ComBat for severe cases (before DESeq2)
library(sva)
batch_corrected <- ComBat_seq(counts = counts(dds), 
                              batch = colData(dds)$batch)
```

---

## Conclusion

### Summary Checklist

- [ ] Raw counts loaded (integers only)
- [ ] Proper experimental design specified
- [ ] Pre-filtering applied
- [ ] DESeq2 pipeline executed
- [ ] QC plots reviewed (PCA, dispersion, p-values)
- [ ] Results extracted with shrinkage
- [ ] Significance thresholds applied (FDR < 0.05, |log2FC| > 1)
- [ ] Results annotated and exported
- [ ] Visualizations created (volcano, heatmap)
- [ ] Biological interpretation completed

### Key Takeaways

1. **RNA-seq counts are not microarray intensities** - use proper statistical models (negative binomial GLM)
2. **DESeq2 is robust and well-tested** - stick with defaults unless you have good reason
3. **Pre-filtering improves power** - remove uninformative low-count genes
4. **Shrinkage improves effect sizes** - use for visualization and ranking
5. **QC is essential** - always check PCA, dispersion, and p-value distributions
6. **Biological significance ≠ statistical significance** - use both p-value and fold change thresholds

### Further Reading

- **DESeq2 paper:** Love et al. (2014) Genome Biology
- **Best practices:** Conesa et al. (2016) Genome Biology
- **Statistical theory:** Robinson & Oshlack (2010) Genome Biology

### Next Steps

After DGE analysis:
1. **Gene Set Enrichment Analysis (GSEA)** - identify affected pathways
2. **Functional validation** - qRT-PCR for top hits
3. **Integration with other omics** - proteomics, metabolomics
4. **Network analysis** - find hub regulators

---

**Questions?** This is a living document. Reach out for clarifications or suggest improvements.

**Repository:** Save this tutorial with your analysis code for reproducibility.
