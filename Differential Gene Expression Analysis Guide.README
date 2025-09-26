# Expression Data Analysis Methods: Microarray vs RNA-seq Guide

## 1. DATA PREPROCESSING & NORMALIZATION

### Microarray Data
**✅ APPROPRIATE:**
- **RMA (Robust Multi-array Average)** - Gold standard for Affymetrix
- **Quantile normalization** - Makes distributions identical across arrays
- **Background correction** - GCRMA, RMA background subtraction
- **Log2 transformation** - Stabilizes variance, makes data normal
- **Loess normalization** - For two-color arrays
- **VSN (Variance Stabilizing Normalization)** - Alternative to log transformation
- **MAS5** - Older but still valid Affymetrix method

**❌ INAPPROPRIATE:**
- **TMM normalization** - Designed for RNA-seq count data
- **DESeq2 size factors** - Count-based normalization
- **CPM/RPKM/FPKM calculations** - Count-based metrics
- **Upper quartile normalization** - RNA-seq specific
- **Voom transformation** - Converts counts to log-CPM

### RNA-seq Data
**✅ APPROPRIATE:**
- **TMM normalization** - Accounts for library size and composition
- **DESeq2 size factors** - Robust to highly expressed genes
- **Upper quartile normalization** - Simple alternative to TMM
- **Quantile normalization** - Can be used but less common
- **TPM (Transcripts Per Million)** - For between-sample comparisons
- **FPKM/RPKM** - For gene length normalization (with caveats)
- **Voom transformation** - For using limma with RNA-seq

**❌ INAPPROPRIATE:**
- **RMA normalization** - Designed for probe intensities
- **MAS5** - Affymetrix-specific
- **GCRMA** - Microarray background correction
- **Direct log transformation of counts** - Doesn't account for mean-variance relationship

---

## 2. QUALITY CONTROL

### Microarray Data
**✅ APPROPRIATE:**
- **Probe-level quality metrics** - RLE, NUSE plots
- **Array pseudo-images** - Spatial artifacts detection
- **MA plots** - Array vs array comparisons
- **Boxplots of intensities** - Distribution assessment
- **Density plots** - Distribution shape analysis
- **PCA on log-intensities** - Sample clustering
- **Correlation heatmaps** - Sample relationships

**❌ INAPPROPRIATE:**
- **Alignment statistics** - No sequencing involved
- **Duplication rates** - Not applicable to arrays
- **rRNA contamination checks** - Array probes are pre-defined
- **Saturation curves** - No sequencing depth concept

### RNA-seq Data
**✅ APPROPRIATE:**
- **Alignment/mapping rates** - Quality of sequencing
- **Duplication rates** - Library complexity assessment
- **rRNA contamination** - Sample preparation quality
- **Gene body coverage** - 5' to 3' bias detection
- **Saturation curves** - Sequencing depth adequacy
- **PCA on log-CPM** - Sample clustering
- **Count distribution plots** - Library size assessment
- **BCV plots** - Biological coefficient of variation

**❌ INAPPROPRIATE:**
- **Probe-level metrics (RLE/NUSE)** - No probes in RNA-seq
- **Array pseudo-images** - No spatial layout
- **Perfect match vs mismatch** - Affymetrix-specific

---

## 3. FILTERING

### Microarray Data
**✅ APPROPRIATE:**
- **Intensity-based filtering** - Remove low-intensity probes
- **Variance-based filtering** - Keep variable probes
- **Detection call filtering** - Present/Absent calls
- **Inter-quartile range (IQR) filtering** - Variability-based
- **Coefficient of variation filtering** - Relative variability

**❌ INAPPROPRIATE:**
- **Count-based filtering** - No discrete counts in microarray
- **CPM thresholds** - Count-specific metric
- **filterByExpr()** - edgeR function for RNA-seq
- **Minimum read count filtering** - No reads in microarray

### RNA-seq Data
**✅ APPROPRIATE:**
- **Count-based filtering** - Remove low-count genes
- **CPM thresholds** - Minimum counts per million
- **filterByExpr()** - edgeR's adaptive filtering
- **Total count filtering** - Minimum total reads per gene
- **Sample proportion filtering** - Present in X% of samples

**❌ INAPPROPRIATE:**
- **Intensity-based filtering** - No intensities in RNA-seq
- **Detection call filtering** - No present/absent in RNA-seq
- **Probe-specific filtering** - No probes in RNA-seq

---

## 4. DIFFERENTIAL EXPRESSION ANALYSIS

### Microarray Data
**✅ APPROPRIATE:**
- **limma** - Linear models for microarray analysis
- **SAM (Significance Analysis of Microarrays)** - Specialized for arrays
- **ANOVA** - Traditional statistical approach
- **t-tests** - Simple two-group comparisons
- **Rank-based methods** - Non-parametric alternatives
- **Bayesian methods** - Like limma's empirical Bayes

**❌ INAPPROPRIATE:**
- **edgeR** - Negative binomial models for counts
- **DESeq2** - Count-based generalized linear models
- **Voom + limma** - Voom is for RNA-seq count transformation
- **NOISeq** - RNA-seq specific non-parametric method

### RNA-seq Data
**✅ APPROPRIATE:**
- **edgeR** - Negative binomial GLMs, handles overdispersion
- **DESeq2** - Robust count-based analysis
- **limma-voom** - Transforms counts for limma analysis
- **NOISeq** - Non-parametric, no distributional assumptions
- **Ballgown** - For transcript-level analysis
- **sleuth** - For transcript/isoform analysis with kallisto

**❌ INAPPROPRIATE:**
- **Direct limma on counts** - Doesn't handle count distributions
- **t-tests on raw counts** - Violates normality assumptions
- **SAM on counts** - Designed for continuous data
- **ANOVA on raw counts** - Count distributions aren't normal

---

## 5. PATHWAY & FUNCTIONAL ANALYSIS

### Both Microarray and RNA-seq
**✅ APPROPRIATE:**
- **Gene Ontology (GO) enrichment** - clusterProfiler, GOstats
- **KEGG pathway analysis** - Pathway databases
- **Reactome analysis** - Biological pathway database
- **GSEA (Gene Set Enrichment Analysis)** - Rank-based analysis
- **MSigDB gene sets** - Curated gene collections
- **STRING network analysis** - Protein-protein interactions
- **IPA (Ingenuity Pathway Analysis)** - Commercial tool
- **David/EASE** - Functional annotation tools

**✅ DATA-TYPE CONSIDERATIONS:**
- **Microarray**: Use probe-to-gene mapping, handle multiple probes per gene
- **RNA-seq**: Direct gene symbols/IDs, consider isoform-level analysis

---

## 6. BATCH EFFECT CORRECTION

### Microarray Data
**✅ APPROPRIATE:**
- **ComBat** - Empirical Bayes batch correction
- **SVA (Surrogate Variable Analysis)** - Unknown batch detection
- **RUV (Remove Unwanted Variation)** - Control gene-based correction
- **Quantile normalization** - Can reduce some batch effects
- **limma's removeBatchEffect()** - For visualization

### RNA-seq Data
**✅ APPROPRIATE:**
- **ComBat** - Works with log-transformed data
- **ComBat-seq** - Specifically designed for count data
- **SVA** - Surrogate variable analysis
- **RUV-seq** - RNA-seq adapted RUV
- **limma-voom with batch terms** - Include batch in model

**❌ BOTH SHOULD AVOID:**
- **Simply ignoring batch effects** - Can lead to false discoveries
- **Over-correction** - Removing biological signal
- **Batch correction without known batches** - Unless using SVA/RUV

---

## 7. VISUALIZATION

### Microarray Data
**✅ APPROPRIATE:**
- **MA plots** - Array comparison plots
- **Volcano plots** - DE gene visualization
- **Heatmaps of intensities** - Expression patterns
- **PCA plots** - Sample relationships
- **Density plots** - Intensity distributions
- **Boxplots** - Array-to-array comparisons

### RNA-seq Data
**✅ APPROPRIATE:**
- **MA plots** - Mean vs log-fold change
- **Volcano plots** - P-value vs fold-change
- **Heatmaps of log-CPM** - Expression patterns
- **PCA plots** - Sample clustering
- **BCV plots** - Biological coefficient of variation
- **Count distribution plots** - Library size assessment

---

## 8. SPECIFIC R PACKAGES RECOMMENDATIONS

### Microarray-Specific Packages
```r
# Core analysis
library(limma)          # Linear models
library(affy)           # Affymetrix arrays
library(oligo)          # Oligonucleotide arrays

# Normalization
library(gcrma)          # GC-robust multi-array average
library(vsn)            # Variance stabilization

# Quality control
library(arrayQualityMetrics)  # Comprehensive QC
library(affyPLM)        # Probe-level models
```

### RNA-seq-Specific Packages
```r
# Core analysis
library(edgeR)          # Negative binomial models
library(DESeq2)         # Count-based DE analysis
library(limma)          # With voom transformation

# Preprocessing
library(tximport)       # Import transcript-level estimates
library(GenomicAlignments)  # Read counting

# Quality control
library(RSeQC)          # RNA-seq quality control
library(Rsubread)       # Alignment and counting
```

---

## 9. COMMON MISTAKES TO AVOID

### Microarray Mistakes
❌ **Using RNA-seq tools on microarray data**
❌ **Not log-transforming intensity data**
❌ **Ignoring probe-to-gene mapping issues**
❌ **Using count-based filtering methods**
❌ **Applying voom to already-continuous data**

### RNA-seq Mistakes
❌ **Using microarray normalization on counts**
❌ **Log-transforming counts without proper offset**
❌ **Ignoring library size differences**
❌ **Using normal distributions for count data**
❌ **Not accounting for overdispersion**

---

## 10. DECISION FLOWCHART

```
Do you have discrete count data?
├── YES → RNA-seq pipeline
│   ├── edgeR/DESeq2 for DE
│   ├── TMM/size factor normalization
│   └── Count-based filtering
└── NO → Do you have continuous intensity data?
    ├── YES → Microarray pipeline
    │   ├── limma for DE
    │   ├── Quantile/RMA normalization
    │   └── Intensity-based filtering
    └── NO → Determine your data type first!
```

This guide should help you choose the right methods for your specific expression data type and avoid common analytical pitfalls.
