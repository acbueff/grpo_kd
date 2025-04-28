# EU LLM Training Project - PowerPoint Content

## Slide 1: Title Slide
**Title:** EU LLM Training Project: Dual-Track Approach to Compliant Language Models
**Subtitle:** Exploring the Balance Between Compliance and Performance
*[Visual: EU flag/project logo, abstract neural network visualization]*

---

## Slide 2: Project Overview
**Title:** Project Overview

* **Objective:** Develop large language models that comply with European regulations while understanding compliance-performance tradeoffs
* **Research Focus:** Dual-track approach using complementary computing resources
* **Compliance Emphasis:** Exploring methods to balance regulatory requirements with model capabilities
* **Long-term Vision:** Building foundation for future compliant AI systems in Europe
* **Partners:** Collaboration across multiple EU research institutions and technical teams

*[Visual: Simplified diagram showing the project structure with main stakeholders]*

---

## Slide 3: Compute Proposals Status
**Title:** Compute Resources Secured

**Leonardo HPC (CONFIRMED):**
* Allocation secured for 100B parameter model training
* Utilizing Modalities training library
* Hardware specifications optimized for dense model architecture

**Jupiter EuroHPC (CONFIRMED):**
* Allocation secured for MoE training with 30B parameter router
* Hardware suited for sparse model architecture and parallel training
* Resource availability aligned with project timeline

*[Visual: Side-by-side comparison of the two computing resources with key specifications]*

---

## Slide 4: Project Timeline
**Title:** Project Roadmap

* **Current Phase** (March-April 2024): Preparation and planning with kickoff meeting April 14th
* **Data Preparation** (May 2024-May 2025): Dataset creation with varying compliance levels 
* **Training Commencement** (June 15-30, 2025): Launch of both Leonardo and Jupiter training runs
* **Interim Evaluation** (December 2025): Analysis of preliminary results
* **Final Model Delivery** (June 2026): Completed models with compliance-performance documentation

*[Visual: Timeline with key milestones, highlighting June 2025 training start date]*

---

## Slide 5: Dual Research Approach
**Title:** Two Complementary Research Tracks

**Track 1: Compliance vs. Performance (Leonardo)**
* Dense 100B parameter model
* Multiple training runs with varying data compliance

**Track 2: Training Optimization (Jupiter)**
* Mixture of Experts (MoE) architecture
* Focus on efficiency improvements over DeepSeek baseline

**Complementary Nature:**
* Findings from both tracks inform each other
* Enables comprehensive understanding of both compliance and efficiency dimensions

*[Visual: Venn diagram showing unique aspects of each track and shared insights]*

---

## Slide 6: Leonardo HPC Research
**Title:** Compliance vs. Performance Study (Leonardo)

* **Model Architecture:** 100B parameter dense transformer model
* **Training Framework:** Modalities library with standardized training protocol
* **Data Approach:** Multiple training runs using progressively filtered datasets
* **Control Variables:** Maintaining identical architecture, hyperparameters across runs
* **Measurement:** Comprehensive evaluation suite examining capabilities across domains
* **Key Question:** How do different levels of data compliance affect model capabilities?

*[Visual: Diagram showing identical model architecture with different data inputs]*

---

## Slide 7: Compliance-Performance Matrix
**Title:** Expected Compliance-Performance Relationship

**X-axis: Compliance Level**
* Low → High
* Less Filtering → More Filtering

**Y-axis: Model Performance**
* Metrics: Accuracy, Capability, Knowledge breadth

**Hypothesis:**
* Decreasing performance with increased compliance requirements
* Non-linear relationship with potential critical thresholds
* Domain-specific effects (e.g., greater impact on cultural knowledge)

*[Visual: Graph showing expected performance curve across compliance spectrum with annotations]*

---

## Slide 8: Jupiter HPC Research
**Title:** MoE Optimization Research (Jupiter)

* **Architecture:** 30B parameter router with experts (total size TBD)
* **Baseline:** DeepSeek training methodology and performance targets
* **Key Innovations:**
  * BitNet implementation for quantized weights
  * Enhanced parameter tuning methodology
  * Optimized training pipeline
* **Goal:** Maintain performance while improving training efficiency

*[Visual: MoE architecture diagram showing router and experts]*

---

## Slide 9: Data Compliance Strategy
**Title:** WP2's Data Compliance Approach

**Primary Approach (Baseline Filtering):**
* Regex-based filtering of structured PII 
* Language-specific pattern recognition for jurisdictional compliance
* High-precision focus on clearly sensitive identifiers
* Implementation focused on ID numbers, bank accounts, contact details

**Implementation Considerations:**
* Balance between thorough filtering and preserving data utility
* Varying approaches for different languages and domains
* Integration with training pipeline for consistent application

*[Visual: Filtering pipeline showing text transformation examples]*

---

## Slide 10: Data Filtering Protocol
**Title:** Three-Layer Filtering Process

**Layer 1: High-Risk Exclusion**
* Removal of copyrighted books, paywalled content, and known protected datasets
* Domain-level filtering for problematic sources

**Layer 2: License Compatibility**
* Removal of NonCommercial (NC) and ShareAlike (SA) licensed content
* Classification based on explicit license metadata

**Layer 3: TDM Exception Application**
* Leveraging Article 4 of CDSM Directive
* Respecting machine-readable opt-outs
* Documentation of lawful access

*[Visual: Funnel diagram showing data volume reduction through filtering layers]*

---

## Slide 11: Research Value
**Title:** Expected Outcomes and Significance

**Scientific Value:**
* Quantified understanding of compliance-performance relationship
* Insights into efficient training methodologies for sparse models
* Data filtering impact assessment across linguistic contexts

**Commercial Implications:**
* Framework for compliance-aware model development
* Optimized training approaches for reduced computing costs
* Clear guidance on regulatory boundaries for model deployment

**Broader Impact:**
* Supporting EU's position in responsible AI development
* Advancing understanding of AI regulation implementation

*[Visual: Impact diagram showing influence on various stakeholders]*

---

## Slide 12: Next Steps
**Title:** Moving Forward Together

**Immediate Action Items:**
* Finalize filtering implementation for varying compliance levels by April 14th, 2024
* Complete baseline dataset preparation (July 31st, 2024)
* Establish evaluation methodology and metrics (September 2024)
* Prepare technical infrastructure for June 2025 training start

**Partner Collaboration:**
* WP2: Continue data protocol refinement
* Technical teams: Infrastructure validation tests
* Evaluation team: Benchmark preparation

**Feedback Request:**
* Input on additional compliance considerations
* Suggestions for evaluation metrics

*[Visual: Roadmap with next milestone highlights and responsible teams]*