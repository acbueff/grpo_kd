# EU LLM Training Project - Integrated Presentation

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

* **Current Phase** (March-April 2025): Final preparation with kickoff meeting April 14th, 2025
* **Pre-Training Setup** (April-June 2025): Final infrastructure preparation and verification
* **Training Commencement** (June 20-30, 2025): Launch of both Leonardo and Jupiter training runs
* **Training Duration** (June-October 2025): 2-4 month training period for both models
* **Interim Evaluation** (August 2025): Analysis of preliminary results
* **Final Model Delivery** (November 2025): Completed models with compliance-performance documentation

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

## Slide 8: Jupiter HPC Research - MoE Optimization
**Title:** Advanced MoE Optimization Research (Jupiter)

**Architecture Overview:**
* 30B parameter router with distributed experts (total model size comparable to DeepSeek V3's 671B parameters)
* Designed for efficient training on Jupiter's parallel computing architecture
* Leveraging sparse architecture principles from DeepSeekMoE (only ~5% of parameters active per token)

**DeepSeek V3 Baseline Metrics:**
* Performance: State-of-the-art on code (51.6 percentile on Codeforces) and math benchmarks (90.2% on MATH-500)
* Efficiency: 2.788M H800 GPU hours for complete training (2.664M for pre-training)
* Architecture: 671B total parameters with 37B activated per token
* Training: 14.8T tokens across diverse datasets
* Cost-efficiency: ~180K GPU hours per trillion tokens trained

**Our Key Innovations:**

1. **Enhanced BitNet Implementation**
   * Quantized weights using BitNet principles for 8-bit and lower precision
   * Custom block-wise quantization strategy reducing memory footprint by 50-75%
   * Improved fine-grained weight-sharing between experts

2. **Advanced Load Balancing Strategy**
   * Auxiliary-loss-free expert routing with dynamic bias adjustment
   * Enhanced expert specialization patterns for domain-specific tasks
   * Node-limited routing with optimized cross-node communication

*[Visual: MoE architecture diagram showing router and expert distribution]*

---

## Slide 9: Jupiter HPC Research - Performance Scaling
**Title:** Performance Scaling Expectations

**Benchmark Performance Targets:**
* MMLU: 76-80% (vs. DeepSeek's 87.1%)
* MATH-500: 80-85% (vs. DeepSeek's 90.2%)
* Codeforces: 47-50% (vs. DeepSeek's 51.6%)

**Efficiency Metrics:**
* ~6.7x smaller total parameter count
* ~5x less compute required for training
* Comparable performance per compute hour
* Dramatically better performance per parameter

**Parameter Activation Ratio Analysis:**
* DeepSeek: 671B × 5.5% = 36.9B effective parameters
* Our model: 100B × 30% = 30B effective parameters
* Similar performance with fewer effective parameters

*[Visual: Performance vs. parameter count scaling curve with our target highlighted]*

---

## Slide 10: Data Compliance Strategy
**Title:** Data Compliance Approach

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

## Slide 11: Data Filtering Protocol
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

## Slide 12: PII Detection Specifics
**Title:** PII Detection Implementation

**Structured Identifiers Targeted:**
* National ID numbers (format varies by country)
* Passport numbers
* Driver's license numbers
* Credit card numbers and CVV codes
* IBANs and bank account numbers
* IP addresses

**Technical Approach:**
* High-precision regex patterns for each identifier type
* Language-specific pattern libraries
* Conservative matching criteria to minimize false positives

*[Visual: Example regex patterns for different types of identifiers]*

---

## Slide 13: Balancing Compliance and Data Utility
**Title:** Balancing Compliance and Data Utility

**Key Considerations:**
* Overly aggressive filtering risks removing valuable linguistic patterns
* Insufficient filtering creates regulatory and ethical issues
* Language-specific challenges require tailored approaches

**Proposed Balance:**
* Focus on high-risk, clearly identifiable elements
* Preserve names, general locations, and common entities
* Document filtering decisions for transparency
* Multiple model variants with different filtering levels

*[Visual: Graph showing relationship between filtering intensity and model performance]*

---

## Slide 14: EU Compliance Integration
**Title:** EU Compliance Integration

**Key Compliance Features:**
* Training data filtering for regulatory alignment
* Structured evaluation for bias and fairness metrics
* Documentation of model development process
* Risk assessment protocols integrated into evaluation framework
* Performance impact analysis at varying compliance levels

**Target Outcome:**
* EU compliance without significant performance degradation
* Transparent model development process
* Documented tradeoffs between performance and compliance

*[Visual: EU compliance and performance balance scale diagram]*

---

## Slide 15: Research Value
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

## Slide 16: Key Research Questions
**Title:** Key Research Questions

**Compliance Impact:**
* How do different levels of filtering affect model performance?
* Are there critical thresholds beyond which utility diminishes?
* Do different languages show varying sensitivity to filtering?

**Technical Considerations:**
* Can we quantify the compliance-performance trade-off?
* How do filtering approaches affect downstream capabilities?
* Are there domain-specific impacts of different filtering levels?

**Efficiency Analysis:**
* What are the most effective ways to optimize MoE architectures?
* How does our approach compare to established scaling laws?
* Can we achieve better parameter utilization than existing models?

*[Visual: Matrix showing research questions and measurement approaches]*

---

## Slide 17: Next Steps
**Title:** Moving Forward Together

**Immediate Action Items:**
* Finalize all technical preparations by April 14th, 2025 meeting
* Complete final infrastructure testing (May 2025)
* Prepare for June 20-30th, 2025 training start

**Partner Collaboration:**
* WP2: Final data protocol validation
* Technical teams: Final infrastructure validation tests
* Evaluation team: Benchmark preparation

**Feedback Request:**
* Input on additional compliance considerations
* Suggestions for evaluation metrics
* Approval of training timeline and resource allocation

*[Visual: Roadmap with next milestone highlights and responsible teams]* 