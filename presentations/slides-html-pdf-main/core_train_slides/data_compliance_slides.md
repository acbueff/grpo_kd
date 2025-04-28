# EU LLM Training Project - Data Compliance & Filtering

## Slide 1: Title Slide
**Title:** Data Compliance & Filtering Protocol
**Subtitle:** Ensuring Regulatory Compliance in Large Language Model Training
*[Visual: Data filtering funnel diagram showing compliance layers]*

---

## Slide 2: Three-Layer Filtering Process
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

*[Visual: Filtering funnel diagram showing data volume reduction through layers]*

---

## Slide 3: PII Management Strategy
**Title:** PII Management Strategy

**Baseline Approach:**
* Regex-based filtering of structured PII
* Language-specific pattern recognition for jurisdictional compliance
* High-precision focus on clearly sensitive identifiers
* Implementation focused on ID numbers, bank accounts, contact details

**Implementation Considerations:**
* Balance between thorough filtering and preserving data utility
* Varying approaches for different languages and domains
* Integration with training pipeline for consistent application

*[Visual: PII filtering pipeline showing text transformation examples]*

---

## Slide 4: Copyright & IPR Compliance
**Title:** Copyright & IPR Compliance Framework

**Core Principles:**
* Respect for machine-readable opt-outs (robots.txt, ai.txt)
* No circumvention of access barriers
* Documentation of lawful access

**Practical Implementation:**
* Web crawler configured to respect TDM reservation standards
* Automated identification and exclusion of blacklisted licenses
* Regular auditing of dataset source domains
* Retention policies aligned with training requirements

*[Visual: License detection and filtering workflow diagram]*

---

## Slide 5: Primary Data Sources
**Title:** Primary Training Data Sources

**Filtered Web Corpora:**
* FineWeb & FineWeb 2
* HPLT (High-Performance Language Technologies)
* DCLM (Distributed Curation for Language Models)

**Supplementary Sources:**
* Parallel and translation corpora for multilingual alignment
* Nordic language datasets (Faroese, Icelandic, Norwegian)
* Domain-specific datasets (legal, public administration, climate)

*[Visual: Data sources composition pie chart with proportions]*

---

## Slide 6: Technical Implementation
**Title:** Technical Implementation

**Data Processing Pipeline:**
* Distributed preprocessing across computing nodes
* Continuous validation of filtering effectiveness
* Automated license detection and classification
* Language-specific PII recognition modules

**Quality Assurance:**
* Regular sampling of filtered outputs
* Manual verification of edge cases
* Retention of filtering metadata for auditing
* Performance impact assessment

*[Visual: Data processing pipeline architecture diagram]*

---

## Slide 7: PII Detection Specifics
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

## Slide 8: Compliance vs. Utility
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

## Slide 9: Comparative Filtering Approaches
**Title:** Comparative Filtering Approaches

**Baseline (Currently Implemented):**
* Regex-based filtering of structured PII
* License-aware content selection
* TDM exception compliance

**Advanced Options (If Required):**
* Named Entity Recognition for broader PII detection
* Synthetic name replacement
* Coreference resolution and chain formatting
* Entity linking and validation

*[Visual: Feature comparison table of different filtering approaches]*

---

## Slide 10: Implementation Timeline
**Title:** Implementation Timeline

**Current Phase (March-April 2024):**
* Finalization of filtering rules and patterns (by April 14, 2024)
* Initial planning meeting with stakeholders (April 14, 2024)
* Infrastructure setup for distributed processing

**Data Preparation (May-December 2024):**
* Initial dataset acquisition (May-July 2024)
* Main preprocessing phase (August-October 2024)
* Quality validation and compliance verification (November-December 2024)

**Training Phase (June 2025-March 2026):**
* Training commencement (June 15-30, 2025)
* Interim evaluation (December 2025) 
* Final model delivery (March 2026)

*[Visual: Timeline showing filtering implementation milestones with specific dates]*

---

## Slide 11: Research Questions
**Title:** Key Research Questions

**Compliance Impact:**
* How do different levels of filtering affect model performance?
* Are there critical thresholds beyond which utility diminishes?
* Do different languages show varying sensitivity to filtering?

**Technical Considerations:**
* Can we quantify the compliance-performance trade-off?
* How do filtering approaches affect downstream capabilities?
* Are there domain-specific impacts of different filtering levels?

*[Visual: Matrix showing research questions and measurement approaches]*

---

## Slide 12: Next Steps
**Title:** Next Steps and Action Items

**Immediate Priorities:**
* Finalize regex pattern libraries for all target languages
* Complete license detection tooling implementation
* Establish quality control sampling methodology

**Collaboration Needs:**
* Input from legal experts on TDM implementation
* Technical feedback on filtering efficiency
* Domain expertise for language-specific patterns

**Decision Points:**
* Confirmation of filtering intensity for initial training runs
* Go/no-go assessment of compliance adequacy
* Determination of variant filtering levels for research

*[Visual: Project milestone diagram with critical path highlighted]* 