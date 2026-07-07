 

## **ATRIA | EchoTrace** 

**Artifact Transformation & Resource Interoperability in Artificial Intelligence** 

**| LLM-tuned Data-Driven Echocardiographic Contour Tracing** 

## **An Implementation of The Adimension’s and DEITY Principles Framework** 

ATRIA EchoTrace is a flagship demonstration of the ATRIA vision for Artifact Transformation & Resources Interoperability in AI. It fully implements the DEITY Principles Framework (Data • Ethics • Informatics • Technology • You) as detailed in the companion technical document “ATRIOM_MedGemma_DEITY_Framework.docx”. 

By treating data as a transparent, reusable asset, embedding continuous human oversight as an ethical core, producing interpretable structured outputs, leveraging efficient modern technology, and — most importantly — centering the human user as an active collaborator, ATRIA advances the vision of Adimensional Intelligence: AI that augments rather than supplants clinical expertise in echocardiography. 

Next steps include full video sequence modeling, joint CAMUS + EchoNet training, extension to additional structures, integration of revised polygons into active learning loops, and deployment of the revision interface as a lightweight clinical tool. All code, artifacts, and this discovery sheet are designed for immediate reuse, extension, and community contribution. 


---


## **Important Disclaimers** 

## **I. INTENDED USE** 
ATRIA EchoTrace is currently classified as a Prototype / Advanced Minimum Viable Product (MVP). The platform, including its MedGemma-based models, LoRA adapters, pipelines, and user interfaces, is explicitly intended for research, academic collaboration, and internal validation. It is NOT a cleared or approved medical device by regulatory bodies at the time of this document generated. 

## **II. NON-CLINICAL USE** 
Not for Direct Clinical Diagnosis.ATRIA EchoTrace is an AI-powered annotation and drafting tool designed to support and augment medical research and dataset creation. The software must not be used as a standalone diagnostic tool, for direct patient care, or as the sole basis for clinical decision-making. Any downstream metrics derived from its outputs (such as ejection fraction, cardiac volume, or strain calculations) are strictly for research and validation purposes. 

## **III. MANDATORY HUMAN OVERSIGHT** 
In strict adherence to the DEITY Principles Framework (specifically the "You" pillar), this system relies on a Human-inthe-Loop (HITL) architecture. All AI-generated structural contours (LV/LA) and annotations are preliminary proposals. They must be independently reviewed, verified, and revised by qualified echocardiographers, cardiologists, or trained medical personnel. 

## **IV. DATA PRIVACY** 
Users of ATRIA EchoTrace are solely responsible for ensuring that all echocardiographic inputs (DICOM, AVI, PNG) are properly anonymized and de-identified prior to processing. 

## **V. REGULATORY COMPLIANCE** 
Users must ensure their use of the platform complies with all applicable local, national, and international healthcare data privacy regulations (e.g., HIPAA, GDPR). 

## **VI. "AS IS" PROVISION** 
The ATRIA EchoTrace codebase, models, artifacts, and documentation are provided "AS IS" without any warranties of accuracy, reliability, or clinical fitness, either express or implied. 

## **VII. LIMITATION OF LIABILITY** 
The developers, contributors, and affiliated research initiatives assume no liability for any direct, indirect, or consequential damages, or clinical outcomes arising from the use, misuse, or inability to use this platform. 

## How can I try **ATRIA EchoTrace** live?
On Google Colab: https://colab.research.google.com/drive/1qofahQ8LztTrB_Us9j1Iyz2aYeS2_2rH?usp=sharing

---


## What is **ATRIA EchoTrace** ? 

ATRIA EchoTrace — Am Interoperable Data-driven MedGemma-based Adapters for Cardiac Imaging Structural Contour Tracing to deliver an interactive AI-powered annotation platform that uses fine-tuned MedGemma with extensive LoRA adapters to automatically generate precise LV and LA endocardial contours on echocardiographic frames, while empowering clinicians to review and revise predictions through an intuitive 4-panel visual interface — producing high-quality, reusable training data in full alignment with the DEITY Principles Framework for responsible medical AI. 

## What Are **ATRIA EchoTrace** ’s Key Features? 

- Config-driven LV/LA tracing via single TARGET_STRUCTURE flag — no code changes required to switch structures 
- Complete end-to-end reproducible pipeline: preprocessing (DICOM/AVI → PNG + JSON) → LoRA fine-tuning → inference → interactive HITL revision → serialized outputs 
- Interactive drag-to-edit polygon canvas with real-time redrawing and visual comparison overlays 
- Automated generation of professional 4-panel visualization figures (Original / Model / User / Overlay) and normalized JSON (model + user polygons) 
- Full DEITY Principles Framework implementation ensuring transparent data practices, ethical governance, interpretable outputs, accessible technology, and human empowerment 
- Memory-efficient training via 4-bit quantization (bitsandbytes NF4), extensive LoRA on vision tower, and custom CUDA memory configuration — runnable on Colab or mid-range GPUs 

## What Does **ATRIA EchoTrace** Do? 

Transforms raw echocardiographic DICOM and AVI files into standardized PNG frames with normalized 30-point polygon traces. Fine-tunes Google’s MedGemma-1.5-4B-it model using parameter-efficient LoRA across the vision tower. Performs inference to predict initial endocardial polygons on apical 2CH/4CH views at ED/ES phases. Presents clinicians with a 4-panel interactive revision interface (Original | Model Prediction in red | User Revision in green | Overlay) where polygons can be edited vertex-by-vertex. Saves both model and revised polygons as structured JSON plus publication-ready visualizations. Revised data feeds directly back into iterative model improvement or serves as enhanced ground truth. The entire pipeline is modular, reproducible, and explicitly designed around the DEITY Principles (Data • Ethics • Informatics • Technology • You). 

## Who Uses **ATRIA EchoTrace** ? 

- Echocardiographers and cardiologists performing annotation, quality control, or clinical research 
- Medical AI researchers and data scientists building cardiac segmentation and analysis models 
- Clinical annotation teams and echo core labs creating large-scale ground-truth datasets 
- Healthcare AI developers and governance teams seeking DEITY-aligned, responsible AI tooling 
- Academic institutions and industry partners collaborating on open, interoperable cardiac imaging AI 

## What Problem Does **ATRIA EchoTrace** Solve? 

- Manual endocardial border tracing is extremely time-consuming, highly subjective, and a major bottleneck for large-scale cardiac AI development 
- Pure AI models frequently produce imperfect initial traces that still require expert correction, with no efficient feedback mechanism 
- Lack of transparent provenance, reproducibility, and ethical human oversight in medical image annotation pipelines 
- High computational barriers prevent many institutions from fine-tuning large vision-language models on echocardiography tasks 
- Fragmented tools and non-standardized outputs hinder interoperability across datasets, institutions, and downstream clinical systems. 

## How Integration-Ready is **ATRIA EchoTrace** ? 

- Input: Any echo dataset producing the standardized three artifacts (PNG frames, tracings.json with normalized polygons, metadata.csv) full provenance and reusable IO contract 
- Output: Structured JSON polygons (machine-readable for volume/EF/strain calculation) + PNG visualizations (human-readable for review/publication) 
- Tools: Hugging Face Transformers + PEFT (LoRA) + PyTorch; modular design allows easy extension to new backbones, structures (RV, valves), or datasets 
- Modularity: Dataset construction, Model Finetuning, Self-contained ablation, Adapter-Human Interactive Interface. 
- Readiness: Direct DICOM/PACS import, lightweight web-based clinical UI, active learning integration, and downstream clinical reporting system connectors 
- Architecture: The Adimension’s foundational DEITY Principles Framework. 
 
## **ATRIA EchoTrace** ? How Does the DEITY Principles Framework define 

This product is explicitly architected around the DEITY Principles Framework (Anwer et al., EHJIMP 2026) to cover the pillars of The Adimension’s foundational framework: 

|**DATA**|Multi-format ready|
|---|---|
||2000+ preprocessed PNG frames|
||JSON 30-pt normalized polygons (LV/LA)|
||Metadata CSV|
||Distributed imaging view, modality, and cardiac cycle stage|
|**ETHICS**|Open CAMUS/EchoNet dataset|
||Resources Accessibility|
||Local-first implementation|
||Ground-truth evolution loop|
||Continuous human oversight|
|**INFORMATICS**|Structured Outputs: JSON coordinates|
||Built-in visualization|
||Configurable Parameterization|
||Ablation-ready sample controls|
||Full pipeline traceability|
|**TECHNOLOGY**|Foundational Model as a base: MedGemma-1.5-4B-it|
||Parameter-Efficient Fine Tuning for specialized adapters|
||Resource-efficient implementation via bitsandbytes 4-bit|
||Modular Python and Jupyter pipeline|
||Data-driven Integration readiness|
|**YOU**|Adapter-Human Interactive revision UI|
||Human edits directly enable data-driven iterative model development|
||Jupyter empowers researchers + clinicians|
||Standardized reusable outputs|
||Open-source codebase for the community|

## What **AI** is Implemented in **ATRIA EchoTrace** ? 

- Google’s MedGemma-1.5-4B-it — a powerful vision-language model pretrained on large-scale medical data, well-suited for echocardiographic semantics and spatial relationships 
- Parameter-efficient fine-tuning with LoRA adapters applied extensively across the vision_tower (self-attention Q/K/V/Out_proj and MLP layers in nearly all 27 encoder layers) plus language components 
- Structured prompting and JSON parsing to output normalized polygon coordinates directly consumable by clinical software 
- 4-bit quantization (NF4) with double quantization and custom PyTorch CUDA allocator for stable, low-memory training and inference 

## What Does the **AI ATRIA EchoTrace** ? Specifically Do for 

Analyzes apical 2-chamber or 4-chamber echocardiogram frames at end-diastole or end-systole and outputs a closed polygon (list of [y, x] coordinate pairs normalized to the 0–1000 range) representing the endocardial border of the left ventricle or left atrium. The model leverages medical-domain visual understanding to propose anatomically plausible initial contours that clinicians can rapidly refine. 

## Why Is AI Important in **ATRIA EchoTrace** ? 

- Dramatically reduces annotation time from minutes of freehand drawing to seconds of review plus targeted edits 
- Provides semantically rich initial predictions that capture echocardiographic context far better than generic computer vision models 
- Creates a scalable, continuous improvement loop: human revisions become higher-quality training signals for subsequent model iterations 
- Embodies the core DEITY 'You' pillar — AI acts as a collaborative drafting partner that amplifies rather than replaces clinical expertise 
- Enables institutions with limited compute resources to participate in state-of-the-art medical VLM fine-tuning through efficient LoRA + quantization 

## What Makes **ATRIA EchoTrace** Valuable? 

- Time & Cost Efficiency: Converts hours of manual tracing into minutes of expert review and correction 
- Annotation Quality & Reproducibility: Structured, normalized, fully traceable outputs with complete pipeline provenance 
- Responsible & Ethical AI: Explicit, auditable alignment with the DEITY Principles Framework across all five pillars 
- Continuous Improvement Engine: Captured human revisions directly enhance both immediate ground truth and future model performance 
- Community & Interoperability: Open notebook format, reusable artifacts, and standardized IO promote crossinstitution collaboration and extension. 


## What Makes **ATRIA EchoTrace** Stand Out from Competitors? 

- Deep, native integration of the DEITY Principles Framework — unique among echocardiography annotation tools for transparent, ethical, human-centered design 
- True bidirectional human-AI synergy: Model proposes → Human refines with pixel-level precision → System learns from every correction (active feedback loop, not one-way inference or post-hoc validation) 
- Extensive, targeted LoRA adaptation on the vision_tower for superior spatial and semantic understanding in echocardiography compared with generic or lightly-adapted models 
- Native multi-structure (LV/LA switchable), multi-view (2CH/4CH), and multi-phase (ED/ES) support within a single modular, config-driven pipeline 
- Production-ready standardized outputs (device-agnostic normalized polygons + rich visualizations) immediately usable in clinical software and research pipelines 
- Fully reproducible, living Jupyter notebook designed as an evolving community template rather than a blackbox commercial tool. 

## What is the Current Status and Stage of **ATRIA EchoTrace** ? 

Prototype / Advanced MVP — Complete, demonstrated end-to-end pipeline including preprocessing, LoRA finetuning on CAMUS-style data, inference, interactive 4-panel revision UI with drag editing, and automated JSON + visualization documentation generation. User revision saving successfully validated on real echocardiographic frames. 

## What Does **ATRIA EchoTrace** Currently Targets? 

Internal development and validation within the ATRIA cardiac AI research initiative. The platform is designed for immediate collaborative adoption by academic hospitals, echocardiography core laboratories, and medical AI research groups working with CAMUS, EchoNet-Dynamic, or institution-specific echo datasets. Open to structured pilot programs focused on annotation efficiency, ground-truth quality improvement, and responsible AI governance. 

## What Are the Current Challenges Facing **ATRIA EchoTrace** ? 

- Transitioning the interactive revision UI from Colab notebook to a lightweight, production-grade web or desktop clinical application 
- Extending the pipeline to full video/sequence modeling, temporal consistency, and additional cardiac structures (RV free wall, valves, strain curves) 
- Clarifying regulatory pathways and intended-use boundaries if the tool evolves from research annotation aid toward clinical decision-support components 
- Building larger-scale clinical validation studies quantifying time savings, inter-observer variability reduction, and downstream impact on ejection fraction/strain measurements 
- Further lowering resource barriers and optimizing for real-time or edge deployment scenarios 

 
## What Does **The Adimension** Prioritize for **ATRIA EchoTrace** ? 

- Engage practicing echocardiographers and cardiology departments for structured usability testing and timemotion studies 
- Position ATRIA as the reference open implementation for DEITY-compliant cardiac imaging AI annotation platforms 
- Establish collaborative pilot programs with echo core labs and academic centers for dataset co-creation 
- Develop a standalone lightweight web application version of the 4-panel revision interface for broader clinical accessibility 
- Add batch processing, direct DICOM import, and multi-structure simultaneous tracing capabilities 
- Create versioned, ATRIA-enhanced ground-truth dataset releases for community use 
- Implement active learning / uncertainty sampling to intelligently prioritize cases for human review 
- Extend modeling to full video sequences (EchoNet-Dynamic style) with temporal coherence constraints 
- Package the pipeline as an installable Python module with CLI and REST API endpoints for easier integration 
- Publish the complete notebook, preprocessed artifacts, and DEITY framework mapping on GitHub and Hugging Face with comprehensive documentation 
- Produce short demonstration videos and case studies highlighting the revision workflow and measurable annotation time reduction 
- Present at major cardiology imaging and medical AI conferences; pursue publication in relevant journals 


## REFERENCES 

- **The Adimension & DEITY Principles** Anwer, S. (2026). The Adimension: Bridging human ingenuity and machine intelligence through the DEITY principles framework. _European Heart Journal - Imaging Methods and Practice_ , _4_ (1), qyaf038. https://doi.org/10.1093/ehjimp/qyaf038 

- **CAMUS Dataset** Leclerc, S., Smistad, E., Pedrosa, J., Østvik, A., Cervenansky, F., Espinosa, F., Espeland, T., Berg, E. A. R., Jodoin, P.-M., Grenier, T., Lartizien, C., Dhooge, J., Løvstakken, L., & Bernard, O. (2019). Deep learning for segmentation using an open large-scale dataset in 2D echocardiography. _IEEE Transactions on Medical Imaging_ , _38_ (9), 2198–2210. https://doi.org/10.1109/tmi.2019.2900516 

- **EchoNet-Dynamic** Ouyang, D., He, B., Ghorbani, A., Yuan, N., Ebinger, J., Langlotz, C. P., Heidenreich, P. A., Harrington, R. A., Liang, D. H., Ashley, E. A., & Zou, J. Y. (2020). Video-based AI for beat-to-beat assessment of cardiac function. _Nature_ , _580_ (7802), 252–256. https://doi.org/10.1038/s41586-020-2145-8 

- **Google MedGemma 1.5 Google. (2026).** MedGemma 1.5: Technical reports and model card (google/medgemma-1.54b-it). Hugging Face. https://huggingface.co/google/medgemma-1.5-4b-it 

- **LoRA: Low-Rank Adaptation** . Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv. https://doi.org/10.48550/arXiv.2106.09685 