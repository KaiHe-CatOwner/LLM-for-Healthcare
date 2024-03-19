



# A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics

- [🔔News](#news)
- [🔔Table of Contents](#table-of-contents)
- [🔔Important Tables and Figures](#important-tables-and-figures)
  - [🔔LLM Information](#llm-information)
  - [🔔PLM Information](#plm-information)
  - [🔔Availble Training Data](#Availble-training-data)
- [🔔Citation](#citation)

## News
- **2023-03-17 update.new paper "Health-LLM: Personalized Retrieval-Augmented Disease Prediction System"**
- **2023-03-17 update.new paper "HealAI: A Healthcare LLM for Effective Medical Documentation"**
- **2023-03-17 update.new paper "BiMediX: Bilingual Medical Mixture of Experts LLM"**
- **2023-03-17 update.new paper "JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability"**
- **2023-03-17 update.new paper "MedChatZH: A tuning LLM for traditional Chinese medicine consultation"**
- **2023-10-18 added new paper "Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue".**
- **2023-10-18 added new paper "Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model".**
- **2023-10-9 We release the version 1 of the survey (https://arxiv.org/abs/2310.05694).**

## Table of Contents

- Introduction  
- What LLMs Can Do for Healthcare? From Fundamental Tasks to Advanced Applications 
  - NER and RE for Healthcare  
  - Text Classification for Healthcare 
  - Semantic Textual Similarity for Healthcare 
  - Question Answering for Healthcare 
  - Dialogue System for Healthcare 
  - Generation of Medical Reports from Images 
  - Summary 
- From LMs to LLMs for Healthcare 
  - LMs for Healthcare 
  - LLMs for Healthcare 

- Train and Use LLM for Healthcare 
  - Pre-training Methods 
  - Masked Language Modeling 
  - Next Word Prediction
  - Sequence-to-sequence MLM 
  - Replaced Token Detection 
  - Sentence Boundary Detection 
  - Next Sentence Prediction 
  - Sentence Order Prediction 
- Post-training Methods 
  - From predicting tokens to follow instructions: Instruction Fine-Tuning and Supervised Fine-tuning 
  - Reinforced Learning from Human Feedback 
  - From Human Feedback to AI Feedback 
  - Summary 
- Usage 
  - From Fine-tuning to In-context Learning 
  - From System 1 Deep Learning To System 2 Deep Learning: Chain-of-Thought 
  - AI Agents 
  - Summary 
- Parameters-, Memory-, and Compute-efficient Methods 
  - Parameters-efficient Methods 
  - Compute-efficient and Memory-efficient Methods 
- Useful Resources 
  - OpenBMB 
  - DeepSpeed Chat 
  - Training Data
  - Summary 
- Evaluation Method 
  - General NLP tasks Evaluation 
  - Healthcare Evaluation
  - Evaluation of Robustness, Bias, and Ethics 
  - Future Directions for Health Evaluation 
  - Summary 
- Improving Fairness, Accountability, Transparency, and Ethics 
  - Fairness 
  - Accountability 
  - Transparency 
  - Ethics 
- Future work and Conclusion 
  - Future Work 
  - Medical knowledge enhancement 
  - Integration with Healthcare process 
  - Effective Interaction with Patients and Doctors 
  - Hallucinations, Misunderstandings and Prompt Brittleness 
- Conclusion 




## Important Tables and Figures
Fig. 2. The organizational framework for the content. Section III, Section IV, Section V are technology details, while Section II, Section VI and Section VI
are more valued for Healthcare professionals
![Alt text](<LLM-for-Healthcare 3f514ac74dff42f49dcf565debf9898a/framework.png>)


---
<br> <br> 


### LLM Information 


| **Model Name**                             | **Base**       | **Para. (B)**     | **Features**                                    | **Date** | **Link**                                                               |
|--------------------------------------------|----------------|-------------------|-------------------------------------------------|----------|------------------------------------------------------------------------|
| GatorTron          | Transformer    | 0.345, 3.9, 8.9   | Training from scratch                           | 06/2022  |  https://github.com/uf-hobi-informatics-lab/GatorTron     |
| Codex-Med              | GPT-3.5        | 175               | CoT, Zero-shot                                  | 07/2022  |  https://github.com/vlievin/medical-reasoning             |
| Galactica        | Transformer    | 1.3, 6.4, 30, 120 | Reasoning, Multidisciplinary                    | 11/2022  |  https://galactica.org                                       |
| Med-PaLM            | Flan-PaLM/PaLM | 540               | CoT, Self-consistency                           | 12/2022  | -                                                                      |
| GPT-4-Med                   | GPT-4          | -                 | no specialized prompt crafting                  | 03/2023  | -                                                                      |
| DeID-GPT                 | GPT-4          | -                 | De-identifying                                  | 03/2023  |  https://github.com/yhydhx/ChatGPT-API                    |
| ChatDoctor    | LLaMA          | 7                 | Retrieve online, external knowledge             | 03/2023  |  https://github.com/Kent0n-Li/ChatDoctor                  |
| DoctorGLM         | ChatGLM        | 6                 | Extra prompt designer                           | 04/2023  |  https://github.com/xionghonglin/DoctorGLM                |
| MedAlpaca           | LLaMA          | 7, 13             | Adapt to Medicine                               | 04/2023  |  https://github.com/kbressem/medAlpaca                    |
| BenTsao               | LLaMA          | 7                 | Knowledge graph                                 | 04/2023  |  https://github.com/SCIR-HI/ Huatuo-Llama-Med-Chinese     |
| PMC-LLaMA                  | LLaMA          | 7                 | Adapt to Medicine                               | 04/2023  |  https://github.com/chaoyi-wu/PMC-LLaMA                   |
| Visual Med-Alpaca  | LLaMA          | 7                 | multimodal generative model, Self-Instruct      | 04/2023  |  https://github.com/cambridgeltl/visual-med-alpaca        |
| BianQue~            | ChatGLM        | 6                 | Chain of Questioning                            | 04/2023  |  https://github.com/scutcyr/BianQue                       |
| Med-PaLM 2        | PaLM 2         | 340               | Ensemble refinement, CoT, Self-consistency      | 05/2023  | -                                                                      |
| GatorTronGPT           | GPT-3          | 5, 20             | Training from scratch for medicine              | 05/2023  |  https://github.com/uf-hobi-informatics-lab/GatorTronGPT  |
| HuatuoGPT         | Bloomz         | 7                 | Reinforced learning from AI feedback            | 05/2023  |  https://github.com/FreedomIntelligence/HuatuoGPT         |
| ClinicalGPT      | BLOOM          | 7                 | multi-round dialogue consultations              | 06/2023  | -                                                                      |
| MedAGI                  | MiniGPT-4      | -                 | multimodal, AGI                                 | 06/2023  |  https://github.com/JoshuaChou2018/MedAGI                 |
| LLaVA-Med                | LLaVA          | 13                | multimodal, self-instruct,  curriculum learning | 06/2023  |  https://github.com/microsoft/LLaVA-Med                   |
| OphGLM                 | ChatGLM        | 6                 | multimodal, Ophthalmology LLM                   | 06/2023  |  https://github.com/ML-AILab/OphGLM                       |
| SoulChat            | ChatGLM        | 6                 | Mental Healthcare                               | 06/2023  |  https://github.com/scutcyr/SoulChat                      |
| Med-Flamingo             | Flamingo       | 80B               | multimodal, Few-Shot generative medical VQA     | 07/2023  |  https://github.com/snap-stanford/med-flamingo            |

---
<br> <br> 


### PLM Information 
TABLE I
BRIEF SUMMARIZATION OF EXISTING PLMS FOR HEALTHCARE.
| Model Name                                         | Base        | Para. (B) | Features                              | Date    | Link                                                                                                     |
|----------------------------------------------------|-------------|-----------|---------------------------------------|---------|----------------------------------------------------------------------------------------------------------|
| BioBERT                       | BERT        | 0.34      | Biomedical Adaption                   | 05/2019 |  https://github.com/naver/biobert-pretrained                                                |
| BlueBERT                    | BERT        | 0.34      | Biomedical Benchmark                  | 06/2019 |  https://github.com/ncbi-nlp/BLUE\_Benchmark                                                |
| MIMIC-BERT               | BERT        | 0.34      | Clinical Concept Extraction           | 08/2019 | -                                                                                                        |
| BioFLAIR~                 | BERT        | 0.34      | Less Computationally Intensive        | 08/2019 |  https://github.com/zalandoresearch/flair                                                   |
| Bio-ELECTRA-small    | ELECTRA     | 0.03      | Training From Scratch                 | 03/2020 | -                                                                                                        |
| AlphaBERT                   | BERT        | 0.11      | Character-level                       | 04/2020 |  https://github.com/wicebing/AlphaBERT.git                                                  |
| Spanish-bert             | BERT        | -         | Spanish                               | 04/2020 | -                                                                                                        |
| GreenCovidSQuADBERT   | BERT        | 0.34      | CPU-only, CORD-19                     | 04/2020 |  https://github.com/npoe/covid-qa                                                           |
| BEHRT                            | Transformer | -         | Training From Scratch                 | 04/2020 |  https://github.com/deepmedicine/BEHRT                                                      |
| BioMed-RoBERTa             | RoBERTa     | 0.11      | Biomedical Adaption                   | 05/2020 |  https://github.com/allenai/dont-stop-pretraining                                           |
| RadBERT~                        | BERT        | -         | RadCore Radiology Reports             | 05/2020 | -                                                                                                        |
| CT-BERT~                     | BERT        | 0.34      | COVID-19                              | 05/2020 |  https://github.com/digitalepidemiologylab/covid-twitter-bert                               |
| French-BERT  | BERT        | 0.11      | French Language Models                | 06/2020 | -                                                                                                        |
| FS-/RAD-/GER-BERT          | BERT        | 0.11      | Chest Radiograph Reports              | 07/2020 |  https://github.com/fast-raidiology/bertfor-radiology                                       |
| Japanese-BERT            | BERT        | 10.11     | Japanese Clinical Narrative           | 07/2020 |  ai-health.m.u-tokyo.ac.jp/home/research/uth-bert                                           |
| MC-BERT              | BERT        | 0.11      | Chinese Biomedical Benchmark          | 08/2020 |  https://github.com/alibabaresearch/ChineseBLUE                                             |
| BioALBERT-ner            | ALBERT      | 0.18      | Biomedical  NER                       | 09/2020 |  https://github.com/usmaann/BioALBERT                                                       |
| BioMegatron              | Megatron    | 1.2       | Training From Scratch                 | 10/2020 |  https://github.com/NVIDIA/NeMo                                                             |
| CharacterBERT            | BERT        | 0.11      | Character-CNN module                  | 10/2020 |  https://github.com/helboukkouri/character-bert                                             |
| ClinicalBert           | BERT        | 0.11      | For Predicting Hospital Readmission   | 11/2020 |  https://github.com/kexinhuang12345/clinicalBERT                                            |
| Clinical XLNet             | XLNet       | 0.11      | Temporal Information                  | 11/2020 |  https://github.com/lindvalllab/clinicalXLNet                                               |
| Bio-LM                   | RoBERTa     | 0.34      | Biomedical Adaption                   | 11/2020 |  https://github.com/facebookresearch/bio-lm                                                 |
| BioBERTpt      | BERT        | 0.11      | Portuguese Clinical                   | 11/2020 |  https://github.com/HAILab-PUCPR/BioBERTpt                                                  |
| RoBERTa-MIMIC               | RoBERTa     | 0.11      | Clinical Concept Extraction           | 12/2020 |  https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER                          |
| Clinical KB-ALBERT          | ALBERT      | 0.03      | Introducing Medical KB                | 12/2020 |  https://github.com/noc-lab/clinical-kb-bert                                                |
| CHMBERT                        | BERT        | 0.11      | Chinese Medical, Cloud Computing      | 01/2021 | -                                                                                                        |
| PubMedBERT                      | BERT        | 0.11      | Training From Scratch                 | 01/2021 |  https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  |
| ouBioBERT                        | BERT        | 0.11      | Up-sampling, Amplified Vocabulary     | 02/2021 |  https://github.com/sy-wada/blue\_benchmark\_with\_transformers                             |
| BERT-EHR               | BERT        | -         | Depression,Chronic Disease Prediction | 03/2021 |  https://github.com/lanyexiaosa/brltm                                                       |
| AraBERT                    | BERT        | 0.11      | Arabic Language                       | 03/2021 |  https://github.com/aub-mind/araBERT                                                        |
| ABioNER                | BERT        | 0.11      | Arabic NER                            | 03/2021 | -                                                                                                        |
| ELECTRAMed               | ELECTRA     | 0.11      | Biomedical Adaption                   | 04/2021 |  https://github.com/gmpoli/electramed                                                       |
| KeBioLM                    | PubMedBERT  | 0.11      | Introducing Medical KB                | 04/2021 |  https://github.com/GanjinZero/KeBioLM                                                      |
| SINA-BERT                 | BERT        | 0.11      | Persian Language                      | 04/2021 | -                                                                                                        |
| Med-BERT                        | BERT        | 0.11      | Stay Length Prediction                | 05/2021 |  https://github.com/ZhiGroup/MedBERT                                                        |
| Galén                  | RoBERTa     | 0.11      | Spanish Language                      | 05/2021 |  https://github.com/guilopgar/ClinicalCodingTransformerES                                   |
| SCIFIVE~                     | T5          | 0.77      | Biomedical Text Generation            | 05/2021 |  https://github.com/justinphan3110/SciFive                                                  |
| BioELECTRA  | ELECTRA     | 0.34      | Training From Scratch                 | 06/2021 |  https://github.com/kamalkraj/BioELECTRA                                                    |
| UmlsBERT                    | BERT        | 0.11      | Introducing Medical KB                | 06/2021 |  https://github.com/gmichalo/UmlsBERT                                                       |
| MedGPT                   | GPT-2       | 1.5       | Temporal Modelling                    | 07/2021 | -                                                                                                        |
| MentalBERT                  | BERT        | 0.11      | Mental Healthcare                     | 10/2021 |  https://huggingface.co/mental                                                         |
| CODER                          | mBERT       | 0.34      | Cross-lingual, Introducing Medical KB | 02/2022 |  https://github.com/GanjinZero/CODER                                                        |
| BioLinkBERT~            | BERT        | 0.34      | PubMed with Citation Links            | 03/2022 |  https://github.com/michiyasunaga/LinkBERT                                                  |
| BioALBERT             | ALBERT      | 0.03      | Biomedical Adaption                   | 04/2022 |  https://github.com/usmaann/BioALBERT                                                       |
| BioBART~                     | BART        | 0.4       | Biomedical NLG                        | 04/2022 |  https://github.com/GanjinZero/BioBART                                                      |
| SAPBERT                          | BERT        | 0.11      | Self-Alignment Pretraining            | 10/2022 |  https://github.com/cambridgeltl/sapbert                                                    |
| VPP                            | BART        | 0.14      | Soft prompt, Biomedical NER           | 03/2023 |  https://github.com/KaiHe-better/VPP                                                        |
| KAD               | BERT        | -         | Multimodal, Chest Radiology Images    | 03/2023 |  https://github.com/xiaoman-zhang/KAD                                                       |







---
<br> <br> 







TABLE II
SUMMARIZATION OF TRAINING DATA AND EVALUATION TASKS FOR
EXISTING PLMS FOR HEALTHCARE.

| Model Name                                         | Method | Training Data                  | Eval task                                             |
|----------------------------------------------------|--------|--------------------------------|-------------------------------------------------------|
| BioBERT                       | FT     | PubMed, PMC                    | Biomedical NER, RE, QA                                |
| BlueBert                    | FT     | PubMed, MIMIC-III              | BLUE                                                  |
| MIMIC-BERT               | FT     | MIMIC-III                      | Biomedical NER                                        |
| BioFLAIR~                 | FT     | PubMed                         | Bio NER                                               |
| Bio-ELECTRA-small    | PT     | PubMed                         | Biomedical NER                                        |
| AlphaBERT                   | FT     | Discharge diagnoses            | Extractive Summarization Task                         |
| Spanish-bert             | FT     | Spanish                        | Spanish Clinical Case Corpus                          |
| GreenCovidSQuADBERT   | FT     | CORD19, PubMed, PMC            | NER, QA                                               |
| BEHRT                            | PT     | CPRD, HES                      | Disease Prediction                                    |
| BioMed-RoBERTa             | FT     | BIOMED                         | CHEMPROT, RCT                                         |
| RadBERT~                        | FT     | Radiology Report Corpus        | Report Coding, Summarization                          |
| CT-BERT~                     | FT     | Tweet                          | COVID-19 Text Classification                          |
| French-BERT  | FT     | French clinical documents      | DEFT challenge                                        |
| FS-/RAD-/GER-BERT          | FT,PT  | Unstructured radiology reports | Chest Radiograph Reports Classification               |
| Japanese-BERT            | FT     | Japanese EHR                   | Symptoms Classification                               |
| MC-BERT              | FT     | Chinese EHR                    | Chinese Biomedical Evaluation benchmark               |
| BioALBERT-ner            | FT     | PubMed, PMC                    | Biomedical NER                                        |
| BioMegatron              | PT     | PubMed                         | biomedical NER, RE, QA                                |
| CharacterBERT            | Bert   | OpenWebText, MIMIC-III, PMC    | Medical NER, NLI, RE, SS                              |
| ClinicalBert           | FT     | MIMIC-III                      | Hospital Readmission Prediction                       |
| Clinical XLNet             | FT     | MIMIC-III                      | PMV, Mortality                                        |
| Bio-LM                   | FT     | PubMed, PMC, MIMIC-III         | 18 Biomedical NLP Tasks                               |
| BioBERTpt      | FT     | Private clinical notes, WMT16  | SemClinBr                                             |
| RoBERTa-MIMIC               | FT     | i2b2 2010, 2012, n2c2 2018     | i2b2 2010, 2012, N2C2 2018                            |
| Clinical KB-ALBERT          | FT     | MIMIC-III, UMLS                | MedNLI, i2b2 2010, 2012                               |
| CHMBERT                        | FT     | Medical text data              | Disease Prediction                                    |
| PubMedBERT                      | PT     | PubMed                         | BLURB                                                 |
| ouBioBERT                        | FT     | PubMed, Wikipedia              | BLUE                                                  |
| BERT-EHR               | FT     | General EHR                    | Myocardial Infarction, Breast Cancer, Liver Cirrhosis |
| AraBERT                    | PT     | Arabic Wikipedia, OSIAN        | Arabic SA, NER, QA                                    |
| ABioNER                | FT     | Arabic scientific literature   | Arabic NER                                            |
| ELECTRAMed               | FT     | PubMed                         | Biomedical NER, RE, and QA                            |
| KeBioLM                    | FT     | PubMed                         | BLURB                                                 |
| SINA-BERT                 | FT     | Online Persian source          | Persian  QA, SA                                       |
| Med-BERT                        | FT     | General EHR                    | Disease prediction                                    |
| Galén                  | FT     | Private clinical cases         | CodiEsp-D, CodiEsp-P, Cantemist-Coding tasks          |
| SCIFIVE~                     | T5     | PubMed, PMC                    | Biomedical NER, RE, NIL, QA                           |
| BioELECTRA  | PT     | PubMed, PMC                    | BLURB, BLUE                                           |
| UmlsBERT                    | FT     | MIMIC-III                      | MedNLI, i2b2 2006,2010, 2012, 2014                    |
| MedGPT                   | FT     | MIMIC-III, private EHRs        | Disorder Prediction                                   |
| MentalBERT                  | FT     | Reddit                         | Depression Stress, Suicide Detection,                 |
| CODER                          | FT     | UMLS                           | MCSM, Medical RE                                      |
| BioLinkBERT~            | FT     | PubMed                         | BLURB, USMLE                                          |
| BioALBERT             | FT     | PubMed, PMC, MIMIC-III         | 6 BioNLP Tasks                                        |
| BioBART~                     | FT     | PubMed                         | Biomedical EL, NER, QA, Dialogue, Summarization       |
| SAPBERT                          | FT     | UMLS                           | MEL                                                   |
| VPP                            | FT     | PubMed                         | Biomedical NER                                        |
| KAD               | FT     | MIMIC-CXR                      | PadChest, ChestXray14, CheXpert and ChestX-Det10      |




---
<br> <br> 


### Availble Training Data

| **Data**                                            | **Type**                                          | **size**                                                          | **Link**                                                                                                   |
|-----------------------------------------------------|---------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| MIMIC-III                                           | EHR                                               | 58,976 hospital admissions for 38,597 patients                    |  https://mimic.mit.edu/docs/iii/                                                            |
| MIMIC-IV                                            | EHR                                               | covering a decade of admissions between 2008 and 2019             |  https://mimic.mit.edu/docs/iv/                                                             |
| CPRD                          | EHR                                               | over 2,000 primary care practices and include 60 million patients |  https://cprd.com/data                                                                      |
| PubMed                                              | Scientific Literature                             | 35M citations and abstracts of biomedical literature              |  https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/                                             |
| PMC                                                 | Scientific Literature                             | 8 million full-text article records                               |  https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk                                              |
| RCT                   | Scientific Literature                             | 4,528 abstract                                                    |  https://github.com/bwallace/RCT-summarization-data                                        |
| MS$\hat{~}$2             | Scientific Literature                             | 470,402 abstract                                                  |  https://github.com/allenai/ms2/                                                           |
| CDSR                         | Scientific Literature                             | 7,805 abstract                                                    |  https://github.com/qiuweipku/Plain\_language\_summarization                               |
| SumPubMed                          | Scientific Literature                             | 33,772 abstract                                                   |  https://github.com/vgupta123/sumpubmed                                                    |
| The Pile                                            | Scientific Literature                             | 825 GB English text                                               |  https://pile.eleuther.ai/                                                                 |
| S2ORC                 | Scientific Literature                             | 63,709 abstract                                                   |  https://github.com/jbshp/GenCompareSum                                                    |
| CORD-19                   | Scientific Literature                             | 1M papers                                                         |  https://github.com/allenai/cord19                                                         |
| MeQSum                                 | Medical Question Summarization                    | 1000 instances                                                    |  https://github.com/abachaa/MeQSum                                                         |
| CHQ-Sum                          | Medical Question Summarization                    | 1507 instances                                                    |  https://github.com/shwetanlp/Yahoo-CHQ-Summ                                               |
| UMLS                                                | Knowledge Base                                    | 2M entities for 900K concepts                                     |  https://www.nlm.nih.gov/research/umls/index.html                                           |
| COMETA                   | Web Data (social media)                           | 800K Reddit posts                                                 |  https://github.com/cambridgeltl/cometa                                                     |
| MedDialog                   | Dialogue                                          | 3.66 million conversations                                        |  https://github.com/UCSD-AI4H/COVID-Dialogue                                                |
| CovidDialog                 | Dialogue                                          | 603 consultations                                                 |  https://github.com/UCSD-AI4H/COVID-Dialogue                                                |
| Medical Flashcards           | Dialogue                                          | 33955 instances                                                   |  https://github.com/kbressem/medalpaca                                                     |
| Wikidoc                      | Dialogue                                          | 67704 instances                                                   |  https://huggingface.co/datasets/medalpaca/medical\_meadow\_wikidoc                        |
| Wikidoc Patient Information  | Dialogue                                          | 5942 instances                                                    |  https://huggingface.co/datasets/medalpaca/medical\_meadow\_wikidoc\_patient\_information  |
| MEDIQA                     | Dialogue                                          | 2208 instances                                                    |  https://huggingface.co/datasets/medalpaca/medical\_meadow\_wikidoc\_patient\_information  |
| CORD-19                   | Dialogue                                          | 1056660 instances                                                 |  https://huggingface.co/datasets/medalpaca/medical\_meadow\_cord19                         |
| MMMLU                     | Dialogue                                          | 3787 instances                                                    |  https://huggingface.co/datasets/medalpaca/medical\_meadow\_mmmlu                          |
| Pubmed Causal          | Dialogue                                          | 2446 instances                                                    |  https://huggingface.co/datasets/medalpaca/medical\_meadow\_pubmed\_causal                 |
| ChatDoctor                   | Dialogue                                          | 215000 instances                                                  |  https://github.com/Kent0n-Li/ChatDoctor                                                   |
| Alpaca-EN-AN                           | English Instructions                              | 52K instructions                                                  |  https://github.com/tatsu-lab/stanford\_alpaca/blob/main/alpaca\_data.json                 |
| Alpaca-CH-AN                           | Chinese Instructions                              | 52K instructions                                                  |  https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main/data                 |
| ShareGPT                                            | Conversations                                     | 61653 long conversations                                          |  https://huggingface.co/datasets/philschmid/sharegpt-raw                                   |
| WebText                                             | Web Data                                          | 40 GB of text                                                     |  https://commoncrawl.org/the-data/get-started/                                             |
| OpenWebText                                         | Web Data                                          | 38 GB of text                                                     |  https://skylion007.github.io/OpenWebTextCorpus/                                           |
| Colossal Clean Crawled Corpus                       | Web Data                                          | 806 GB of text                                                    |  https://www.tensorflow.org/datasets/catalog/c4                                            |
| OpenI                                               | EHR, Multimodel                                   | 3.7 million images from about 1.2 million papers                  |  https://openi.nlm.nih.gov/faq\#collection                                                  |
| U-Xray                    | Multimodel                                        | 3,955 reports and 7,470 images                                    |  https://openi.nlm.nih.gov/                                                                 |
| ROCO                       | Multimodel                                        | 81,000 radiology images and corresponding captions                |  https://github.com/razorx89/roco-dataset                                                   |
| MedICaT                | Multimodel                                        | 17,000 images includes captions                                   |  https://github.com/allenai/medicat                                                         |
| PMC-OA                             | Multimodel                                        | 1.6M image-caption pairs                                          |  https://huggingface.co/datasets/axiong/pmc\_oa\_beta                                       |
| CheXpert                    | Multimodel                                        | 224,316 chest radiographs with associated reports                 |  https://aimi.stanford.edu/chexpert-chest-x-rays                                            |
| PadChest                   | Multimodel                                        | 160,000 images  with related text                                 |  http://bimcv.cipf.es/bimcv-projects/padchest/                                              |
| MIMIC-CXR                                           | Multimodel                                        | 227,835 imaging studies for 64,588 patients                       |  https://mimic.mit.edu/docs/iv/modules/cxr/                                                 |
| PMC-15M                        | Multimodel                                        | 15 million Figure-caption                                         |
| pairs                                               |  https://arxiv.org/abs/2303.00915  |
| OpenPath                             | Multimodel                                        | 208,414 pathology images related descriptions                     |  https://laion.ai/blog/laion-5b/                                                            |





### The Statistics of Computation Cost

TABLE VIII
THE STATISTICS OF COMPUTATION COST FOR EXISTING HEALTHCARE
LLM.
| **Model Name**    | **Total data size**             | **epoch** | **Batch size** | **GPU type** | **GPU number** | **GPU time** |
|-------------------|---------------------------------|-----------|----------------|--------------|----------------|--------------|
| Visual Med-Alpaca | 54k data points                 | 3         | 128            | A100-80G     | 4              | 2.51 hours   |
| GatorTron         | \textgreater 90 billion words   | 10        | -              | A100         | 992            | 6 days       |
| Galactica         | -                               | -         | -              | A100-80G     | 128            | -            |
| ChatDoctor        | 100k conversations              | 3         | 192            | A100         | 6              | 3 hours      |
| DoctorGLM         | 3.5G                            | 1         | 4              | A100-80G     | 1              | 8 hours      |
| PMC-LLaMA         | 75B tokens                      | 5         | 128            | A100         | 8              | 7 days       |
| Visual Med-Alpaca | 44.8MB* (without   images)      | -         | 128            | A100-80G     | 4              | 2.51 hours   |
| BianQue 1.0       | 9 million samples               | 1         | -              | RTX 4090     | 8              | 16 days      |
| GatorTronGPT      | 277B tokens                     |           | 1,120/560      | A100-80G     | 560            | 26 days      |
| HuatuoGPT         | 226,042 instances               | 3         | 128            | A100         | 8              | -            |
| LLaVA-Med         | 15 million figure-caption pairs | -         | -              | A100         | 8              | 15 hours     |
| Med-Flamingo      | 1.3M image-caption pairs        | -         | 400            | A100-80G     | 8              | 6.75 days    |


---
<br> <br> 

TABLE IX
ESTIMATED FLOPS AND TRAINING TOKENS FOR DIFFERENT MODEL
SIZES.
| **Parameters** | **FLOPs** | **FLOPs   (in Gopher unit)** | **Tokens**       |
|----------------|-----------|------------------------------|------------------|
| 400   Million  | 1.92e+19  | 1/29,   968                  | 8.0   Billion    |
| 1   Billion    | 1.21e+20  | 1/4, 761                     | 20.2   Billion   |
| 10   Billion   | 1.23e+22  | 1/46                         | 205.1   Billion  |
| 67   Billion   | 5.76e+23  | 1                            | 1.5   Trillion   |
| 175   Billion  | 3.85e+24  | 6.7                          | 3.7 Trillion     |
| 280   Billion  | 9.90e+24  | 17.2                         | 5.9   Trillion   |
| 520   Billion  | 3.43e+25  | 59.5                         | 11.0   Trillion  |
| 1   Trillion   | 1.27e+26  | 221.3                        | 21.2   Trillion  |
| 10   Trillion  | 1.30e+28  | 22515.9                      | 216.2   Trillion |



## Citation


```bibtex
@misc{he2023survey,
      title={A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics}, 
      author={Kai He and Rui Mao and Qika Lin and Yucheng Ruan and Xiang Lan and Mengling Feng and Erik Cambria},
      year={2023},
      eprint={2310.05694},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
