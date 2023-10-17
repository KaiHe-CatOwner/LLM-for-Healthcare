



# A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics

- [🔔News](#news)
- [🔔Table of Contents](#table-of-contents)
- [🔔Important Tables and Figures](#important-tables-and-figures)
- [🔔Citation](#citation)

## News
- **2023-10-9 We release the version 1 of the survey (https://arxiv.org/abs/2310.05694)".**


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
---

TABLE I
BRIEF SUMMARIZATION OF EXISTING PLMS FOR HEALTHCARE.
| Model Name                                         | Base        | Para. (B) | Features                              | Date    | Link                                                                                                     |
|----------------------------------------------------|-------------|-----------|---------------------------------------|---------|----------------------------------------------------------------------------------------------------------|
| BioBERT~\cite{lee2020biobert}                      | BERT        | 0.34      | Biomedical Adaption                   | 05/2019 | \href{https://github.com/naver/biobert-pretrained}{Github}                                               |
| BlueBERT~\cite{peng2019transfer}                   | BERT        | 0.34      | Biomedical Benchmark                  | 06/2019 | \href{https://github.com/ncbi-nlp/BLUE\_Benchmark}{Github}                                               |
| MIMIC-BERT~\cite{kraljevic2021medgpt}              | BERT        | 0.34      | Clinical Concept Extraction           | 08/2019 | -                                                                                                        |
| BioFLAIR~~\cite{sharma2019bioflair}                | BERT        | 0.34      | Less Computationally Intensive        | 08/2019 | \href{https://github.com/zalandoresearch/flair}{Github}                                                  |
| Bio-ELECTRA-small~\cite{ozyurt2020effectiveness}   | ELECTRA     | 0.03      | Training From Scratch                 | 03/2020 | -                                                                                                        |
| AlphaBERT~\cite{chen2020modified}                  | BERT        | 0.11      | Character-level                       | 04/2020 | \href{https://github.com/wicebing/AlphaBERT.git}{Github}                                                 |
| Spanish-bert~\cite{akhtyamova2020named}            | BERT        | -         | Spanish                               | 04/2020 | -                                                                                                        |
| GreenCovidSQuADBERT~\cite{poerner2020inexpensive}  | BERT        | 0.34      | CPU-only, CORD-19                     | 04/2020 | \href{https://github.com/npoe/covid-qa}{Github}                                                          |
| BEHRT~\cite{li2020behrt}                           | Transformer | -         | Training From Scratch                 | 04/2020 | \href{https://github.com/deepmedicine/BEHRT}{Github}                                                     |
| BioMed-RoBERTa~\cite{gururangan2020don}            | RoBERTa     | 0.11      | Biomedical Adaption                   | 05/2020 | \href{https://github.com/allenai/dont-stop-pretraining}{Github}                                          |
| RadBERT~~\cite{meng2020self}                       | BERT        | -         | RadCore Radiology Reports             | 05/2020 | -                                                                                                        |
| CT-BERT~~\cite{muller2023covid}                    | BERT        | 0.34      | COVID-19                              | 05/2020 | \href{https://github.com/digitalepidemiologylab/covid-twitter-bert}{Github}                              |
| French-BERT~\cite{copara-etal-2020-contextualized} | BERT        | 0.11      | French Language Models                | 06/2020 | -                                                                                                        |
| FS-/RAD-/GER-BERT~\cite{bressem2020highly}         | BERT        | 0.11      | Chest Radiograph Reports              | 07/2020 | \href{https://github.com/fast-raidiology/bertfor-radiology}{Github}                                      |
| Japanese-BERT~\cite{kawazoe2020clinical}           | BERT        | 10.11     | Japanese Clinical Narrative           | 07/2020 | \href{ai-health.m.u-tokyo.ac.jp/home/research/uth-bert}{Github}                                          |
| MC-BERT~\cite{zhang2020conceptualized}             | BERT        | 0.11      | Chinese Biomedical Benchmark          | 08/2020 | \href{https://github.com/alibabaresearch/ChineseBLUE}{Github}                                            |
| BioALBERT-ner~\cite{naseem2021bioalbert}           | ALBERT      | 0.18      | Biomedical  NER                       | 09/2020 | \href{https://github.com/usmaann/BioALBERT}{Github}                                                      |
| BioMegatron~\cite{shin2020biomegatron}             | Megatron    | 1.2       | Training From Scratch                 | 10/2020 | \href{https://github.com/NVIDIA/NeMo}{Github}                                                            |
| CharacterBERT~\cite{kraljevic2021medgpt}           | BERT        | 0.11      | Character-CNN module                  | 10/2020 | \href{https://github.com/helboukkouri/character-bert}{Github}                                            |
| ClinicalBert~\cite{huang2019clinicalbert}          | BERT        | 0.11      | For Predicting Hospital Readmission   | 11/2020 | \href{https://github.com/kexinhuang12345/clinicalBERT}{Github}                                           |
| Clinical XLNet~\cite{huang2020clinical}            | XLNet       | 0.11      | Temporal Information                  | 11/2020 | \href{https://github.com/lindvalllab/clinicalXLNet}{Github}                                              |
| Bio-LM~\cite{lewis2020pretrained}                  | RoBERTa     | 0.34      | Biomedical Adaption                   | 11/2020 | \href{https://github.com/facebookresearch/bio-lm}{Github}                                                |
| BioBERTpt~\cite{schneider-etal-2020-biobertpt}     | BERT        | 0.11      | Portuguese Clinical                   | 11/2020 | \href{https://github.com/HAILab-PUCPR/BioBERTpt}{Github}                                                 |
| RoBERTa-MIMIC~\cite{yang2020clinical}              | RoBERTa     | 0.11      | Clinical Concept Extraction           | 12/2020 | \href{https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER}{Github}                         |
| Clinical KB-ALBERT~\cite{hao2020enhancing}         | ALBERT      | 0.03      | Introducing Medical KB                | 12/2020 | \href{https://github.com/noc-lab/clinical-kb-bert}{Github}                                               |
| CHMBERT~\cite{wang2021cloud}                       | BERT        | 0.11      | Chinese Medical, Cloud Computing      | 01/2021 | -                                                                                                        |
| PubMedBERT~\cite{gu2021domain}                     | BERT        | 0.11      | Training From Scratch                 | 01/2021 | \href{https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext}{Huggingface} |
| ouBioBERT~\cite{wada2020pre}                       | BERT        | 0.11      | Up-sampling, Amplified Vocabulary     | 02/2021 | \href{https://github.com/sy-wada/blue\_benchmark\_with\_transformers}{Github}                            |
| BERT-EHR~\cite{meng2021bidirectional}              | BERT        | -         | Depression,Chronic Disease Prediction | 03/2021 | \href{https://github.com/lanyexiaosa/brltm}{Github}                                                      |
| AraBERT~\cite{antoun2020arabert}                   | BERT        | 0.11      | Arabic Language                       | 03/2021 | \href{https://github.com/aub-mind/araBERT}{Github}                                                       |
| ABioNER~\cite{boudjellal2021abioner}               | BERT        | 0.11      | Arabic NER                            | 03/2021 | -                                                                                                        |
| ELECTRAMed~\cite{miolo2021electramed}              | ELECTRA     | 0.11      | Biomedical Adaption                   | 04/2021 | \href{https://github.com/gmpoli/electramed}{Github}                                                      |
| KeBioLM~\cite{yuan2021improving}                   | PubMedBERT  | 0.11      | Introducing Medical KB                | 04/2021 | \href{https://github.com/GanjinZero/KeBioLM}{Github}                                                     |
| SINA-BERT~\cite{taghizadeh2021sina}                | BERT        | 0.11      | Persian Language                      | 04/2021 | -                                                                                                        |
| Med-BERT~\cite{rasmy2021med}                       | BERT        | 0.11      | Stay Length Prediction                | 05/2021 | \href{https://github.com/ZhiGroup/MedBERT}{Github}                                                       |
| Galén~\cite{lopez2021transformers}                 | RoBERTa     | 0.11      | Spanish Language                      | 05/2021 | \href{https://github.com/guilopgar/ClinicalCodingTransformerES}{Github}                                  |
| SCIFIVE~~\cite{phan2021scifive}                    | T5          | 0.77      | Biomedical Text Generation            | 05/2021 | \href{https://github.com/justinphan3110/SciFive}{Github}                                                 |
| BioELECTRA~\cite{kanakarajan-etal-2021-bioelectra} | ELECTRA     | 0.34      | Training From Scratch                 | 06/2021 | \href{https://github.com/kamalkraj/BioELECTRA}{Github}                                                   |
| UmlsBERT~\cite{hao2020enhancing}                   | BERT        | 0.11      | Introducing Medical KB                | 06/2021 | \href{https://github.com/gmichalo/UmlsBERT}{Github}                                                      |
| MedGPT~\cite{kraljevic2021medgpt}                  | GPT-2       | 1.5       | Temporal Modelling                    | 07/2021 | -                                                                                                        |
| MentalBERT~\cite{ji2021mentalbert}                 | BERT        | 0.11      | Mental Healthcare                     | 10/2021 | \href{https://huggingface.co/mental}{huggingface}                                                        |
| CODER~\cite{yuan2022coder}                         | mBERT       | 0.34      | Cross-lingual, Introducing Medical KB | 02/2022 | \href{https://github.com/GanjinZero/CODER}{Github}                                                       |
| BioLinkBERT~~\cite{yasunaga2022linkbert}           | BERT        | 0.34      | PubMed with Citation Links            | 03/2022 | \href{https://github.com/michiyasunaga/LinkBERT}{Github}                                                 |
| BioALBERT~\cite{naseem2022benchmarking}            | ALBERT      | 0.03      | Biomedical Adaption                   | 04/2022 | \href{https://github.com/usmaann/BioALBERT}{Github}                                                      |
| BioBART~~\cite{yuan2022biobart}                    | BART        | 0.4       | Biomedical NLG                        | 04/2022 | \href{https://github.com/GanjinZero/BioBART}{Github}                                                     |
| SAPBERT~\cite{liu2020self}                         | BERT        | 0.11      | Self-Alignment Pretraining            | 10/2022 | \href{https://github.com/cambridgeltl/sapbert}{Github}                                                   |
| VPP~\cite{he2023virtual}                           | BART        | 0.14      | Soft prompt, Biomedical NER           | 03/2023 | \href{https://github.com/KaiHe-better/VPP}{Github}                                                       |
| KAD~\cite{zhang2023knowledgeenhanced}              | BERT        | -         | Multimodal, Chest Radiology Images    | 03/2023 | \href{https://github.com/xiaoman-zhang/KAD}{Github}                                                      |

---
---

TABLE II
SUMMARIZATION OF TRAINING DATA AND EVALUATION TASKS FOR
EXISTING PLMS FOR HEALTHCARE.

| Model Name                                         | Method | Training Data                  | Eval task                                             |
|----------------------------------------------------|--------|--------------------------------|-------------------------------------------------------|
| BioBERT~\cite{lee2020biobert}                      | FT     | PubMed, PMC                    | Biomedical NER, RE, QA                                |
| BlueBert~\cite{peng2019transfer}                   | FT     | PubMed, MIMIC-III              | BLUE                                                  |
| MIMIC-BERT~\cite{kraljevic2021medgpt}              | FT     | MIMIC-III                      | Biomedical NER                                        |
| BioFLAIR~~\cite{sharma2019bioflair}                | FT     | PubMed                         | Bio NER                                               |
| Bio-ELECTRA-small~\cite{ozyurt2020effectiveness}   | PT     | PubMed                         | Biomedical NER                                        |
| AlphaBERT~\cite{chen2020modified}                  | FT     | Discharge diagnoses            | Extractive Summarization Task                         |
| Spanish-bert~\cite{akhtyamova2020named}            | FT     | Spanish                        | Spanish Clinical Case Corpus                          |
| GreenCovidSQuADBERT~\cite{poerner2020inexpensive}  | FT     | CORD19, PubMed, PMC            | NER, QA                                               |
| BEHRT~\cite{li2020behrt}                           | PT     | CPRD, HES                      | Disease Prediction                                    |
| BioMed-RoBERTa~\cite{gururangan2020don}            | FT     | BIOMED                         | CHEMPROT, RCT                                         |
| RadBERT~~\cite{meng2020self}                       | FT     | Radiology Report Corpus        | Report Coding, Summarization                          |
| CT-BERT~~\cite{muller2023covid}                    | FT     | Tweet                          | COVID-19 Text Classification                          |
| French-BERT~\cite{copara-etal-2020-contextualized} | FT     | French clinical documents      | DEFT challenge                                        |
| FS-/RAD-/GER-BERT~\cite{bressem2020highly}         | FT,PT  | Unstructured radiology reports | Chest Radiograph Reports Classification               |
| Japanese-BERT~\cite{kawazoe2020clinical}           | FT     | Japanese EHR                   | Symptoms Classification                               |
| MC-BERT~\cite{zhang2020conceptualized}             | FT     | Chinese EHR                    | Chinese Biomedical Evaluation benchmark               |
| BioALBERT-ner~\cite{naseem2021bioalbert}           | FT     | PubMed, PMC                    | Biomedical NER                                        |
| BioMegatron~\cite{shin2020biomegatron}             | PT     | PubMed                         | biomedical NER, RE, QA                                |
| CharacterBERT~\cite{kraljevic2021medgpt}           | Bert   | OpenWebText, MIMIC-III, PMC    | Medical NER, NLI, RE, SS                              |
| ClinicalBert~\cite{huang2019clinicalbert}          | FT     | MIMIC-III                      | Hospital Readmission Prediction                       |
| Clinical XLNet~\cite{huang2020clinical}            | FT     | MIMIC-III                      | PMV, Mortality                                        |
| Bio-LM~\cite{lewis2020pretrained}                  | FT     | PubMed, PMC, MIMIC-III         | 18 Biomedical NLP Tasks                               |
| BioBERTpt~\cite{schneider-etal-2020-biobertpt}     | FT     | Private clinical notes, WMT16  | SemClinBr                                             |
| RoBERTa-MIMIC~\cite{yang2020clinical}              | FT     | i2b2 2010, 2012, n2c2 2018     | i2b2 2010, 2012, N2C2 2018                            |
| Clinical KB-ALBERT~\cite{hao2020enhancing}         | FT     | MIMIC-III, UMLS                | MedNLI, i2b2 2010, 2012                               |
| CHMBERT~\cite{wang2021cloud}                       | FT     | Medical text data              | Disease Prediction                                    |
| PubMedBERT~\cite{gu2021domain}                     | PT     | PubMed                         | BLURB                                                 |
| ouBioBERT~\cite{wada2020pre}                       | FT     | PubMed, Wikipedia              | BLUE                                                  |
| BERT-EHR~\cite{meng2021bidirectional}              | FT     | General EHR                    | Myocardial Infarction, Breast Cancer, Liver Cirrhosis |
| AraBERT~\cite{antoun2020arabert}                   | PT     | Arabic Wikipedia, OSIAN        | Arabic SA, NER, QA                                    |
| ABioNER~\cite{boudjellal2021abioner}               | FT     | Arabic scientific literature   | Arabic NER                                            |
| ELECTRAMed~\cite{miolo2021electramed}              | FT     | PubMed                         | Biomedical NER, RE, and QA                            |
| KeBioLM~\cite{yuan2021improving}                   | FT     | PubMed                         | BLURB                                                 |
| SINA-BERT~\cite{taghizadeh2021sina}                | FT     | Online Persian source          | Persian  QA, SA                                       |
| Med-BERT~\cite{rasmy2021med}                       | FT     | General EHR                    | Disease prediction                                    |
| Galén~\cite{lopez2021transformers}                 | FT     | Private clinical cases         | CodiEsp-D, CodiEsp-P, Cantemist-Coding tasks          |
| SCIFIVE~~\cite{phan2021scifive}                    | T5     | PubMed, PMC                    | Biomedical NER, RE, NIL, QA                           |
| BioELECTRA~\cite{kanakarajan-etal-2021-bioelectra} | PT     | PubMed, PMC                    | BLURB, BLUE                                           |
| UmlsBERT~\cite{hao2020enhancing}                   | FT     | MIMIC-III                      | MedNLI, i2b2 2006,2010, 2012, 2014                    |
| MedGPT~\cite{kraljevic2021medgpt}                  | FT     | MIMIC-III, private EHRs        | Disorder Prediction                                   |
| MentalBERT~\cite{ji2021mentalbert}                 | FT     | Reddit                         | Depression Stress, Suicide Detection,                 |
| CODER~\cite{yuan2022coder}                         | FT     | UMLS                           | MCSM, Medical RE                                      |
| BioLinkBERT~~\cite{yasunaga2022linkbert}           | FT     | PubMed                         | BLURB, USMLE                                          |
| BioALBERT~\cite{naseem2022benchmarking}            | FT     | PubMed, PMC, MIMIC-III         | 6 BioNLP Tasks                                        |
| BioBART~~\cite{yuan2022biobart}                    | FT     | PubMed                         | Biomedical EL, NER, QA, Dialogue, Summarization       |
| SAPBERT~\cite{liu2020self}                         | FT     | UMLS                           | MEL                                                   |
| VPP~\cite{he2023virtual}                           | FT     | PubMed                         | Biomedical NER                                        |
| KAD~\cite{zhang2023knowledgeenhanced}              | FT     | MIMIC-CXR                      | PadChest, ChestXray14, CheXpert and ChestX-Det10      |























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