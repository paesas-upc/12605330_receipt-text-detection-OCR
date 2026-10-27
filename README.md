# FairShare: Splitting Trip Costs from Receipt Photos

> **Applied Deep Learning - Assignment 1: Initiate**  
> **Student:** Pau Escobar Asensio  
> **Matrikelnummer:** 12605330  
> **Main project type:** Bring your own method  
> **Topic:** Computer vision - receipt text detection and OCR

## Project summary

I am an Erasmus student, and I often travel around Europe on weekends with people from my student residence. We use Tricount to keep track of shared costs. At the moment, we can enter the total price and split it among everyone, but this doesn't account for who bought or used each item. Someone who bought only a few things can end up paying the same as someone who bought much more.

For this project, I want to build a receipt-reading prototype that can recognize the text on a shopping receipt, especially the item names, quantities, units, and prices. The computer vision component will be the main contribution: I will implement and adapt a deep learning OCR method, then evaluate its performance on a public receipt dataset. If the OCR output is reliable enough, a later stage can organize the recognized text into item rows. Users will review the rows and assign items, or parts of items, to people in the group. The app will calculate each person's share from those assignments instead of dividing the receipt total equally. The model cannot know who consumed a product, so participants will make those assignments themselves.

My main goal is to develop and evaluate a computer vision method for OCR on receipt images. As a secondary extension, if time and computing resources allow, I will use a small language model to turn the recognized text into structured item information. I will also try to improve a recently published OCR result on KORIE, but beating it is an experiment rather than a guaranteed outcome. If time allows, I would like to test the finished system on a small set of my own receipt photos.

## Topic and project type

- **Topic:** Computer vision and deep learning for text detection and OCR on receipts.
- **Main project type:** Bring your own method. I will implement and adapt an OCR neural network for receipt text, and evaluate it on an existing public dataset.
- **Secondary objective:** Try to improve a recently published OCR baseline on the same dataset, using a comparable test split and evaluation protocol.
- **Optional extension:** Convert recognized text into structured item rows with a small language model, then connect the result to a cost-splitting demo.
- **Optional test set:** Test the pipeline on a small, separately collected set of receipt photos.

The main research work is the OCR method and its evaluation. Structured extraction, language-model fine-tuning, and the application are extensions that make the project more complete if time permits.

## Dataset

### Primary dataset: KORIE

I plan to use [KORIE](https://github.com/MahmoudSalah/KORIE), a public Korean retail receipt dataset introduced in 2026. It contains receipt images, OCR text crops and transcriptions, and structured annotations for receipt fields and item lines. The paper describes receipt-level training, validation, and test splits, which helps keep different crops or lines from the same receipt out of more than one split.

The dataset is relatively small: the paper reports 748 receipts, while the repository README reports 774. The data are provided via external download links, so I will first confirm I can access the required files and that they match the paper's annotation format. KORIE receipts are Korean, not European, and the dataset is limited in size. It is suitable for developing and evaluating receipt OCR, but results on it will not prove that the model works equally well on other European receipts.

The authors report the following baselines:

| Task | Reported baseline | Result |
|---|---|---:|
| Text detection | YOLOv11 | mAP@0.50: 0.888; mAP@0.50:0.95: 0.762 |
| OCR recognition | PaddleOCR | CER: 15.84%; WER: 26.73% |
| OCR recognition | Attention-based BiGRU | CER: 22.69%; WER: 44.47% |
| Information extraction | Qwen2.5-3B-Instruct, zero-shot | Overall F1: 25% |

OCR scores can depend on whether the recognizer receives ground-truth text crops or crops predicted by a detector. I will check the KORIE evaluation setup and report comparisons only when the crop source and test protocol are sufficiently comparable. The main benchmark will be OCR recognition, measured with character error rate (CER) and word error rate (WER). Text detection results will be useful context, but detector improvement is not the primary method contribution. The information-extraction score is included for context only; it is not the main target of this project.

### Fallback dataset: CORU / ReceiptSense

If I cannot access or use KORIE's image and label files, I will check [CORU, also described as ReceiptSense](https://huggingface.co/datasets/abdoelsayed/CORU) as a fallback. It includes receipt detection, OCR, item information, and question-answering resources. However, the item information is presented separately from the receipt images, and I have not confirmed that its item rows can be linked to particular receipt images. I will use it for an image-to-item experiment only if that linkage is verified. Otherwise, it may still support a narrower OCR experiment, with the project scope adjusted accordingly.

### Optional personal receipt photos

If time allows, I will collect receipt photos from purchases made by me or friends who agree to contribute them. This small set would be used only as an exploratory check of how the model handles European receipts.

## Proposed method and experiments

The main experiment will focus on recognizing text from receipt image crops. I will first establish a reproducible baseline using a compact OCR recognizer, such as a CNN-BiGRU model trained with the Connectionist Temporal Classification (CTC) loss. I will then implement a modified recognizer that adds a lightweight visual self-attention or Transformer encoder to the CNN feature sequence before decoding. This lets me test whether attention helps the model use context across a line of receipt text while keeping the network small enough for free Colab or Kaggle GPU sessions.

I will use the OCR crops and transcriptions supplied with KORIE where possible. If the project also uses full receipt images, a text detector can first locate text regions and pass the crops to the recognizer. I will keep the detector fixed for the main recognizer comparison so the experiment measures the OCR method itself. I will document whether each result uses annotated crops or detector-predicted crops, since these settings can produce different scores.

If the main OCR work is complete and time remains, I will add a secondary post-processing step. A small language model could organize the recognized text into item names, quantities, units, and prices. This step depends on OCR output and is not the project's main research contribution. Fine-tuning the language model or trying a metric-aware reward method will be considered only as a stretch goal; the ReceiptBench paper's full SFT-plus-GRPO experiments used four A800 GPUs, which I cannot assume will be available to me. The core OCR experiment will be designed to run with compact models on free Colab or Kaggle GPU resources.

### Baselines and evaluation

- **Primary baseline:** A computer-vision OCR recognizer implemented for this project.
- **Proposed method:** The same recognizer with a lightweight attention or Transformer component added to its visual feature sequence.
- **Published comparison:** KORIE's reported OCR baselines, including PaddleOCR and its attention-based BiGRU, where the crop source, split, and metric allow a fair comparison.
- **Primary metric:** Character error rate (CER), where lower is better.
- **Secondary metrics:** Word error rate (WER), training time, model size, and, where applicable, text-detection mAP.
- **Evaluation protocol:** Keep KORIE's receipt-level test set separate from training and tuning. Select settings on the training and validation splits, and report the final test result once the method is fixed.
- **Optional extraction evaluation:** If the language-model stage is completed, evaluate structured fields such as item name, quantity, and price separately. This result will be reported as an extension and will not replace the OCR evaluation.
- **Secondary goal:** Claim an improvement over a published KORIE result only if it is higher under a comparable split, crop setup, and evaluation metric. Results on my optional personal set will be reported separately as exploratory.

## Application to present the results

I plan to build a small Streamlit application. A user will upload a receipt photo, review and correct the OCR text and item rows, and assign each item or quantity to one or more participants. The application will then calculate each person's share based on those assignments. It will show the receipt total and the resulting shares so the group can check that the split adds up.

The application is a demonstration of how the OCR method could be used. The model may misread text or prices, so users need to be able to correct its output before using the calculated shares. If the optional language-model stage is not feasible, the application can still present the OCR text and allow users to enter or correct item rows themselves.

## Feasibility and risks

- **Free GPU availability:** Colab and Kaggle GPU access and session lengths can vary. A compact OCR model, moderate image sizes, small batches, and saved checkpoints will help keep the main experiment manageable.
- **KORIE access and version details:** The dataset is downloaded from external links, and the paper and repository list different receipt counts. Checking access, files, splits, and annotations is the first dataset task.
- **Language mismatch:** KORIE receipts are Korean, while my intended use is European trips. My optional receipt set may give an initial indication of this gap, but it will be too small to establish broad performance on European receipts.
- **Fair OCR comparison:** OCR scores depend on crop quality, so I will record whether each recognizer uses annotated or detector-predicted crops and avoid presenting non-comparable results as a direct ranking.
- **Optional language-model compute:** Fine-tuning and metric-aware reward optimization can require substantially more compute than the OCR experiment. These are stretch goals and are not required for the core project.
- **Correct item assignment is a separate user decision:** A receipt does not reveal who consumed an item. The application will ask users to assign items rather than making that decision automatically.

## Work breakdown and time estimates

| Work package | Planned tasks | Estimate |
|---|---|---:|
| **Literature review and experiment design** | Read the dataset and OCR method papers; settle the model comparisons, metrics, and evaluation protocol. | 4 h |
| **Dataset collection and preparation** | Download and inspect KORIE; verify image-label links and receipt-level splits; prepare OCR crops and transcriptions. No new data collection is required for the core experiment. | 4 h |
| **Network design and implementation** | Implement a compact OCR baseline and the modified recognizer; prepare training and evaluation code. | 12 h |
| **Training and fine-tuning** | Train the baseline and proposed model, tune using validation data, compare results, and analyze test-set errors. | 16 h |
| **Optional structured extraction** | If time and GPU resources allow, organize OCR output into structured item fields and evaluate the extension. | +4 h |
| **Optional personal test set** | With contributors' permission, collect and redact 20-30 receipt photos and prepare a separate exploratory test set. | +5 h |
| **Application to present the results** | Build a Streamlit demo for receipt upload, OCR review, optional item extraction, item assignment, and cost splitting. | 7 h |
| **Final report** | Describe related work, the dataset, method, evaluation, results, limitations, and compute use. | 8 h |
| **Presentation** | Prepare slides, select examples, explain the results and limitations, and rehearse. | 4 h |
| **Core project total** | Excludes the optional extensions and personal test set. | **55 h** |

## Scientific references

1. Kasem, M. S., Mahmoud, M., Senussi, M. F., Abdalla, M., & Kang, H. S. (2026). **KORIE: A Multi-Task Benchmark for Detection, OCR, and Information Extraction on Korean Retail Receipts.** *Mathematics, 14*(1), 187. [Paper](https://www.mdpi.com/2227-7390/14/1/187). The primary dataset, annotations, splits, and reported baselines.
2. Shi, B., Bai, X., & Yao, C. (2016). **An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition.** *IEEE Transactions on Pattern Analysis and Machine Intelligence, 39*(11), 2298-2304. [Paper](https://arxiv.org/abs/1507.05717). A foundational CNN-RNN-CTC approach for image text recognition that informs the OCR baseline.
3. Wang, Y., Zhan, L., Huang, Z., Luo, T., Jiang, Y., Dong, W., Gan, L., & Chen, J. (2026). **From Recognition to Reasoning: Benchmarking and Enhancing MLLMs on Real-World Receipt Document Understanding.** In *Proceedings of ACL 2026*. [Paper](https://aclanthology.org/2026.acl-long.2135/). ReceiptBench and metric-aware GRPO, which may inform an optional structured-extraction extension.
4. Abdallah, A., et al. (2024). **CORU: Comprehensive Post-OCR Parsing and Receipt Understanding Dataset.** [Paper](https://arxiv.org/abs/2406.04493). A receipt dataset considered as a fallback if its image and item-label links support the task.
