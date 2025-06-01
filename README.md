# cs221-childsafety

**data.py**: Contains initial PKU-SafeRLHF dataset harm category distribution analysis, manual train/dev set split, and baseline model implementation

**error_analysis.py**: Contains implementation of error analysis on finetuned BERT classifier, overall and per-category metrics

**evaluate_finetuned.py**: Evaluates finetuned BERT classifier performance

**evalaute_minorbench.py**: Zero-shot transfer to MinorBench

**mapping.py**: Generated mapping of PKU -> MinorBench labels using Gemini

**train_pku.py**: Training/finetuning BERT to classify harm

**interpret.py**: Computing SHAP values and per-category threshold tuning

**analyze_interpret.py**: Analyzing SHAP values
