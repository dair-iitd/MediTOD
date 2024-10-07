Set OPENAI_API_KEY, OPENAI_API_CHECKPOINT and OPENAI_MODEL environment variable properly.

To evalute NLU.
```
python nlu_metrics.py -data=../OsceTOD/data/dialogs.json -pred=<prediction_file> -model=<model>
```
<model> can be PPTOD, Flan-T5, BioGPT or LLM. Choose LLM for LLama3, OpenBioLLM, ChatGPT, GPT4 supervised and few-shot models.
<prediction_file> is the processed prediction file from the models.

To evalute POL.
```
python pol_metrics.py -data=../OsceTOD/data/dialogs.json -pred=<prediction_file> -model=<model>
```
<model> can be PPTOD, Flan-T5, BioGPT or LLM. Choose LLM for LLama3, OpenBioLLM, ChatGPT, GPT4 supervised and few-shot models.
<prediction_file> is the processed prediction file from the models.

To evalute NLG.
```
python nlg_metrics.py -data=../OsceTOD/data/dialogs.json -pred=<prediction_file> -model=<model>
```
<model> can be PPTOD, Flan-T5, BioGPT or LLM. Choose LLM for LLama3, OpenBioLLM, ChatGPT, GPT4 supervised and few-shot models.
<prediction_file> is the processed prediction file from the models.
