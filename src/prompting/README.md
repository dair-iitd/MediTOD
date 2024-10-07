To generate prompts run the following commands.
```
python prompt_gen.py --raw_file=../../data/dialogs.json --train_file=../../data/train_ids.json --test_file=../../data/test_ids.json --tar_file=nlu_prompts.json --task=nlu
python prompt_gen.py --raw_file=../../data/dialogs.json --train_file=../../data/train_ids.json --test_file=../../data/test_ids.json --tar_file=pol_prompts.json --task=pol
python prompt_gen.py --raw_file=../../data/dialogs.json --train_file=../../data/train_ids.json --test_file=../../data/test_ids.json --tar_file=nlg_prompts.json --task=nlg
```

To run inference use following command.
```
python run_prompt.py --prompt_file=<prompt_file> --model=<model> --destpath=<output_dir>
```
The command runs the inference on prompts using the provided model. Make sure OPENAI_API_KEY and OPENAI_API_BASE are set properly. Model can be your OpenAI endpoint or it can be Llama3 or OpenBioLLM wildcards.

To post process model outputs run following command.
```
python post.py --path=<output_dir> --tar_file=<target_file_path> --model=<model> --data=../../data/dialogs.json --task=<task>
```
The command compiles the results from <output_dir> into <target_file_path> JSON file. <task> can be NLU, POL or NLG.