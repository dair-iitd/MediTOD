This code trains Llama3 and OpenBioLLM models on MediTOD dataset.
Training data is located under *../../data/pptod/*
Training command

```
python train.py -cfg=<config_file_path>
```
**configs** folder contains config files for NLU, POL and NLG tasks.
The training command will save the checkpoints under **runs** directory.

To run inference use following command.

```
python infer.py -cfg=<config_file_path> -model_path=<checkpoint_path> -rs=<results_file_path> -it=test
```
The command runs the inference on test set using the provided checkpoint. Generated results are stored in results_file_path file.

To post process model outputs run following command.
```
python post.py --path=<results_file_path> --tar_file=<proc_file_path> --data=../../data/pptod/fine-processed-test.json --task=<task>
```
The command compiles the results from <results_file_path> into <proc_file_path>. <task> can be NLU, POL or NLG.
