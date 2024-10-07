This code trains FlanT5 and BioGPT models on MediTOD dataset.
Training data is located under *../../data/pptod/*
Training command

```
python train.py -cfg=<config_file_path>
```
**configs** folder contains config files for NLU, POL and NLG tasks.
The training command will save the checkpoints under **runs** directory.

To run inference use following command.

```
python infer.py -cfg=<config_file_path> -model_path=<run_path> -chkpt=<checkpoint_number> -rs=<results_file_path>
```
The command runs the inference on test set using the provided checkpoint. Generated results are stored in results_file_path file.
