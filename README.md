# ML-Project-Pipeline-With-Hydra
Pipeline for an ML project with Hydra

## Virtual Environment Creation

### Creating env from the exported env
```
cd dependencies/
conda env create -f environment.yml
```

### Creating env from a simple base
1. Create the environment from the environment.yml file:
```
conda env create -f env.yml
```
2. Activate the new environment: conda activate myenv
```
conda activate ml-venv
```
3. Exporting the env 
```
conda env export > environment.yml
```
4. Deactivating the env
```
conda deactivate
```
5. Removing the environment
```
conda remove --name ml-venv --all
```
To verify that the environment was removed, run:
```
conda info --envs
```

## Run the program
Download the desired dataset from the drive and put it inside the folder with dataset name. 
Change corresponding variable in cof/config.yaml file, if needed. 
Run the main file with:
```
conda activate ml-venv
cd ../src/
nohup python main.py model='llm-agent'> dummy_output.log 2>&1 &
```