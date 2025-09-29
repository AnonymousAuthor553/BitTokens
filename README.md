# BitTokens: Efficient numeracy in language models through single-token number embeddings

![Figure 1](/images/fig1.png)
LLMs perform poorly on arithmetic tasks, requiring excessive reasoning tokens to achieve good performance. Our BitTokens tokenization strategy allows language models to solve arithmetic tasks both effectively and efficiently.

## BitTokens
The implementation of BitTokens can be found in the [float64.embedding.py](networks/number_embedding_modules/float64_embedding.py) file.


## Setup
> ℹ️ Info:
> We recommend using the fast package manager uv for dependency management, but you may use any other package manager. We provide an additional `requirements.txt` file for this. Replace `uv run` with `python` in the commands.

Download and install the fast package manager [UV](https://docs.astral.sh/uv/#highlights). 
```sh
# Download and install uv with python version >=3.13
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Sync uv environment
```sh
uv sync
```
Install remaining dependencies:
```sh
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install git+https://github.com/KellerJordan/Muon
```
> ℹ️ Info:
> 
> [FlashAttention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) sometimes causes trouble when installing. If you run into an error, refer to the official install guide.


Create an `.env` file and define the following variables:
```sh
PROJECT_PATH=... # Absolute path to the 'BitTokens/' folder
DATA_PATH=...    # Absolute path to data folder

# [Optional] If you want to use the eval_scripts
OPENROUTER_API_KEY=...
```

For convenience, load the `.env` file to execute the next commands.
```sh
source .env
```

### Get the datasets
1. Generate the number problems for each task for each phase:
    ```sh
    # Decimal version (used for all base-10 baselines and for testing)
    uv run $PROJECT_PATH/data_generation/data_generation_v2.py --save_dir $DATA_PATH
    # Binary version (used for BitToken training)
    uv run $PROJECT_PATH/data_generation/data_generation_v2.py --save_dir $DATA_PATH --significant_digits_distribution binary_uniform
    ```
2. Download the fineweb text data
    Download the fineweb_10BT subset from https://huggingface.co/datasets/HuggingFaceFW/fineweb and save it under `$DATA_PATH/`
3. Decode fineweb to `.txt`
    ```sh
    uv run $PROJECT_PATH/data_generation/decode_fineweb.py --folder_dir $DATA_PATH/sample/10BT/ --save_path $DATA_PATH/
    ```


## Running experiments
To recreate a BitToken model in a multiTask setting similar to the manuscript, run:
```sh
uv run $PROJECT_PATH/train.py --load_config_from $PROJECT_PATH/configs/config_fe_multiTask.py --tqdm --verbose --deterministic --seed 999
```
This has been tested on a `Nvidia DGX A100 80GB` GPU.

The results will be stored in the folder `$PROJECT_PATH/trained`.