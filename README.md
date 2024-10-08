## A convenient GeneFormer interface for

This is a pip-installable wrapper for [GeneFormer](https://huggingface.co/ctheodoris/Geneformer) that streamlines certain tasks: 

- Fine-tuning for cell type classification, including hyperparameter optimization
- Extraction of post-perturbation cell embeddings 

### Installation

Install GeneFormer using the [official instructions](https://huggingface.co/ctheodoris/Geneformer#installation). Then do:

```
pip install git+https://github.com/ekernf01/geneformer_embeddings
```

### Usage

The input AnnData object must have columns:

- `"perturbation"` containing HGNC symbols for perturbed genes. These should be comma-separated for multiple genes at once and empty string for controls.
- `"perturbation_type"` containing the string `"overexpression"` or `"knockout"` for each observation
- Optionally, cluster labels for fine-tuning. Below, we use `"louvain"`. 

If you need a small example, try the Nakatake data from our [collection](https://github.com/ekernf01/perturbation_data/).

```python
import pereggrn_perturbations
pereggrn_perturbations.set_data_path("path/to/perturbation_data/perturbations")
adata = pereggrn_perturbations.load_perturbation("nakatake")
```

#### No fine-tuning (not recommended)

```python
from geneformer_embeddings import geneformer_embeddings
emb = geneformer_embeddings.get_geneformer_perturbed_cell_embeddings(adata, perturb_type = "overexpress")
```

#### With fine-tuning

```python
from geneformer_embeddings imoprt geneformer_embeddings, geneformer_hyperparameter_optimization 
file_with_tokens = geneformer_embeddings.tokenize(adata, "louvain")
optimal_hyperparameters = geneformer_hyperparameter_optimization.optimize_hyperparameters(file_with_tokens, n_cpu = 15)
model_save_path = geneformer_hyperparameter_optimization.finetune_classify(
    file_with_tokens, 
    column_with_labels = "louvain",
    max_input_size = 2 ** 11,  # 2048
    max_lr                = optimal_hyperparameters[2]["learning_rate"],
    freeze_layers = 0,
    geneformer_batch_size = optimal_hyperparameters[2]["per_device_train_batch_size"],
    lr_schedule_fn        = optimal_hyperparameters[2]["lr_scheduler_type"],
    warmup_steps          = optimal_hyperparameters[2]["warmup_steps"],
    epochs                = optimal_hyperparameters[2]["num_train_epochs"],
    optimizer = "adamw",
    GPU_NUMBER = [], 
    seed                  = 42, 
    weight_decay          = optimal_hyperparameters[2]["weight_decay"],
)
emb = geneformer_embeddings.get_geneformer_perturbed_cell_embeddings(adata, perturb_type = "overexpress")
```