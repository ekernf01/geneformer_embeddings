## Extract cell embeddings from GeneFormer

### Installation

[Install GeneFormer](https://huggingface.co/ctheodoris/Geneformer#installation). Then do:

```
pip install git+https://github.com/ekernf01/geneformer_embeddings
```

### Usage

Input AnnData object must have columns:

- `perturbation` containing HGNC symbols for perturbed genes -- comma-separated for multiple genes at once and empty string for controls.
- `perturbation_type` containing the string `overexpression` or `knockout` for each observation

```python
from geneformer_embeddings import geneformer_embeddings
emb = geneformer_embeddings.get_geneformer_perturbed_cell_embeddings(adata, perturb_type = "overexpress")
```