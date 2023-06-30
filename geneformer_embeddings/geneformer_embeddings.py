import os 
import pathlib
from geneformer import in_silico_perturber, InSilicoPerturber, TranscriptomeTokenizer
from datasets import Dataset
from transformers import BertForMaskedLM
import biomart
import pandas as pd
import anndata
import pickle
import numpy as np 
import shutil
import torch 
import math

def _get_ensembl_mappings():
    """Obtain dictionaries to map between HGNC symbol and ensembl ID"""                                   
    # Set up connection to server                                               
    server = biomart.BiomartServer('http://useast.ensembl.org/biomart')         
    mart = server.datasets['hsapiens_gene_ensembl']                            
                                                                                                                                                    
    # Get the mapping between the attributes                                    
    response = mart.search({'attributes': ['hgnc_symbol', 'ensembl_gene_id']})                          
    data = response.raw.data.decode('ascii')                                    
                                                                                
    ensembl_to_genesymbol = {}                                                  
    genesymbol_to_ensembl = {}
    for line in data.splitlines():                                              
        hgnc_symbol, ensembl_gene_id = line.split('\t')                                                 
        if hgnc_symbol != "" and ensembl_gene_id != "":
            genesymbol_to_ensembl[hgnc_symbol] = ensembl_gene_id                    
            ensembl_to_genesymbol[ensembl_gene_id] = hgnc_symbol                       
                                                                   
    return {"ensembl_to_genesymbol": ensembl_to_genesymbol,
            "genesymbol_to_ensembl": genesymbol_to_ensembl}


def _perturb_tokenized_representation(control_expression, 
                            perturb_type: str, 
                            perturbation_list_list):
    """Perturb the tokenized representation of a cell as typically done by 
    GeneFormer: put overexpressed genes first and deleted genes last."""
    
    # The original code appears not to support overexpressing genes unless they are  
    # detected in the starting cell state. in_silico_perturber.overexpress_index can only 
    # perturb a gene whose token is already present. Here we add the missing tokens at the 
    # end to better support reprogramming.
    cell_token_lists = [None for i in perturbation_list_list]
    for i,perturbation_list in enumerate(perturbation_list_list):
        undetected_tokens = set(perturbation_list).difference(set(control_expression[i]["input_ids"]))
        cell_token_lists[i] = control_expression[i]["input_ids"]
        cell_token_lists[i][0:len(undetected_tokens)] = undetected_tokens
    perturbation_dataset = Dataset.from_dict(
        {
            "input_ids": cell_token_lists, 
            "length": [len(c) for c in cell_token_lists], 
            "perturb_index": 
            [
                [
                    cell_token_lists[i].index(p)
                    for p in perturbation_list_list[i]
                ] 
                for i in range(len(cell_token_lists))
            ],
        }
    )
    if perturb_type.lower() in {"delete", "knockdown", "knockout"}:
        perturbation_dataset = perturbation_dataset.map(in_silico_perturber.delete_index, num_proc=15)
    elif perturb_type.lower() in {"overexpress", "overexpression"}:
        perturbation_dataset = perturbation_dataset.map(in_silico_perturber.overexpress_index, num_proc=15)
    else:
        raise ValueError(f"perturb_type must be 'delete', 'knockdown', 'knockout', or 'overexpress', or 'overexpression'; got {perturb_type}.")
    return perturbation_dataset

def get_geneformer_perturbed_cell_embeddings(
        adata_train: anndata.AnnData, 
        layer_to_quant: int = -1,
        assume_unrecognized_genes_are_controls: bool = False,
        apply_perturbation_explicitly: bool = True,
        gene_name_converter: dict = None,
    ):
    """Fit a model to predict perturbation-induced fold change by transforming GeneFormer 
    embeddings into quantitative expression values

    Args:

        adata_train (anndata.AnnData): expression data with raw counts in .raw.X, perturbations 
            in .obs["perturbation"], and perturbation types in .obs["perturbation_type"].
        layer_to_quant (int): What layer of the network to extract embeddings from.
        apply_perturbation_explicitly: If True, apply perturbations as in GeneFormer: alter the rank order to place overexpressed 
            genes first and deleted genes last. Otherwise, assume the input expression already reflects perturbation.
        assume_unrecognized_genes_are_controls: If True, treat unrecognized gene names as controls when reading perturbation info. 
            We recommend False, but True can be useful for e.g. controls labeled "GFP" or "ctrl" or "scramble".  
        gene_name_converter (dict): dict where keys are HGNC symbols and values are Ensembl gene ID's.

    Returns:

        numpy array with one column per feature and one row per observation in adata.train
    """

    with open(in_silico_perturber.TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)

    if gene_name_converter is None:
        gene_name_converter = _get_ensembl_mappings()["genesymbol_to_ensembl"]  

    # Prereqs: raw counts, ensembl genes, certain metadata, save to loom file, tokenize
    adata_train.var["ensembl_id"] = [gene_name_converter[g] if g in gene_name_converter else "" for g in adata_train.var_names]
    adata_train = adata_train[:, adata_train.var["ensembl_id"] != ""] 
    adata_train.obs["filter_pass"] = True
    adata_train.obs["cell_type"] = "unknown"
    try:
        adata_train.obs["n_counts"] = adata_train.raw.X.sum(axis = 1)
        adata_train.X = adata_train.raw[adata_train.obs_names, adata_train.var_names].X
    except:
        raise ValueError("GeneFormer requires that raw counts be available in .raw.X.")
    adata_train.obs.columns = [c.replace('/', '_') for c in adata_train.obs.columns] # Loom hates slashes in names
    # Delete prior loom data
    try:
        shutil.rmtree("geneformer_loom_data")
    except FileNotFoundError:
        pass
    os.makedirs("geneformer_loom_data", exist_ok=True)
    adata_train.obs_names = [str(s) for s in adata_train.obs_names] #loom hates Categorical, just like everyone else
    adata_train.var_names = [str(s) for s in adata_train.var_names]
    adata_train.write_loom("geneformer_loom_data/adata_train.loom")
    tk = TranscriptomeTokenizer({}, nproc=15)
    # Delete prior tokenized data
    try:
        shutil.rmtree("tokenized_data")
    except FileNotFoundError:
        pass
    tk.tokenize_data(pathlib.Path("geneformer_loom_data"), "tokenized_data", "demo")
    isp = InSilicoPerturber(model_type = "Pretrained")
    filtered_input_data = isp.load_and_filter(input_data_file = "tokenized_data/demo.dataset")

    # Obtain perturbed embeddings
    geneformer_model = BertForMaskedLM.from_pretrained("ctheodoris/Geneformer", output_hidden_states=True, output_attentions=False)
    embeddings = np.zeros((adata_train.n_obs, 256))
    tokens_to_perturb = []
    for int_i, i in enumerate(adata_train.obs_names):
        tokens_to_perturb.append([])
        # Handle nasty cases: multi-gene or bad gene name
        if apply_perturbation_explicitly:
            perts_comma_separated = str(adata_train.obs.loc[i, "perturbation"])
            for g in perts_comma_separated.split(","):
                try:
                    tokens_to_perturb[int_i].append(gene_token_dict[gene_name_converter[g]])
                except KeyError as e:
                    if not assume_unrecognized_genes_are_controls:
                        raise KeyError(f"Gene {g} either has no GeneFormer token or no Ensembl ID, so it cannot be perturbed. Original error: {repr(e)}")
    adata_train.obs["perturbation_type"] = adata_train.obs["perturbation_type"].astype(str)
    assert len(adata_train.obs["perturbation_type"].unique()) == 1, "Our GeneFormer interface cannot handle deletion and overexpression in the same dataset."
    assert len(filtered_input_data) != len(tokens_to_perturb), "Internal error: number of tokenized cells does not match number of perturbations."
    perturbation_batch = _perturb_tokenized_representation(
        control_expression = filtered_input_data, 
        perturb_type = adata_train.obs["perturbation_type"][0],
        perturbation_list_list = tokens_to_perturb
    )
    # We can't do the forward pass all in one batch due to very large memory requirements.
    perturbation_batch.set_format(type="torch")
    embeddings = np.zeros((adata_train.n_obs, 256))
    batches = np.array_split(range(adata_train.n_obs), math.ceil(adata_train.n_obs/100))
    print(f"Extracting cell embeddings from GeneFormer in {len(batches)} batches.")
    for batch in batches:
        print(".", end = "")
        with torch.no_grad():
            outputs = geneformer_model(
                input_ids = perturbation_batch[batch]["input_ids"].to("cpu")
            )
        embeddings[batch, :] = outputs.hidden_states[layer_to_quant].sum(axis=1).to_dense()
        del outputs
    # Clean up temporary files
    shutil.rmtree("geneformer_loom_data")
    shutil.rmtree("tokenized_data")
    print("Done extracting features.", flush = True)
    return embeddings