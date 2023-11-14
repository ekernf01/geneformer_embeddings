import geneformer_embeddings.geneformer_embeddings as geneformer_embeddings
import geneformer_embeddings.geneformer_hyperparameter_optimization as geneformer_hyperparameter_optimization
import load_perturbations
load_perturbations.set_data_path("perturbation_data/perturbations")
adata = load_perturbations.load_perturbation("nakatake")
file_with_tokens = geneformer_embeddings.tokenize(adata)
some_good_fucken_hyperparameters = geneformer_hyperparameter_optimization.optimize_hyperparameters(file_with_tokens, n_cpu = 15)
model_save_path = geneformer_hyperparameter_optimization.finetune_classify(
    file_with_tokens, 
    column_with_labels = "louvain",
    max_input_size = 2 ** 11,  # 2048
    max_lr                = some_good_fucken_hyperparameters[2]["learning_rate"],
    freeze_layers = 0,
    geneformer_batch_size = some_good_fucken_hyperparameters[2]["per_device_train_batch_size"],
    lr_schedule_fn        = some_good_fucken_hyperparameters[2]["lr_scheduler_type"],
    warmup_steps          = some_good_fucken_hyperparameters[2]["warmup_steps"],
    epochs                = some_good_fucken_hyperparameters[2]["num_train_epochs"],
    optimizer = "adamw",
    GPU_NUMBER = [], 
    seed                  = 42, 
    weight_decay          = some_good_fucken_hyperparameters[2]["weight_decay"],
)
'geneformer_finetuned/231114_geneformer_CellClassifier_L2048_B12_LR0.0002627891502464483_LScosine_WU407.19691568088837_E1_Oadamw_F0/'