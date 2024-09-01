import os
from collections import Counter
import datetime
import pickle
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
# initiate runtime environment for raytune
import pyarrow # must occur prior to ray import
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import random
import geneformer_embeddings.geneformer_embeddings as geneformer_embeddings

def optimize_hyperparameters(
        file_with_tokens,
        column_with_labels, 
        num_proc = 1, 
        freeze_layers = 2,
        geneformer_batch_size = 12,
        epochs = 10,
        logging_steps = None,
        seed = 42,
        output_dir = "geneformer_hyperparams",
        n_cpu = 8,
    ):
    ray.shutdown() 
    conda_path = os.popen("which conda").read().strip()
    runtime_env = {"conda": "ggrn",
               "env_vars": {"LD_LIBRARY_PATH": conda_path}}
    ray.init(runtime_env=runtime_env)
    train_dataset=load_from_disk(file_with_tokens)
    target_names = list(set(train_dataset[column_with_labels]))
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    train_dataset = train_dataset.rename_column(column_with_labels,"label")
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    trainset_v4 = train_dataset.map(classes_to_ids, num_proc=num_proc)
    # separate into train, validation, test sets
    random.seed(seed)
    indiv_set = set(trainset_v4["individual"])
    train_indiv = random.sample(indiv_set,round(0.7*len(indiv_set)))
    eval_indiv = [indiv for indiv in indiv_set if indiv not in train_indiv]
    valid_indiv = random.sample(eval_indiv,round(0.5*len(eval_indiv)))
    test_indiv = [indiv for indiv in eval_indiv if indiv not in valid_indiv]
    def if_train(example):
        return example["individual"] in train_indiv
    classifier_trainset = trainset_v4.filter(if_train,num_proc=num_proc).shuffle(seed=42)
    def if_valid(example):
        return example["individual"] in valid_indiv
    classifier_validset = trainset_v4.filter(if_valid,num_proc=num_proc).shuffle(seed=42)

    # Overwrite previously saved model?
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")
    os.makedirs(output_dir, exist_ok = True)

    def model_init():
        model = BertForSequenceClassification.from_pretrained("../Geneformer",
                                                            num_labels=len(target_names),
                                                            output_attentions = False,
                                                            output_hidden_states = False)
        if freeze_layers is not None:
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        model = model.to("cpu")
        return model

    # define metrics
    # note: macro f1 score recommended for imbalanced multiclass classifiers
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds)
        return {
        'accuracy': acc,
        }

    # set training arguments
    if logging_steps is None:
        logging_steps = round(len(classifier_trainset)/geneformer_batch_size/10)
    training_args = {
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "eval_steps": logging_steps,
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": True,
        "skip_memory_metrics": True, # memory tracker causes errors in raytune
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": False,
        "output_dir": output_dir,
    }

    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=classifier_trainset,
        eval_dataset=classifier_validset,
        compute_metrics=compute_metrics,
    )

    # specify raytune hyperparameter search space
    ray_config = {
        "num_train_epochs": tune.choice([epochs]),
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "weight_decay": tune.uniform(0.0, 0.3),
        "lr_scheduler_type": tune.choice(["linear","cosine","polynomial"]),
        "warmup_steps": tune.uniform(100, 2000),
        "seed": tune.uniform(0,100),
        "per_device_train_batch_size": tune.choice([geneformer_batch_size])
    }

    hyperopt_search = HyperOptSearch(metric="eval_accuracy", mode="max")

    # optimize hyperparameters
    hyperparameters = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        resources_per_trial={"cpu":n_cpu,"gpu":0},
        hp_space=lambda _: ray_config,
        search_alg=hyperopt_search,
        n_trials=1, # number of trials
        progress_reporter=tune.CLIReporter(max_report_frequency=600,
                                            sort_by_metric=True,
                                            max_progress_rows=100,
                                            mode="max",
                                            metric="eval_accuracy",
                                            metric_columns=["loss", "eval_loss", "eval_accuracy"])
        )
    ray.shutdown() 
    return hyperparameters

def finetune_classify(
        file_with_tokens,
        column_with_labels,
        base_model,
        max_input_size = 2 ** 11,  # 2048
        max_lr = 5e-5,
        freeze_layers = 2,
        geneformer_batch_size = 12,
        lr_schedule_fn = "linear",
        warmup_steps = 500,
        epochs = 10,
        optimizer = "adamw",
        GPU_NUMBER = [], 
        seed = 42,
        weight_decay = 0.001,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
    os.environ["NCCL_DEBUG"] = "INFO"
    train_dataset=load_from_disk(file_with_tokens)
    train_dataset = train_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.rename_column(column_with_labels, "label")
    target_names = list(Counter(train_dataset["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = train_dataset.map(classes_to_ids, num_proc=16)
    labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*0.8))])
    labeled_eval_split  = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*0.8),len(labeled_trainset))])
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy and macro f1 using sklearn's function
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        return {
        'accuracy': acc,
        'macro_f1': macro_f1
        }

    # set logging steps
    logging_steps = round(len(labeled_train_split)/geneformer_batch_size/10)
    
    # reload pretrained model
    model = BertForSequenceClassification.from_pretrained(
        base_model, 
        num_labels=len(target_names),
        output_attentions = False,
        output_hidden_states = False
    ).to("cpu")

    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = f"geneformer_finetuned/{datestamp}_geneformer_CellClassifier_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
    
    # Remove any previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    try: 
        os.unlink(saved_model_test)
    except FileNotFoundError:
        pass

    # make output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=labeled_train_split,
        eval_dataset=labeled_eval_split,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier
    trainer.train()
    predictions = trainer.predict(labeled_eval_split)
    with open(f"{output_dir}predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)
    return output_dir