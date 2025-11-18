import os
from os.path import isfile
from pathlib import Path
import click
import pandas as pd
import warnings

# Suppress sklearn warnings about insufficient samples for cross-validation
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
warnings.filterwarnings("ignore", message="The least populated class in y has only")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from yaml import safe_load
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

from gene_benchmark.deserialization import load_class
from gene_benchmark.tasks import EntitiesTask

# ==============================================================================
# Helper Functions for Gene Overlap Calculation
# ==============================================================================

def load_all_embeddings(embedding_dir: str) -> dict:
    """Loads all processed .csv embeddings from a directory."""
    all_gene_embeddings = {}
    if not os.path.exists(embedding_dir):
        print(f"WARNING: Embedding directory not found: '{embedding_dir}'.")
        return all_gene_embeddings

    print(f"Loading embeddings from '{embedding_dir}' for overlap calculation...")
    for filename in sorted(os.listdir(embedding_dir)):
        if filename.endswith('.csv'):
            file_path = os.path.join(embedding_dir, filename)
            try:
                df = pd.read_csv(file_path).set_index('symbol')
                embedding_name = filename.replace('embeddings_', '').replace('.csv', '')
                all_gene_embeddings[embedding_name] = df
            except Exception as e:
                print(f"  -> ERROR loading {filename}: {e}")
    return all_gene_embeddings


def calculate_overlap(embedding_dict: dict, keys_to_include: list) -> list:
    """Calculates the list of overlapping genes for a given list of embeddings."""
    valid_keys = [key for key in keys_to_include if key in embedding_dict]
    if not valid_keys:
        print("Warning: No valid embeddings found for the given keys.")
        return []

    first_key = valid_keys[0]
    common_genes = set(embedding_dict[first_key].index)

    for key in valid_keys[1:]:
        common_genes.intersection_update(set(embedding_dict[key].index))

    return sorted(list(common_genes))


def get_report(task, output_file_name, append_results):
    """
    Create a summary dataframe, if append_results it loads the output_file_name and append the summary from res.

    Args:
    ----
        task : A task object with the data to summarize
        output_file_name (str): the file to append to
        append_results (bool): whether or not to append

    Returns:
    -------
        pd.DataFrame: A summary data frame

    """
    this_run_df = pd.DataFrame.from_dict(task.summary(), orient="index")
    this_run_df = this_run_df.transpose()
    if append_results and isfile(output_file_name):
        old_report = pd.read_csv(output_file_name)
        this_run_df = pd.concat([old_report, this_run_df], axis=0)
    return this_run_df


def load_yaml_file(model_config):
    with open(model_config) as f:
        loaded_yaml = safe_load(f)
    return loaded_yaml


def expand_task_list(task_list):
    tsk_list = []
    for task in task_list:
        if ".yaml" in task:
            tsk_list = tsk_list + load_yaml_file(task)
        else:
            tsk_list.append(task)
    return tsk_list


def validate_task_with_reduced_genes(task_name, tasks_folder, encoder, description_builder,
                                   exclude_symbols, include_symbols, scoring, base_model, cv,
                                   sub_sample, model_name, post_processing, sub_task,
                                   multi_label_th, cat_label_th, verbose):
    """
    Validates a task using the reduced set of genes to check if it has enough samples.
    Returns (is_valid, task_object) tuple.
    """
    if verbose:
        print(f"Validating task '{task_name}' with reduced gene set...")

    # Instantiate the task to load data
    task = EntitiesTask(
        task_name,
        tasks_folder=tasks_folder,
        encoder=encoder,
        description_builder=description_builder,
        exclude_symbols=exclude_symbols,
        include_symbols=include_symbols,
        scoring=scoring,
        base_model=base_model,
        cv=cv,
        frac=float(sub_sample),
        model_name=model_name,
        encoding_post_processing=post_processing,
        sub_task=sub_task,
        multi_label_th=multi_label_th,
        cat_label_th=cat_label_th,
    )

    # Set the target labels explicitly
    try:
        task.y = task.task_definitions.outcomes
    except (AttributeError, TypeError) as e:
        if verbose:
            print(f"--> ⚠️  SKIPPING TASK: Could not resolve target labels. Task may not have been properly initialized. Error: {e}")
        return False, None

    # Check if there are enough samples to perform cross-validation
    n_samples = len(task.y)
    MIN_SAMPLES_FOR_CV = 5

    if verbose:
        print(f"Task '{task_name}' | Total samples available: {n_samples}")

        # Print class distribution for classification tasks for easier debugging
        if hasattr(scoring, '__iter__') and not isinstance(scoring, str):
            scoring_list = scoring
        else:
            scoring_list = [scoring]

        if any('roc_auc' in str(s) or 'accuracy' in str(s) or 'precision' in str(s) or 'recall' in str(s) or 'f1' in str(s) for s in scoring_list):
            if isinstance(task.y, pd.Series):  # For binary or single-column category
                print("Class distribution:\n", task.y.value_counts())
            elif isinstance(task.y, pd.DataFrame):  # For multi-label
                print("Label distribution (positive samples per class):\n", task.y.sum())

    if n_samples < MIN_SAMPLES_FOR_CV:
        if verbose:
            print(f"--> ⚠️  SKIPPING TASK: Insufficient samples ({n_samples}) for {MIN_SAMPLES_FOR_CV}-fold cross-validation.")
        return False, None

    return True, task


def run_single_task(args):
    """
    Worker function to run a single task. This function needs to be defined at module level
    for multiprocessing to work properly.

    Args:
        args: tuple containing all the parameters needed to run the task

    Returns:
        tuple: (success, result_dataframe, error_message)
    """
    (task_name, sub_task, model_config_path, tasks_folder, exclude_symbols, include_symbols,
     scoring_type, sub_sample, model_name, post_processing, multi_label_th, cat_label_th) = args

    try:
        # Load model config
        with open(model_config_path) as f:
            model_dict = safe_load(f)

        description_builder = load_class(**model_dict["descriptor"]) if "descriptor" in model_dict else None

        # Define the number of CV splits to check against
        MIN_SAMPLES_FOR_CV = 5

        if scoring_type == "category":
            scoring = ("roc_auc_ovr_weighted", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted")
            cv = MIN_SAMPLES_FOR_CV
            base_model = LogisticRegression(max_iter=5000, n_jobs=1)  # Use 1 job per worker
        elif scoring_type == "regression":
            scoring = ("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error")
            cv = KFold(n_splits=MIN_SAMPLES_FOR_CV, shuffle=True)
            base_model = LinearRegression(n_jobs=1)  # Use 1 job per worker
        elif scoring_type == "multi":
            scoring = {
                "roc_auc_weighted": make_scorer(roc_auc_score, average="weighted"),
                "hamming_loss": make_scorer(hamming_loss), "accuracy": make_scorer(accuracy_score),
                "precision_weighted": make_scorer(precision_score, average="weighted"),
                "recall_weighted": make_scorer(recall_score, average="weighted"),
                "f1_weighted": make_scorer(f1_score, average="weighted"),
            }
            cv = KFold(n_splits=MIN_SAMPLES_FOR_CV, shuffle=True)
            base_model = MultiOutputClassifier(LogisticRegression(max_iter=5000, n_jobs=1))  # Use 1 job per worker
        else: # binary
            scoring = ("roc_auc", "accuracy", "precision", "recall", "f1")
            cv = MIN_SAMPLES_FOR_CV
            base_model = LogisticRegression(max_iter=5000, n_jobs=1)  # Use 1 job per worker

        if "base_model" in model_dict:
            base_model = load_class(**model_dict["base_model"])
        encoder = load_class(**model_dict["encoder"])

        # Create and run the task
        task = EntitiesTask(
            task_name,
            tasks_folder=tasks_folder,
            encoder=encoder,
            description_builder=description_builder,
            exclude_symbols=exclude_symbols,
            include_symbols=include_symbols,
            scoring=scoring,
            base_model=base_model,
            cv=cv,
            frac=float(sub_sample),
            model_name=model_name,
            encoding_post_processing=post_processing,
            sub_task=sub_task,
            multi_label_th=multi_label_th,
            cat_label_th=cat_label_th,
        )

        # Run the task
        _ = task.run()

        # Create the summary dataframe
        this_run_df = pd.DataFrame.from_dict(task.summary(), orient="index").transpose()
        this_run_df["model_config"] = os.path.basename(model_config_path)

        return True, this_run_df, None

    except Exception as e:
        error_msg = f"Error running task {task_name} with model {os.path.basename(model_config_path)}: {str(e)}"
        return False, None, error_msg


@click.command()
@click.option(
    "--tasks-folder",
    "-tf",
    type=click.STRING,
    help="The folder where tasks are stored. Defaults to `GENE_BENCHMARK_TASKS_FOLDER`",
    default=None,
)
@click.option(
    "--task-names",
    "-t",
    type=click.STRING,
    help="The path to the task yamls, or the task name",
    default=["long vs short range TF"],
    multiple=True,
)
@click.option(
    "--model-config-files",
    "-m",
    type=click.STRING,
    help="path to model config files",
    default=[str(Path(__file__).parent / "models" / "ncbi_multi_class.yaml")],
    multiple=True,
)
@click.option(
    "--model-config-dir",
    "-md",
    type=click.STRING,
    help="Directory with model config files",
    default=None
)
@click.option(
    "--excluded-symbols-file",
    "-e",
    type=click.STRING,
    help="A path to a yaml file containing symbols to be excluded",
    default=None,
)
@click.option(
    "--include-symbols-file",
    "-i",
    type=click.STRING,
    help="A path to a yaml file containing symbols to be included. This is ignored if --embedding-subset is used.",
    default=None,
)
@click.option(
    "--embedding-subset",
    type=click.Choice(['all', 'no_perturb_seq', 'no_perturbert'], case_sensitive=False),
    help="Exclude certain models from gene overlap AND from the evaluation run. "
         "'no_perturb_seq' excludes Perturb-seq models. "
         "'no_perturbert' excludes PerturBERT models (identified by 'perturbert_' prefix).",
    default='all',
)
@click.option(
    "--embedding-dir",
    type=click.STRING,
    help="Directory with processed embedding CSV files for overlap calculation.",
    default='data/processed_gene_embeddings/',
)
@click.option(
    "--output-file-name",
    type=click.STRING,
    help="The output file name.",
    default="task_report.csv",
)
@click.option(
    "--append-results",
    type=click.BOOL,
    help="Append results to the files",
    default=True,
)
@click.option(
    "--verbose",
    type=click.BOOL,
    help="print progress",
    default=True,
)
@click.option(
    "--sub-sample",
    type=click.FLOAT,
    help="sub sample the task",
    default=1,
)
@click.option(
    "--scoring_type",
    "-s",
    type=click.STRING,
    help="use different scoring",
    default="binary",
)
@click.option(
    "--multi-label-th",
    "-th",
    type=click.FLOAT,
    help="threshold of imbalance of labels in multi class tasks",
    default=0.0,
)
@click.option(
    "--cat-label-th",
    "-cth",
    type=click.FLOAT,
    help="threshold of imbalance of labels in category tasks",
    default=0.0,
)
@click.option(
    "--n-workers",
    type=click.INT,
    help="Number of parallel workers to use. Defaults to number of CPU cores.",
    default=None,
)
def main(
    tasks_folder,
    task_names,
    model_config_files,
    model_config_dir,
    excluded_symbols_file,
    include_symbols_file,
    embedding_subset,
    embedding_dir,
    output_file_name,
    append_results,
    verbose,
    sub_sample,
    scoring_type,
    multi_label_th,
    cat_label_th,
    n_workers,
):
    if tasks_folder is None:
        tasks_folder = Path(os.environ["GENE_BENCHMARK_TASKS_FOLDER"])
        assert tasks_folder.exists()

    if excluded_symbols_file:
        with open(excluded_symbols_file) as f:
            exclude_symbols = safe_load(f)
    else:
        exclude_symbols = []

    # --- Gene Overlap & Model Exclusion Logic ---
    # This section handles two important aspects:
    # 1. Gene overlap calculation: determines which genes are available across all included embeddings
    # 2. Model exclusion: ensures that excluded models are not used in evaluation
    # 
    # When using --embedding-subset, models are excluded from BOTH:
    # - Gene overlap calculation (affects which genes are available)
    # - Model evaluation (affects which models are run)
    include_symbols = None
    models_to_exclude = []
    
    # Define the list of YAML files to keep for no_perturbert option
    YAML_FILES_TO_KEEP = [
        'gene2vec.yaml',
        'GenePT_ada.yaml',
        'nadig_hepg2.yaml',
        'nadig_jurkat.yaml',
        'replogle_k562_essential_unfiltered.yaml',
        'replogle_k562_gw.yaml',
        'replogle_rpe1_essential_unfiltered.yaml',
        'scGPT_full_library.yaml'
    ]

    if embedding_subset != 'all':
        print(f"--- Calculating gene overlap for subset: '{embedding_subset}' ---")
        all_embeddings = load_all_embeddings(embedding_dir)
        if not all_embeddings:
            print("Could not load any embeddings for overlap calculation. Aborting.")
            return

        all_keys = list(all_embeddings.keys())
        keys_to_use = []

        # Handle different embedding subset options
        if embedding_subset == 'no_perturb_seq':
            PERTURB_SEQ_ALL = [
                'replogle_rpe1_essential_unfiltered', 'replogle_k562_essential_unfiltered',
                'replogle_k562_gw', 'nadig_hepg2', 'nadig_jurkat'
            ]
            models_to_exclude.extend(PERTURB_SEQ_ALL)
            keys_to_use = [k for k in all_keys if k not in models_to_exclude]
            print(f"Excluding {len(models_to_exclude)} Perturb-seq models from overlap AND evaluation.")

        elif embedding_subset == 'no_perturbert':
            # For the 'no_perturbert' option, we keep only the specified YAML files
            # These are all the non-perturbert embeddings
            MODELS_TO_KEEP = [
                'gene2vec',
                'GenePT_ada',
                'nadig_hepg2',
                'nadig_jurkat',
                'replogle_k562_essential_unfiltered',
                'replogle_k562_gw',
                'replogle_rpe1_essential_unfiltered',
                'scGPT_full_library'
            ]
            
            # Keys to use for the run are the models from our keep list that are actually present.
            keys_to_use = [key for key in all_keys if key in MODELS_TO_KEEP]
            
            # Models to exclude are any detected models that are not in our explicit keep list.
            models_to_exclude = [key for key in all_keys if key not in MODELS_TO_KEEP]

            print("\n--- Evaluation for 'no_perturbert' option ---")
            print(f"The evaluation is being run on the following {len(keys_to_use)} specified models:")
            for model in sorted(keys_to_use):
                print(f"  -> {model}")

            if verbose and models_to_exclude:
                print(f"\nExcluding {len(models_to_exclude)} other models found in the directory.")

        else:
            # For any other subset option (future extensions), use all available embeddings
            keys_to_use = all_keys

        # Calculate gene overlap using only the included embeddings
        include_symbols = calculate_overlap(all_embeddings, keys_to_use)
        print(f"--> Found {len(include_symbols)} overlapping genes to be used as the inclusion list.")
        if not include_symbols:
            print("Warning: Overlap resulted in 0 genes. The task will likely fail.")

    elif include_symbols_file:
        # If no embedding subset is specified, use the provided inclusion file
        with open(include_symbols_file) as f:
            include_symbols = safe_load(f)

    task_names = expand_task_list(task_names)

    # --- Filter Model Configs based on Exclusion List ---
    # This filtering ensures that when using --embedding-subset, models that use excluded embeddings
    # are not evaluated. The filtering works by excluding model configs whose names match
    # the excluded embedding names.
    all_model_configs = []
    

    
    if model_config_dir:
        print(f"\n--- Loading model configs from '{model_config_dir}' ---")
        for file_name in sorted(os.listdir(model_config_dir)):
            if file_name.endswith(".yaml"):
                # For no_perturbert option, only keep the specified YAML files
                if embedding_subset == 'no_perturbert':
                    if file_name not in YAML_FILES_TO_KEEP:
                        if verbose:
                            print(f"  -> 🚫 SKIPPING model config: {file_name} (excluded by no_perturbert)")
                        continue
                
                # Skip if excluded by embedding subset
                model_name_from_file = file_name.replace('.yaml', '')
                if model_name_from_file in models_to_exclude:
                    if verbose:
                        print(f"  -> 🚫 SKIPPING model config: {file_name} (excluded by --embedding-subset)")
                    continue
                all_model_configs.append(os.path.join(model_config_dir, file_name))
        print(f"Found {len(all_model_configs)} model configs to run.")
    elif model_config_files:
        # Also filter the files provided via -m
        temp_configs = []
        for config_path in model_config_files:
            file_name = os.path.basename(config_path)
            # For no_perturbert option, only keep the specified YAML files
            if embedding_subset == 'no_perturbert':
                if file_name not in YAML_FILES_TO_KEEP:
                    if verbose:
                        print(f"  -> 🚫 SKIPPING model config: {file_name} (excluded by no_perturbert)")
                    continue
            
            # Skip if excluded by embedding subset
            model_name_from_file = file_name.replace('.yaml', '')
            if model_name_from_file in models_to_exclude:
                if verbose:
                    print(f"  -> 🚫 SKIPPING model config: {file_name} (excluded by --embedding-subset)")
                continue
            temp_configs.append(config_path)
        all_model_configs = temp_configs
    
    # Summary of filtering results
    if models_to_exclude and verbose:
        print(f"\n📋 Filtering Summary:")
        print(f"   -> Total embedding models excluded: {len(models_to_exclude)}")
        print(f"   -> Models to evaluate: {len(all_model_configs)}")
        if embedding_subset == 'no_perturbert':
            print(f"   -> ✅ No PerturBERT models will be evaluated")
            print(f"   -> Only keeping specified YAML files: {', '.join(YAML_FILES_TO_KEEP)}")

    if not all_model_configs:
        print("\nERROR: No model configurations left to run after filtering. Aborting.")
        return

    # --- Pre-validate all tasks with reduced gene set ---
    print("="*80)
    print("Pre-validating tasks with reduced gene set...")
    print("="*80)

    valid_tasks = []
    for task_name in task_names:
        with open(all_model_configs[0]) as f:
            model_dict = safe_load(f)

        description_builder = load_class(**model_dict["descriptor"]) if "descriptor" in model_dict else None
        MIN_SAMPLES_FOR_CV = 5

        if scoring_type == "category":
            scoring = ("roc_auc_ovr_weighted", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted")
            cv = MIN_SAMPLES_FOR_CV
            base_model = LogisticRegression(max_iter=5000, n_jobs=-1)
        elif scoring_type == "regression":
            scoring = ("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error")
            cv = KFold(n_splits=MIN_SAMPLES_FOR_CV, shuffle=True)
            base_model = LinearRegression(n_jobs=-1)
        elif scoring_type == "multi":
            scoring = {
                "roc_auc_weighted": make_scorer(roc_auc_score, average="weighted"),
                "hamming_loss": make_scorer(hamming_loss), "accuracy": make_scorer(accuracy_score),
                "precision_weighted": make_scorer(precision_score, average="weighted"),
                "recall_weighted": make_scorer(recall_score, average="weighted"),
                "f1_weighted": make_scorer(f1_score, average="weighted"),
            }
            cv = KFold(n_splits=MIN_SAMPLES_FOR_CV, shuffle=True)
            base_model = MultiOutputClassifier(LogisticRegression(max_iter=5000, n_jobs=-1))
        else: # binary
            scoring = ("roc_auc", "accuracy", "precision", "recall", "f1")
            cv = MIN_SAMPLES_FOR_CV
            base_model = LogisticRegression(max_iter=5000, n_jobs=-1)

        model_name = model_dict.get("model_name")
        post_processing = model_dict.get("post_processing", "average")
        if "base_model" in model_dict:
            base_model = load_class(**model_dict["base_model"])
        encoder = load_class(**model_dict["encoder"])

        sub_task = None
        if ";" in task_name:
            task_name, sub_task = task_name.split(";", 1)

        is_valid, task_obj = validate_task_with_reduced_genes(
            task_name, tasks_folder, encoder, description_builder, exclude_symbols,
            include_symbols, scoring, base_model, cv, sub_sample, model_name,
            post_processing, sub_task, multi_label_th, cat_label_th, verbose
        )

        if is_valid:
            valid_tasks.append((task_name, sub_task, task_obj))
            print(f"✅ Task '{task_name}' validated with {len(task_obj.y)} samples")
        else:
            print(f"❌ Task '{task_name}' failed validation - insufficient samples")

    print(f"\n--- Task Validation Complete ---")
    print(f"Total tasks: {len(task_names)}")
    print(f"Valid tasks: {len(valid_tasks)}")
    print(f"Failed tasks: {len(task_names) - len(valid_tasks)}")
    print("="*80)

    if not valid_tasks:
        print("No valid tasks found. Exiting.")
        return

    # --- Run tasks in parallel ---
    print("="*80)
    print("Running tasks in parallel...")
    print("="*80)

    if n_workers is None:
        n_workers = min(os.cpu_count(), 120)

    print(f"Using {n_workers} parallel workers")

    all_task_combinations = []
    for model_config in all_model_configs:
        with open(model_config) as f:
            model_dict = safe_load(f)
        model_name = model_dict.get("model_name")
        post_processing = model_dict.get("post_processing", "average")

        for task_name, sub_task, validated_task in valid_tasks:
            all_task_combinations.append((
                task_name, sub_task, model_config, tasks_folder, exclude_symbols,
                include_symbols, scoring_type, sub_sample, model_name, post_processing,
                multi_label_th, cat_label_th
            ))

    print(f"Total task combinations to run: {len(all_task_combinations)}")

    all_results_df_list = []
    failed_tasks = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {
            executor.submit(run_single_task, task_args): task_args
            for task_args in all_task_combinations
        }

        for future in as_completed(future_to_task):
            task_args = future_to_task[future]
            task_name, sub_task, model_config = task_args[0], task_args[1], task_args[2]

            try:
                success, result_df, error_msg = future.result()
                if success:
                    all_results_df_list.append(result_df)
                    print(f"✅ Completed: {os.path.basename(model_config)} | Task: {task_name} | Sub-task: {sub_task}")
                else:
                    failed_tasks.append((task_name, sub_task, model_config, error_msg))
                    print(f"❌ Failed: {os.path.basename(model_config)} | Task: {task_name} | Sub-task: {sub_task}")
                    if verbose:
                        print(f"   Error: {error_msg}")
            except Exception as e:
                failed_tasks.append((task_name, sub_task, model_config, str(e)))
                print(f"❌ Exception: {os.path.basename(model_config)} | Task: {task_name} | Sub-task: {sub_task}")
                if verbose:
                    print(f"   Exception: {str(e)}")

    print("="*80)
    print("PARALLEL EXECUTION COMPLETE")
    print(f"Successful tasks: {len(all_results_df_list)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    print("="*80)

    if failed_tasks and verbose:
        print("Failed task details:")
        for task_name, sub_task, model_config, error in failed_tasks:
            print(f"  - {os.path.basename(model_config)} | {task_name} | {sub_task}: {error}")

    if all_results_df_list:
        final_report_df = pd.concat(all_results_df_list, ignore_index=True)
        if not final_report_df.empty:
            if append_results and os.path.isfile(output_file_name):
                old_report_df = pd.read_csv(output_file_name)
                combined_df = pd.concat([old_report_df, final_report_df], ignore_index=True)
                combined_df.to_csv(output_file_name, index=False)
            else:
                final_report_df.to_csv(output_file_name, index=False)
            print("="*80)
            print(f"Final report saved to {output_file_name}")
        else:
            print("No results were generated to save.")
    else:
        print("No tasks were successfully run. No report generated.")


if __name__ == "__main__":
    main()