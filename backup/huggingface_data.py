import os
from csv import writer

def get_huggingface_dataset():
    try:
        import huggingface_hub
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Package `huggingface_hub` not found is needed "
            "for HuggingFaceDatasetSaver. Try 'pip install huggingface_hub'."
        )
    HF_TOKEN = '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'  # my HF token
    DATASET_NAME = 'crowdsourced-robotinder-demo'
    FLAGGING_DIR = 'flag/'
    path_to_dataset_repo = huggingface_hub.create_repo(
        repo_id=DATASET_NAME,
        token=HF_TOKEN,
        private=False,
        repo_type="dataset",
        exist_ok=True,
    )    
    dataset_dir = os.path.join(DATASET_NAME, FLAGGING_DIR)
    repo = huggingface_hub.Repository(
        local_dir=dataset_dir,
        clone_from=path_to_dataset_repo,
        use_auth_token=HF_TOKEN,
    )
    repo.git_pull(lfs=True)
    log_file = os.path.join(dataset_dir, "flag_data.csv")
    with open(log_file, 'a') as file:
        writer_object = writer(file)
        info = ['env_name', 'user_choice', 'left_video', 'right_video', 'time']  # for test
        writer_object.writerow(info)
        file.close()
    repo.push_to_hub(commit_message="Flagged sample")

get_huggingface_dataset()