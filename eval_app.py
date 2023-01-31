import gradio as gr
import os
import random
import numpy as np
import pandas as pd
import gdown
import base64
from time import gmtime, strftime
from csv import writer
import json
import zipfile
from os import listdir
from os.path import isfile, join, isdir
# from datasets import load_dataset
# from hfserver import HuggingFaceDatasetSaver, HuggingFaceDatasetJSONSaver

# all 20 tasks
# ENVS = ['ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
# 'ShadowHandCatchOver2Underarm', 'ShadowHandBottleCap', 'ShadowHandLiftUnderarm', 'ShadowHandTwoCatchUnderarm',
# 'ShadowHandDoorOpenInward', 'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward', 'ShadowHandDoorCloseOutward',
# 'ShadowHandPushBlock', 'ShadowHandKettle', 
# 'ShadowHandScissors', 'ShadowHandPen', 'ShadowHandSwingCup', 'ShadowHandGraspAndPlace', 'ShadowHandSwitch']

# selected 17 tasks
ENVS = ['ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
'ShadowHandCatchOver2Underarm', 'ShadowHandBottleCap', 'ShadowHandLiftUnderarm', 'ShadowHandTwoCatchUnderarm',
'ShadowHandDoorOpenInward', 'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward',
'ShadowHandPushBlock',
'ShadowHandScissors', 'ShadowHandPen', 'ShadowHandGraspAndPlace', 'ShadowHandSwitch']

# unseen 4 envs
# ENVS = ['ShadowHandCatchAbreastPen', 'ShadowHandCatchUnderarmPen', 'ShadowHandTwoCatchAbreast', 'ShadowHandGraspAndPlaceEgg']


# download data from huggingface dataset
# dataset = load_dataset("quantumiracle-git/robotinder-data")
# os.remove('.git/hooks/pre-push')  # https://github.com/git-lfs/git-lfs/issues/853
LOAD_DATA_GOOGLE_DRIVE = False

# this will install git-lfs
# os.system('curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash')
# os.system('sudo apt-get install git-lfs')

if LOAD_DATA_GOOGLE_DRIVE:  # download data from google drive
    # url = 'https://drive.google.com/drive/folders/1JuNQS4R7axTezWj1x4KRAuRt_L26ApxA?usp=sharing'  # './processed/' folder in google drive
    # url = 'https://drive.google.com/drive/folders/1o8Q9eX-J7F326zv4g2MZWlzR46uVkUF2?usp=sharing'  # './processed_zip/' folder in google drive
    # url = 'https://drive.google.com/drive/folders/1ZWgpPiZwnWfwlwta8Tu-Jtu2HsS7HAEa?usp=share_link'  # './filter_processed_zip/' folder in google drive
    # url = 'https://drive.google.com/drive/folders/1ROkuX6rQpyK7vLqF5fL2mggKiMCdKSuY?usp=share_link'  # './split_processed_zip/' folder in google drive

    # output = './'
    # id = url.split('/')[-1]
    # os.system(f"gdown --id {id} -O {output} --folder --no-cookies --remaining-ok")
    # # VIDEO_PATH = 'processed_zip'
    # # VIDEO_PATH = 'filter_processed_zip'
    # VIDEO_PATH = 'split_processed_zip'

    # # unzip the zip files to the same location and delete zip files
    # path_to_zip_file = VIDEO_PATH
    # zip_files = [join(path_to_zip_file, f) for f in listdir(path_to_zip_file)]
    # for f in zip_files:
    #     if f.endswith(".zip"):
    #         directory_to_extract_to = path_to_zip_file # extracted file itself contains a folder
    #         print(f'extract data {f} to {directory_to_extract_to}')
    #         with zipfile.ZipFile(f, 'r') as zip_ref:
    #             zip_ref.extractall(directory_to_extract_to)
    #         os.remove(f)

    ### multiple urls to handle the retrieve error
    # urls = [
    #     'https://drive.google.com/drive/folders/1BbQe4XtcsalsvwGVLW9jWCkr-ln5pvyf?usp=share_link',  # './filter_processed_zip/1' folder in google drive
    #     'https://drive.google.com/drive/folders/1saUTUuObPhMJFguc2J_O0K5woCJjYHci?usp=share_link',  # './filter_processed_zip/2' folder in google drive
    #     'https://drive.google.com/drive/folders/1Kh9_E28-RH8g8EP1V3DhGI7KRs9LB7YJ?usp=share_link',  # './filter_processed_zip/3' folder in google drive
    #     'https://drive.google.com/drive/folders/1oE75Dz6hxtaSpNhjD22PmQfgQ-PjnEc0?usp=share_link',  # './filter_processed_zip/4' folder in google drive
    #     'https://drive.google.com/drive/folders/1XSPEKFqNHpXdLho-bnkT6FZZXssW8JkC?usp=share_link',  # './filter_processed_zip/5' folder in google drive
    #     'https://drive.google.com/drive/folders/1XwjAHqR7kF1uSyZZIydQMoETfdvi0aPD?usp=share_link',
    #     'https://drive.google.com/drive/folders/1TceozOWhLsbqP-w-RkforjAVo1M2zsRP?usp=share_link',
    #     'https://drive.google.com/drive/folders/1zAP9eDSW5Eh_isACuZJadXcFaJNqEM9u?usp=share_link',
    #     'https://drive.google.com/drive/folders/1oK8fyF9A3Pv5JubvrQMjTE9n66vYlyZN?usp=share_link',
    #     'https://drive.google.com/drive/folders/1cezGNjlM0ONMM6C0N_PbZVCGsTyVSR0w?usp=share_link',
    # ]

    urls = [
        'https://drive.google.com/drive/folders/1SF5jQ7HakO3lFXBon57VP83-AwfnrM3F?usp=share_link',  # './split_processed_zip/1' folder in google drive
        'https://drive.google.com/drive/folders/13WuS6ow6sm7ws7A5xzCEhR-2XX_YiIu5?usp=share_link',  # './split_processed_zip/2' folder in google drive
        'https://drive.google.com/drive/folders/1GWLffJDOyLkubF2C03UFcB7iFpzy1aDy?usp=share_link',  # './split_processed_zip/3' folder in google drive
        'https://drive.google.com/drive/folders/1UKAntA7WliD84AUhRN224PkW4vt9agZW?usp=share_link',  # './split_processed_zip/4' folder in google drive
        'https://drive.google.com/drive/folders/11cCQw3qb1vJbviVPfBnOVWVzD_VzHdWs?usp=share_link',  # './split_processed_zip/5' folder in google drive
        'https://drive.google.com/drive/folders/1Wvy604wCxEdXAwE7r3sE0L0ieXvM__u8?usp=share_link',
        'https://drive.google.com/drive/folders/1BTv_pMTNGm7m3hD65IgBrX880v-rLIaf?usp=share_link',
        'https://drive.google.com/drive/folders/12x7F11ln2VQkqi8-Mu3kng74eLgifM0N?usp=share_link',
        'https://drive.google.com/drive/folders/1OWkOul2CCrqynqpt44Fu1CBxzNNfOFE2?usp=share_link',
        'https://drive.google.com/drive/folders/1ukwsfrbSEqCBNmRSuAYvYBHijWCQh2OU?usp=share_link',
        'https://drive.google.com/drive/folders/1EO7zumR6sVfsWQWCS6zfNs5WuO2Se6WX?usp=share_link',
        'https://drive.google.com/drive/folders/1aw0iBWvvZiSKng0ejRK8xbNoHLVUFCFu?usp=share_link',
        'https://drive.google.com/drive/folders/1szIcxlVyT5WJtzpqYWYlue0n82A6-xtk?usp=share_link',
    ]

    output = './'
    # VIDEO_PATH = 'processed_zip'
    # VIDEO_PATH = 'filter_processed_zip'
    VIDEO_PATH = 'split_processed_zip'
    for i, url in enumerate(urls):
        id = url.split('/')[-1]
        os.system(f"gdown --id {id} -O {output} --folder --no-cookies --remaining-ok")

        # unzip the zip files to the same location and delete zip files
        path_to_zip_file = str(i+1)
        zip_files = [join(path_to_zip_file, f) for f in listdir(path_to_zip_file)]
        for f in zip_files:
            if f.endswith(".zip"):
                directory_to_extract_to = VIDEO_PATH # extracted file itself contains a folder
                print(f'extract data {f} to {directory_to_extract_to}')
                with zipfile.ZipFile(f, 'r') as zip_ref:
                    zip_ref.extractall(directory_to_extract_to)
                os.remove(f)

else:
    compared_list = [1,4]  # compare videos for any two iterations
    VIDEO_INFOS = []
    for itr in compared_list:
        VIDEO_PATH = f'google_drive_data/processed/itr{itr}'
        VIDEO_INFOS.append(os.path.join(VIDEO_PATH, 'video_info.json'))
        
    # path_to_zip_file = VIDEO_PATH
    # zip_files = [join(path_to_zip_file, f) for f in listdir(path_to_zip_file)]
    # for f in zip_files:
    #     if f.endswith(".zip"):
    #         directory_to_extract_to = path_to_zip_file # extracted file itself contains a folder
    #         print(f'extract data {f} to {directory_to_extract_to}')
    #         with zipfile.ZipFile(f, 'r') as zip_ref:
    #             zip_ref.extractall(directory_to_extract_to)
    #         os.remove(f)
            
# for test only
# else:  # local data
#     VIDEO_PATH = 'test-data'

# VIDEO_INFO = os.path.join(VIDEO_PATH, 'video_info.json')

def inference(video_path):
    # for displaying mp4 with autoplay on Gradio
    with open(video_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    html = (
            f"""
            <video controls autoplay muted loop>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            </video> 
            """
    )
    return html

def video_identity(video):
    return video

def nan():
    return None

FORMAT = ['mp4', 'gif'][0]

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
    log_file = os.path.join(dataset_dir, f"flag_data_compare_{compared_list[0]}&{compared_list[1]}.csv")
    return repo, log_file


def update(user_choice, left, right, choose_env, data_folder=VIDEO_PATH, flag_to_huggingface=True):
    global last_left_video_path 
    global last_right_video_path 
    global last_infer_left_video_path
    global last_infer_right_video_path
    global repo
    global log_file
    
    if flag_to_huggingface: # log
        env_name = str(last_left_video_path).split('/')[4]  # 'robotinder-data/google_drive_data/raw/itrx/ENV_NAME/'
        current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        info = [env_name, user_choice, last_left_video_path, last_right_video_path, current_time]
        print(info)
        # repo, log_file = get_huggingface_dataset()
        with open(log_file, 'a') as file: # incremental change of the file
            writer_object = writer(file)
            writer_object.writerow(info)
            file.close()
        if int(current_time.split('-')[-2]) % 10 == 0:  # push only on certain minutes
            try:
                repo.push_to_hub(commit_message=f"Flagged sample at {current_time}")
            except:
                repo.git_pull(lfs=True)  # sync with remote first
                repo.push_to_hub(commit_message=f"Flagged sample at {current_time}")
    if choose_env == 'Random' or choose_env == '': # random or no selection
        envs = ENVS   
        env_name = envs[random.randint(0, len(envs)-1)]
    else:
        env_name = choose_env
    # choose video
    left, right = randomly_select_videos(env_name)

    last_left_video_path = left
    last_right_video_path = right
    last_infer_left_video_path = inference(left)
    last_infer_right_video_path = inference(right)
    
    return last_infer_left_video_path, last_infer_right_video_path, env_name

def replay(left, right):  
    return left, right

# def parse_envs(folder=VIDEO_PATH, filter=True, MAX_ITER=20000, DEFAULT_ITER=20000):
#     """
#     return a dict of env_name: video_paths
#     """
#     files = {}
#     if filter:
#         df = pd.read_csv('Bidexhands_Video.csv')
#         # print(df)
#     for env_name in os.listdir(folder):
#         env_path = os.path.join(folder, env_name)
#         if os.path.isdir(env_path):
#             videos = os.listdir(env_path)
#             video_files = []
#             for video in videos:  # video name rule: EnvName_Alg_Seed_Timestamp_Checkpoint_video-episode-EpisodeID
#                 if video.endswith(f'.{FORMAT}'):
#                     if filter:
#                         if len(video.split('_')) < 6:
#                             print(f'{video} is wrongly named.')
#                         seed = video.split('_')[2]
#                         checkpoint = video.split('_')[4]
#                         try:
#                             succeed_iteration = df.loc[(df['seed'] == int(seed)) & (df['env_name'] == str(env_name))]['succeed_iteration'].iloc[0]
#                         except:
#                             print(f'Env {env_name} with seed {seed} not found in Bidexhands_Video.csv')
                            
#                         if 'unsolved' in succeed_iteration:
#                             continue
#                         elif pd.isnull(succeed_iteration):
#                             min_iter = DEFAULT_ITER
#                             max_iter = MAX_ITER
#                         elif '-' in succeed_iteration:
#                             [min_iter, max_iter] = succeed_iteration.split('-')
#                         else:
#                             min_iter = succeed_iteration
#                             max_iter = MAX_ITER

#                         # check if the checkpoint is in the valid range
#                         valid_checkpoints = np.arange(int(min_iter), int(max_iter)+1000, 1000)
#                         if int(checkpoint) not in valid_checkpoints:
#                             continue
                    
#                     video_path = os.path.join(folder, env_name, video)
#                     video_files.append(video_path)
#                     # print(video_path)

#             files[env_name] = video_files

#     with open(VIDEO_INFO, 'w') as fp:
#         json.dump(files, fp)

#     return files

# def get_env_names():
#     with open(VIDEO_INFO, 'r') as fp:
#         files = json.load(fp)
#     return list(files.keys())

def randomly_select_videos(env_name):
    # load the parsed video info
    with open(VIDEO_INFOS[0], 'r') as fp:
        left_files = json.load(fp)
    with open(VIDEO_INFOS[1], 'r') as fp:
        right_files = json.load(fp)

    left_env_files = left_files[env_name]
    right_env_files = right_files[env_name]
    # randomly choose two videos
    if len(left_env_files) == 0 or len(right_env_files) == 0: # there may be unsolved hard tasks, e.g. ShadowHandKettle
        succeed = False
    else:
        left_video_id = np.random.choice(len(left_env_files))
        right_video_id = np.random.choice(len(right_env_files))
        selected_video_ids= [left_video_id, right_video_id]
        # selected_video_ids = np.random.choice(len(env_files), 2, replace=False)
        succeed = True
    try:
        left_video_path = left_env_files[left_video_id]
        right_video_path = right_env_files[right_video_id]
    except:
        print(f'error: {selected_video_ids}, env_files')

    # here is a modification for the video name rule
    if 'itr' not in left_video_path:
        index = left_video_path.find('processed/')
        left_video_path = left_video_path[:index+10] + f'itr{compared_list[0]}/' + left_video_path[index+10:]  # +10 to get the right position before iter
    if 'itr' not in right_video_path:
        index = right_video_path.find('processed/')
        right_video_path = right_env_files[:index+10] + f'itr{compared_list[1]}/' + right_video_path[index+10:]
    return left_video_path, right_video_path

def build_interface(data_folder=VIDEO_PATH):
    import sys
    import csv
    csv.field_size_limit(sys.maxsize)
    
    HF_TOKEN = os.getenv('HF_TOKEN')
    print(HF_TOKEN)
    HF_TOKEN = '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'  # my HF token
    ## hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "crowdsourced-robotinder-demo")  # HuggingFace logger instead of local one: https://github.com/gradio-app/gradio/blob/master/gradio/flagging.py
    ## callback = gr.CSVLogger()
    # hf_writer = HuggingFaceDatasetSaver(HF_TOKEN, "crowdsourced-robotinder-demo")
    # callback = hf_writer

    # parse the video folder 
    # files = parse_envs(filter=False)   

    global repo
    global log_file
    repo, log_file = get_huggingface_dataset() 
    
    # build gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("## Here is <span style=color:cyan>RoboTinder</span>!")
        gr.Markdown("### Select the best robot behaviour in your choice!")
        # some initial values
        env_name = ENVS[random.randint(0, len(ENVS)-1)] # random pick an env 
        with gr.Row():
            str_env_name = gr.Markdown(f"{env_name}")

        # choose video
        left_video_path, right_video_path = randomly_select_videos(env_name)
        
        with gr.Row():
            if FORMAT == 'mp4':
                # left = gr.PlayableVideo(left_video_path, label="left_video")
                # right = gr.PlayableVideo(right_video_path, label="right_video")

                infer_left_video_path = inference(left_video_path)
                infer_right_video_path = inference(right_video_path)
                left = gr.HTML(infer_left_video_path, label="left_video")
                right = gr.HTML(infer_right_video_path, label="right_video")
            else:
                left = gr.Image(left_video_path, shape=(1024, 768), label="left_video")
                # right = gr.Image(right_video_path).style(height=768, width=1024)
                right = gr.Image(right_video_path, label="right_video")

        global last_left_video_path 
        last_left_video_path = left_video_path
        global last_right_video_path 
        last_right_video_path = right_video_path

        global last_infer_left_video_path
        last_infer_left_video_path = infer_left_video_path
        global last_infer_right_video_path
        last_infer_right_video_path = infer_right_video_path

        # btn1 = gr.Button("Replay")
        # user_choice = gr.Radio(["Left", "Right", "Not Sure", "Both Good", "Both Bad"], label="Which one is your favorite?")
        user_choice = gr.Radio(["Left", "Right", "Not Sure"], label="Which one is your favorite?")
        choose_env = gr.Radio(["Random"]+ENVS, label="Choose the next task:")
        btn2 = gr.Button("Next")

        # This needs to be called at some point prior to the first call to callback.flag()
        # callback.setup([user_choice, left, right], "flagged_data_points")
        
        # btn1.click(fn=replay, inputs=[left, right], outputs=[left, right])
        btn2.click(fn=update, inputs=[user_choice, left, right, choose_env], outputs=[left, right, str_env_name])

        # We can choose which components to flag -- in this case, we'll flag all of them
        # btn2.click(lambda *args: callback.flag(args), [user_choice, left, right], None, preprocess=False)  # not using the gradio flagging anymore

    return demo

if __name__ == "__main__":
    last_left_video_path = None
    last_right_video_path = None

    demo = build_interface()
    demo.launch(share=True)  # local server
    # demo.launch(share=False)
