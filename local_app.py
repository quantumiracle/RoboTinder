import gradio as gr
import os
import random
import base64
import numpy as np
from utils.flagging import CSVLogger
from time import gmtime, strftime
from csv import writer
import json
import pandas as pd

# ENVS = ['ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
# 'ShadowHandCatchOver2Underarm', 'ShadowHandBottleCap', 'ShadowHandLiftUnderarm', 'ShadowHandTwoCatchUnderarm',
# 'ShadowHandDoorOpenInward', 'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward', 'ShadowHandDoorCloseOutward',
# 'ShadowHandPushBlock', 'ShadowHandKettle', 
# 'ShadowHandScissors', 'ShadowHandPen', 'ShadowHandSwingCup', 'ShadowHandGraspAndPlace', 'ShadowHandSwitch']

ENVS = ['ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
        'ShadowHandCatchOver2Underarm', 'ShadowHandBottleCap', 'ShadowHandLiftUnderarm', 'ShadowHandTwoCatchUnderarm', 'ShadowHandDoorOpenInward',
        'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward', 'ShadowHandGraspAndPlace', 'ShadowHandPushBlock',
        'ShadowHandScissors', 'ShadowHandPen', 'ShadowHandSwitch']


# unseen 4 envs
# ENVS = ['ShadowHandCatchAbreastPen', 'ShadowHandCatchUnderarmPen', 'ShadowHandTwoCatchAbreast', 'ShadowHandGraspAndPlaceEgg']


def inference(video_path):
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

# demo = gr.Interface(video_identity, 
#                     gr.Video(), 
#                     "playable_video", 
#                     examples=[
#                         os.path.join(os.path.dirname(__file__), 
#                                      "videos/rl-video-episode-0.mp4")], 
#                     cache_examples=True)

FORMAT = ['mp4', 'gif'][0]
# VIDEO_PATH = 'test-data'
itr = 4
VIDEO_PATH = f'google_drive_data/processed/itr{itr}'
VIDEO_INFO = os.path.join(VIDEO_PATH, 'video_info.json')

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
    log_file = os.path.join(dataset_dir, "local_flag_data.csv")
    return repo, log_file

def update(user_choice, user_name, left, right, choose_env, data_folder=VIDEO_PATH, flag_to_huggingface=True):
    global last_left_video_path 
    global last_right_video_path 
    global last_infer_left_video_path
    global last_infer_right_video_path

    if flag_to_huggingface: # log
        env_name = str(last_left_video_path).split('/')[1]  # 'robotinder-data/ENV_NAME/'
        current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        info = [env_name, user_choice, last_left_video_path, last_right_video_path, current_time, user_name]
        print(info)
        repo, log_file = get_huggingface_dataset()
        with open(log_file, 'a') as file: # incremental change of the file
            writer_object = writer(file)
            writer_object.writerow(info)
            file.close()
        if int(current_time.split('-')[-2]) % 2 == 0:  # push only on certain minutes
            try:
                repo.push_to_hub(commit_message=f"Flagged sample at {current_time}")
            except:
                repo.git_pull(lfs=True)  # sync with remote first
                repo.push_to_hub(commit_message=f"Flagged sample at {current_time}")
    if choose_env == 'Random' or choose_env == '': # random or no selection
        envs = get_env_names()   
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

# def update():
#     if FORMAT == 'mp4':
#         left = os.path.join(os.path.dirname(__file__), 
#                              "videos/rl-video-episode-2.mp4")
#         right = os.path.join(os.path.dirname(__file__), 
#                              "videos/rl-video-episode-3.mp4")
#     else:
#         left = os.path.join(os.path.dirname(__file__), 
#                              "videos/rl-video-episode-2.gif")
#         right = os.path.join(os.path.dirname(__file__), 
#                              "videos/rl-video-episode-3.gif")  
#     print(left, right)     
#     return left, right

def replay(left, right):  
    return left, right

def parse_envs(folder=VIDEO_PATH, filter=False, MAX_ITER=20000, DEFAULT_ITER=20000):
    """
    return a dict of env_name: video_paths
    """
    files = {}
    if filter:
        df = pd.read_csv('Bidexhands_Video.csv')
        # print(df)
    for env_name in os.listdir(folder):
        env_path = os.path.join(folder, env_name)
        if os.path.isdir(env_path):
            videos = os.listdir(env_path)
            video_files = []
            for video in videos:  # video name rule: EnvName_Alg_Seed_Timestamp_Checkpoint_video-episode-EpisodeID
                if video.endswith(f'.{FORMAT}'):
                    if filter:
                        if len(video.split('_')) < 6:
                            print(f'{video} is wrongly named.')
                        seed = video.split('_')[2]
                        checkpoint = video.split('_')[4]
                        try:
                            succeed_iteration = df.loc[(df['seed'] == int(seed)) & (df['env_name'] == str(env_name))]['succeed_iteration'].iloc[0]
                        except:
                            print(f'Env {env_name} with seed {seed} not found in Bidexhands_Video.csv')
                            
                        if 'unsolved' in succeed_iteration:
                            continue
                        elif pd.isnull(succeed_iteration):
                            min_iter = DEFAULT_ITER
                            max_iter = MAX_ITER
                        elif '-' in succeed_iteration:
                            [min_iter, max_iter] = succeed_iteration.split('-')
                        else:
                            min_iter = succeed_iteration
                            max_iter = MAX_ITER

                        # check if the checkpoint is in the valid range
                        valid_checkpoints = np.arange(int(min_iter), int(max_iter)+1000, 1000)
                        if int(checkpoint) not in valid_checkpoints:
                            continue
                    
                    video_path = os.path.join(folder, env_name, video)
                    video_files.append(video_path)
                    # print(video_path)

            files[env_name] = video_files

    with open(VIDEO_INFO, 'w') as fp:
        json.dump(files, fp)

    return files

def get_env_names():
    with open(VIDEO_INFO, 'r') as fp:
        files = json.load(fp)
    return list(files.keys())

def randomly_select_videos(env_name):
    # load the parsed video info
    with open(VIDEO_INFO, 'r') as fp:
        files = json.load(fp)
    env_files = files[env_name]
    # randomly choose two videos
    selected_video_ids = np.random.choice(len(env_files), 2, replace=False)
    left_video_path = env_files[selected_video_ids[0]]
    right_video_path = env_files[selected_video_ids[1]]
    return left_video_path, right_video_path

def build_interface(iter=3, data_folder=VIDEO_PATH):
    # callback = gr.CSVLogger()  # this one works well locally
    callback = CSVLogger()  # used for debugging
    import sys
    import csv

    csv.field_size_limit(sys.maxsize)
    
    # parse the video folder 
    files = parse_envs()   

    # build gradio interface
    with gr.Blocks() as demo:
        # gr.Markdown("## Here is **RoboTinder**!")
        gr.Markdown("## Here is <span style=color:cyan>RoboTinder</span>!")
        # gr.Markdown("## Here is <span class=rainbow-lr>RoboTinder</span>!") # https://rainbowcoding.com/posts/how-to-create-rainbow-text-in-html-css-javascript
        gr.Markdown("### Select the best robot behaviour in your choice!")

        # some initial values
        env_name = list(files.keys())[random.randint(0, len(files)-1)] # random pick an env 
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
                left = gr.Image(left_video_path, label="left_video", shape=(768, 512))
                # right = gr.Image(right_video_path).style(height=768, width=1024)
                right = gr.Image(right_video_path, label="right_video", shape=(768, 512))

        global last_left_video_path 
        last_left_video_path = left_video_path
        global last_right_video_path 
        last_right_video_path = right_video_path

        global last_infer_left_video_path
        last_infer_left_video_path = infer_left_video_path
        global last_infer_right_video_path
        last_infer_right_video_path = infer_right_video_path

        btn1 = gr.Button("Replay")
        user_name = gr.Textbox(label='Your name:')
        user_choice = gr.Radio(["Left", "Right", "Not Sure"], label="Which one is your favorite?")
        choose_env = gr.Radio(["Random"]+ENVS, label="Choose the next task:")
        btn2 = gr.Button("Next")

        # This needs to be called at some point prior to the first call to callback.flag()
        callback.setup([user_choice, left, right], "flagged_data_points")
        btn1.click(fn=replay, inputs=[left, right], outputs=[left, right])
        btn2.click(fn=update, inputs=[user_choice, user_name, left, right, choose_env], outputs=[left, right, str_env_name])  # preprocess will create tmp file for image: https://github.com/gradio-app/gradio/issues/1679 
        # import  pdb; pdb.set_trace()

        # We can choose which components to flag -- in this case, we'll flag all of them
        btn2.click(lambda *args: callback.flag(args), [user_choice, left, right], None, preprocess=False, postprocess=False)
    
    return demo

if __name__ == "__main__":
    last_left_video_path = None
    last_right_video_path = None

    demo = build_interface()
    # demo.launch(share=True)
    demo.launch(share=False)
