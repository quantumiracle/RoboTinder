## Collect Human Preference Data

* Put the collected trajectories data under: `google_drive_data/raw`.

* Process data: `python raw2processed.py`, the processed data is under: `google_drive_data/processed`.

* Parse the processed data with: `python parse_data.py` (no need to run this if next step is `python local_app.py`, build the app will also parse the data), this will create `video_info.json` under `google_drive_data/processed/iterx/`.

* Run human preference interface app locally:

  `python local_app.py`

* Evaluate two different results with human preference interface app locally:

  `python eval_app.py`

* Human preference data analysis: `flag_data_analyse.ipynb` , `per_user_analyse.ipynb`, `eval_data_analyze.ipynb`

## Train Reward Model

* Specify the frame stack number and training iteration in:

  `common.py`

* Process data for training reward model:

  `python rm_data.py`

  The processed data will be saved at: `processed_data`.

* Train the reward model with:

  `python train_rm.py`

  






* streamlit local run:
  `streamlit run streamlit_app.py --server.fileWatcherType none`

