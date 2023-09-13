# Data Science Lab 2023: Group 5 Targaryen 🐉
This repository contains our project of the phase 1 of the **Practical Course: Data Science for Scientific Data** at Karlsruhe Institute of Technology (KIT). The project is about the 'Richter's Predictor: Modeling Earthquake Damage' competition ([Link](https://www.drivendata.org/competitions/57/)).

## Group Members 👤 
| Forename | Surname  | Matr.#  |
|----------|----------|---------|
| Nina     | Mertins  | 2107539 |
| Kevin    | Hartmann | 1996265 |
| Alessio  | Negrini  | 2106547 |

## Folder Structure 🗂️
```
📦phase-1
 ┣ 📂config                    <-- Configuration files for the pipeline
 ┣ 📂data                      <-- Data used as input during development with Jupyter notebooks. 
 ┃ ┣ 📂predictions             <-- Contains the predicted data build during development.
 ┃ ┣ 📂raw                     <-- Contains the raw data provided by the supervisors.
 ┃ ┗ 📂processed               <-- Contains the processed data build during development.
 ┣ 📂models                    <-- Saved models during Development.
 ┣ 📂notebooks                 <-- Jupyter Notebooks used in development.
 ┃ ┗ 📂weekXX                  <-- Contains the notebooks for weekly subtasks and experimenting.
 ┣ 📂src                       <-- Source code.
 ┃ ┣ 📜data_cleaning.py        <-- Contains the functions for data cleaning.
 ┃ ┣ 📜feature_engineering.py  <-- Contains the functions for feature engineering.
 ┃ ┣ 📜feature_selection.py    <-- Contains the functions for feature selection.
 ┃ ┣ 🕹️main.py                 <-- Main file for running the pipeline.
 ┃ ┗ 📜modelling.py            <-- Contains the functions for model training.
 ┣ 📜.gitignore                <-- Specifies intentionally untracked files to ignore when using Git.
 ┣ 📜README.md                 <-- The top-level README for developers using this project. 
 ┗ 📜requirements.txt          <-- The requirenments file for reproducing the environment, e.g. generated with 
                                    'pip freeze > requirenments.txt'.
```

## Setting up the environment and run the code ▶️

1. Clone the repository by running the following command in your terminal:

   ```
   git clone https://git.scc.kit.edu/data-science-lab-2023/group-5-targaryen/phase-1.git
   ```


2. Navigate to the project root directory by running the following command in your terminal:

   ```
   cd phase-1
   ```

3. [Optional] Create a virtual environment and activate it. For example, using the built-in `venv` module in Python:
   ```
   python3 -m venv venv-phase-1
   source venv-phase-1/bin/activate
   ```

4. Install the required packages by running the following command in your terminal:

   ```
   pip install -r requirements.txt
   ```

5. Place the data in the phase-1/data/raw folder. Ensure that the data is in the appropriate format and structure 
required by the pipeline. The dataset can be downloaded as is from the 
[competition website](https://www.drivendata.org/competitions/57/nepal-earthquake/).

6. Run the pipeline with the following command:

   ```
   python3 src/main.py --config "configs/config.yml"
   ```

By following these steps, you should be able to successfully run the  pipeline on the data and obtain the desired 
results. You can also monitor the pipeline's progress through the logs printed in the terminal. If any errors or issues 
occur, the logs will provide valuable information for troubleshooting. 