import os
from kaggle.api.kaggle_api_extended import KaggleApi
import argparse

parser = argparse.ArgumentParser(description='Submit a file to Kaggle competition')
parser.add_argument('--competition', type=str, required=True, help='Kaggle competition name')
parser.add_argument('--file', type=str, required=True, help='File to submit')
parser.add_argument('--message', type=str, default='Submission via script', help='Submission message')
args = parser.parse_args()

# Parameters
competition_name = args.competition
submission_file = args.file
message = args.message


def submit_to_kaggle(competition, file, msg):
    if not os.path.exists(file):
        print(f"Error: File '{file}' does not exist.")
        return

    api = KaggleApi()
    api.authenticate()
    
    print(f"Submitting '{file}' to competition '{competition}'...")
    api.competition_submit(file_name=file, competition=competition, message=msg)
    print("Submission successful!")

submit_to_kaggle(competition_name, submission_file, message)