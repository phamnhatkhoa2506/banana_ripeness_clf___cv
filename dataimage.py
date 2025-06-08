import os
from roboflow import Roboflow
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()

    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    project = rf.workspace("waaaaaaaa").project("banana-ripeness-classification-sf26k")
    version = project.version(1)
    dataset = version.download("folder")