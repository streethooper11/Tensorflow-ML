# cpsc501_assignment2



## Dataset

I am using the ""Heart Failure Prediction Dataset", found on Kaggle with Open Database License

[Source](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction "Source")

From the source:\
"Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help."

There is 1 table, and it has a total of 12 columns:
- 6 numerical input columns
- 5 text/category input columns
- 1 numerical output column "HeartDisease" which predicts whether the client in each row will likely get one.

There are a total of 918 rows at the time of writing.

Below is a screenshot of a part of the dataset:
![Image of data table](/participation/Participation_3.png)

## Dockerfile

The Dockerfile at the root repository sets up a python environment with packages needed for the assignment.

Example command after building the docker image (for Windows):\
docker run -it --rm -v "%cd%/Part 1":/app "image name" python /app/MNIST-Complete.py