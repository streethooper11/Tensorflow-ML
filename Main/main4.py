import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import sklearn.model_selection


# Main Project Stage 4: Add non-numeric data
# Source: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.ngroup.html
# Convert text column to number by assigning a group number for each unique text value
def encode_text_to_numeric(input_data, text_column_list):
    return_data = input_data.copy()  # made a copy to prevent warning
    for text in text_column_list:
        return_data[text] = input_data.groupby(text).ngroup()

    return return_data


# Main Project Stage 3
# draws a line graph given data, with 'Age' as input and 'HeartDisease' as output
# y axis will be how likely heart disease is predicted with each age group
def draw_line_stage3(data):
    min_age = data['Age'].min()
    max_age = data['Age'].max()
    num_age_groups = (max_age // 10) - (min_age // 10) + 1  # number of dots in the line graph
    min_age = (data['Age'].min() // 10) * 10  # drop the last digit

    x = 'Age'
    y = 'Heart Disease Rate'
    graph_data_pred = pd.DataFrame(columns=[x, y])  # Custom column names
    graph_data_csv = pd.DataFrame(columns=[x, y])  # Custom column names

    for i in range(num_age_groups):
        min_age_in_group = min_age + (i * 10)
        max_age_in_group = min_age + ((i + 1) * 10)
        # Condition to filter age groups
        in_age_group = (data['Age'] >= min_age_in_group) & (data['Age'] < max_age_in_group)

        # Create dataframe with age group and maximum HR for people with output from the created model
        temp_data_pred = pd.DataFrame([{x: (str(min_age_in_group) + '-' + str(max_age_in_group - 1)),
                                        y: data['Predict'][in_age_group].mean()}])

        graph_data_pred = pd.concat([graph_data_pred, temp_data_pred], ignore_index=True)

        # Create dataframe with age group and maximum HR for people with output from the dataset
        temp_data_csv = pd.DataFrame([{x: (str(min_age_in_group) + '-' + str(max_age_in_group - 1)),
                                       y: data['HeartDisease'][in_age_group].mean()}])

        graph_data_csv = pd.concat([graph_data_csv, temp_data_csv], ignore_index=True)

    plt.plot(graph_data_pred[x], graph_data_pred[y], marker='o', color='b')
    plt.plot(graph_data_csv[x], graph_data_csv[y], marker='x', color='r')

    plot_then_save_graph(x, y, 'Average Rate of a Heart Disease with each Age Group', 'stage3.jpg')


# Step 3 of Main Project Stage 2
# Trains the model with the given input training data and output training data, and the given input shape
def train_model(input_train, output_train, shape):
    # make model; Edit for Project Stage 4
    model_result = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=shape),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_result.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # fit model
    model_result.fit(input_train, output_train, epochs=200, verbose=2)

    return model_result


# Step 3-3 of Main Project Stage 1
# It loads the cleaned data and saves a pie graph
# The pie will show % of individuals who get heart failure in relation to
# whether or not the individuals got exercise angina
def save_pie_part3(data):
    x = 'ExerciseAngina'
    y = 'HeartDisease'
    no_angina_disease = (data[x] == 'N') & (data[y] == 1)
    with_angina_disease = (data[x] == 'Y') & (data[y] == 1)

    plt.pie([data[x][no_angina_disease].count(), data[x][with_angina_disease].count()],
            labels=['No Angina', 'Has Angina'])
    plot_then_save_graph(x, y, 'Angina occurrence in individuals with a heart failure', 'plot3.jpg')


# Step 3-2 of Main Project Stage 1
# It loads the cleaned data and saves line graph
# x-axis will be age group ('Age'), with step of 10
# y-axis will be average maximum heart rate (uses mean of 'MaxHR' for each corresponding age group)
def save_line_part2(data):
    min_age = data['Age'].min()
    max_age = data['Age'].max()
    num_age_groups = (max_age // 10) - (min_age // 10) + 1  # number of dots in the line graph
    min_age = (data['Age'].min() // 10) * 10  # drop the last digit

    x = 'Age Group'
    y = 'Average Max Heart Rate'
    graph_data = pd.DataFrame(columns=[x, y])  # Custom column names

    for i in range(num_age_groups):
        min_age_in_group = min_age + (i * 10)
        max_age_in_group = min_age + ((i + 1) * 10)
        # Condition to filter age groups
        in_age_group = (data['Age'] >= min_age_in_group) & (data['Age'] < max_age_in_group)

        # Create dataframe with age group label and average maximum heart rate of all people in the age group
        temp_data = pd.DataFrame([{x: (str(min_age_in_group) + '-' + str(max_age_in_group - 1)),
                                   y: data['MaxHR'][in_age_group].mean()}])

        graph_data = pd.concat([graph_data, temp_data], ignore_index=True)

    plt.plot(graph_data[x], graph_data[y], marker='o')
    plot_then_save_graph(x, y, 'Average Maximum Heart Rate of each Age Group', 'plot2.jpg')


# It plots, saves, then clears graph
# x: Name of the x-axis of the graph
# y: Name of the y-axis of the graph
# title: Title of the graph
# output_file: File name to save the graph to
def plot_then_save_graph(x, y, title, output_file):
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(output_file)
    plt.clf()  # Clear the graph


# Step 3-1 of Main Project Stage 1
# It loads the cleaned data and saves histogram
# data: DataFrame to read from
# x: Name of the x-axis of the histogram and column name of the DataFrame
# y: Name of the y-axis of the histogram
# title: Title of the histogram
# output_file: File name to save the histogram to
def save_histogram(data, x, y, title, output_file):
    plt.hist(data[x])
    plot_then_save_graph(x, y, title, output_file)


# Step 2 of Main Project Stage 1
# It loads output data from load_then_save_input and cleans it
# Then saves an output as a file
def clean_then_save_data():
    data = pd.read_csv('data.csv')
    # Remove rows with a NaN value
    data.dropna()
    # 1. Remove rows with non-positive values for resting blood pressure level
    # 2. Remove rows with non-positive values for serum cholesterol level
    # 3. Remove rows with abnormal maximum heart rate:
    # Non-positive values are considered abnormal
    # Also, as maximum heart rate is usually calculated as 220 - age,
    # Heart rate higher than 220 are also considered abnormal
    data = data.drop(data.index[(data['RestingBP'] <= 0) |
                                (data['Cholesterol'] <= 0) |
                                (data['MaxHR'] <= 0) |
                                (data['MaxHR'] > 220)])
    data.to_csv('data-cleaned.csv', index=False)


# Step 1 of Main Project Stage 1
# It loads data-input.csv and then saves as output
# There is no merging as there is only one input file
def load_then_save_input():
    last_data = pd.DataFrame()
    for filepath in glob.glob('data-input*.csv'):
        data = pd.read_csv(filepath)
        last_data = pd.concat([last_data, data])

    last_data.to_csv('data.csv', index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sns.set()
    load_then_save_input()
    clean_then_save_data()
    cleaned_data = pd.read_csv('data-cleaned.csv')  # read the cleaned data to be used
    save_histogram(cleaned_data, 'Age', 'Count', 'Distribution of Age among individuals', 'plot1.jpg')
    save_line_part2(cleaned_data)
    save_pie_part3(cleaned_data)

    input_variables = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    output_variable = 'HeartDisease'
    (x_train, x_test, y_train, y_test) = sklearn.model_selection.train_test_split(
        cleaned_data[input_variables],
        cleaned_data[output_variable],
        random_state=0,
        train_size=0.8
    )  # Training data gets 80%

    # Main Project Stage 4: Add non-numeric data
    text_inputs = ['Sex', 'ChestPainType', 'RestingBP', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    x_train = encode_text_to_numeric(x_train, text_inputs)
    x_test = encode_text_to_numeric(x_test, text_inputs)

    input_shape = (11, 1)  # input shape for the dataset
    # train model
    model = train_model(x_train, y_train, input_shape)

    # evaluate model
    model_loss1, model_acc1 = model.evaluate(x_train,  y_train, verbose=2)
    model_loss2, model_acc2 = model.evaluate(x_test,  y_test, verbose=2)
    print(f"Train / Test Accuracy: {model_acc1*100:.1f}% / {model_acc2*100:.1f}%")

    # Main Project Stage 3 starts here; get input variables, predicted output, and csv output
    part3_data_input = cleaned_data[input_variables]

    # Modification for Stage 4 as input variables changed; encode text to numeric columns
    part3_data_input = encode_text_to_numeric(part3_data_input, text_inputs)
    # code modified from predict_test.py; include prediction values in the dataframe
    # Usage of insert method from https://sparkbyexamples.com/pandas/pandas-add-column-to-dataframe/
    part3_data_input.insert(0, 'Predict', model.predict(part3_data_input))  # Add prediction to 'Predict' column
    part3_data_input.insert(0, 'HeartDisease', cleaned_data['HeartDisease'])  # Add CSV output

    draw_line_stage3(part3_data_input)  # draw a line graph to show relationship
