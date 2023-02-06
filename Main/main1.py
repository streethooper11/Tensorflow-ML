import pandas as pd
import glob

import matplotlib.pyplot as plt
import seaborn as sns


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
