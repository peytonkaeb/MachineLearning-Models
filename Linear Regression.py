import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

# Display all columns when printing DataFrames
pd.set_option('display.max_columns', None)


# --------------------------------------------
# Load and preprocess the dataset
# --------------------------------------------

# Reads the csv file of data that trains our model:
df = pd.read_csv("C:\\Users\\skkae\\Downloads\\personality_datasert.csv")

# Encoding our data
# Label Encoding "Stage_fear":
# Creates a LabelEncoder object that converts text like yes or no into numbers like 1 and 0
label_encoder = preprocessing.LabelEncoder()

# .fit() tells the encoder to learn the unique values in stage_fear
# .transform() replaces those unique values with numeric code
df['Stage_fear']= label_encoder.fit_transform(df['Stage_fear'])


# Label Encoding "Drained_after_socializing":
label_encoder = preprocessing.LabelEncoder()
df['Drained_after_socializing']= label_encoder.fit_transform(df['Drained_after_socializing'])
df['Drained_after_socializing'].unique()

# Label Encoding "Personality":
label_encoder = preprocessing.LabelEncoder()
df['Personality']= label_encoder.fit_transform(df['Personality'])
df['Personality'].unique()


# Feature Scaling, scaled every feature that wasn't label encoded:
# Create a Normalized object from SkLearn:
scaler = Normalizer()

# Making a list of only the features we want to scale, not the one's we Label Encoded
features_to_scale = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                     'Friends_circle_size', 'Post_frequency']

# Grab the data from those columns and normalize them
#.fit() learns the structure
#.transform() applies normalization to each row
scaled_data = scaler.fit_transform(df[features_to_scale])

# Create a new Dataframe with names of the columns the same so it looks like the original table
scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale)

#Prints the first 5 rows
print(scaled_df.head())

# --------------------------------------------
# Model Training
# --------------------------------------------

# Load our data set
# Create the X and y arrays: X is input, Y is output
X = df[["Time_spent_Alone", "Stage_fear", "Social_event_attendance","Going_outside", "Drained_after_socializing", "Friends_circle_size","Post_frequency"]]
y = df["Personality"]

# Split the data set in a training set (75%) and a test set (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make predictions later
joblib.dump(model, 'PersonalityModel.pkl')

# User Prediction questions:
model = joblib.load('PersonalityModel.pkl')
print("Enter a number for each following prediction question:")


#Time_spent_Alone
print("How many hours do you spend alone daily: (0-11)")
PredictionTraits = []
PredictionTraits.append(int(input()))


#Stage_fear
print("Do you have stage fear: ")
print("0 = no, 1 = yes")
PredictionTraits.append(int(input()))


#Social_event_attendance
print("How often do you attend social events: (0-10)")
print("0 = never, 10= very frequent")
PredictionTraits.append(int(input()))


#Going_outside
print("How often do you go outside: (0-10)")
print("0 = never; 10 = very frequent")
PredictionTraits.append(int(input()))


#Drained_after_socializing
print("Do you feel drained after social events:")
print("0 = no, 1 = yes")
PredictionTraits.append(int(input()))


#Friends_circle_size
print("How many close friends do you have: (0-15)")
PredictionTraits.append(int(input()))


#Post_frequency
print("How often do you post on social media: (0-10)")
print("0 = never; 10 = very frequent")
PredictionTraits.append(int(input()))

traits = [PredictionTraits]

# Make prediction
trait_values = model.predict(traits)
predicted_value = trait_values[0]
rounded_prediction = round(predicted_value)

# Map 0/1 back to "Introvert"/"Extrovert"
personality_label = label_encoder.inverse_transform([rounded_prediction])[0]

print(f"Estimated Personality Type: {personality_label}")

# --------------------------------------------
# Model Evaluation
# --------------------------------------------

# Report how well the model is performing
print("Model training results:")

# Report an error rate on the training set
mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

# Report an error rate on the test set
mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Test Set Error: {mse_test}")