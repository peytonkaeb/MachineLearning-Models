import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load and clean the dataset
df = pd.read_csv("C:\\Users\\skkae\\Downloads\\personality_datasert.csv")
df.dropna(inplace=True)  # Remove rows with missing data

# Step 2: Encode categorical columns into numbers
le_stage = LabelEncoder()
df['Stage_fear'] = le_stage.fit_transform(df['Stage_fear'])

le_drained = LabelEncoder()
df['Drained_after_socializing'] = le_drained.fit_transform(df['Drained_after_socializing'])

le_personality = LabelEncoder()
df['Personality'] = le_personality.fit_transform(df['Personality'])

# Step 3: Select features and label
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']
X = df[features]      # Features (inputs)
y = df['Personality'] # Target label (personality type)

# Step 4: Split the data into training and testing sets
# 80% of the data will be used for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the decision tree model on training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 6: Test the model on test data
y_pred = clf.predict(X_test)

# Step 7: Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("📊 Model Accuracy on Test Data:", round(accuracy * 100, 2), "%")

# Step 8: Ask the user for input to predict their personality

print("\nLet's predict your personality type! (0-10) 0 being not really, 10 being high \n")

# Ask user to enter values for each category
time_alone = float(input("On a scale of 0–10, how much time do you spend alone? "))
stage_fear_input = input("Do you have stage fear? (Yes/No): ").strip().capitalize()
social_events = float(input("On a scale of 0–10, how often do you attend social events? "))
going_outside = float(input("On a scale of 0–10, how often do you go outside? "))
drained_input = input("Do you feel drained after socializing? (Yes/No): ").strip().capitalize()
friends_circle = float(input("On a scale of 0–10, how big is your friend circle? "))
post_freq = float(input("On a scale of 0–10, how often do you post on social media? "))

# Create a dictionary of the user's input
new_person = {
    'Time_spent_Alone': time_alone,
    'Stage_fear': stage_fear_input,
    'Social_event_attendance': social_events,
    'Going_outside': going_outside,
    'Drained_after_socializing': drained_input,
    'Friends_circle_size': friends_circle,
    'Post_frequency': post_freq
}

# Encode the yes/no answers
new_person['Stage_fear'] = le_stage.transform([new_person['Stage_fear']])[0]
new_person['Drained_after_socializing'] = le_drained.transform([new_person['Drained_after_socializing']])[0]

# Convert to DataFrame for prediction
X_new = pd.DataFrame([new_person])[features]

# Predict personality
pred = clf.predict(X_new)[0]
predicted_label = le_personality.inverse_transform([pred])[0]

# Show result
print("\n Your predicted personality type is:", predicted_label)
