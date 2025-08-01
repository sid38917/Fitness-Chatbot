import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random
import os
import csv
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import json
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import nltk


from database_user import create_user, update_user, initialize_db, collect_details, change_user_name, store_maintenance_calories, record_purchase, user_has_plan, get_fitness_goal, set_last_active_user, get_last_active_user
from sentiment import train_sentiment_model, is_sentiment_related
from workouts_file import log_workout, log_weight, log_diet, display_progress, delete_workout_entry, delete_diet_entry

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')



sentiment_clf, count_vect, tf_idf_transform_sentiment = train_sentiment_model()

#loads datasets for main classifier
small_talk = pd.read_csv('small_talk.csv')
fitness_functionality = pd.read_csv('functionality.csv')
qanda = pd.read_csv('qanda_dataset.csv')

#standardizes category labels
small_talk['Category'] = 'small_talk'
fitness_functionality['Category'] = 'fitness_query'
qanda['Category'] = 'qanda'

#combine all the datasets for main classifier
data = pd.concat([small_talk, qanda, fitness_functionality], ignore_index=True)

#lemmatizes input
def preprocess_input(text):
    text = text.lower()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

#apply preprocessing to dataset
data['ProcessedQuestions'] = data['Questions'].apply(preprocess_input)

#main classifier vectorization
vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
X = vectorizer.fit_transform(data['ProcessedQuestions']).toarray()
y = data['Category']

#train test split for main classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
svm_classifier = LinearSVC()
svm_classifier.fit(X_train, y_train)


#svm classifier for main classifier
y_pred = svm_classifier.predict(X_test)
# print("Main Classifier Accuracy:", accuracy_score(y_test, y_pred))

nb_classifier = MultinomialNB()

#train the naive baiyes classifier and prints the classification accuracy
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
# print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

#train the random forest classifier and prints the classification accuracy
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


#obtains the fitness data for the fitness sub-classifier
fitness_data = pd.read_csv('combined_fitness.csv')
fitness_data.dropna(subset=['Questions', 'Category'], inplace=True)
X_fitness = fitness_data['Questions']
y_fitness = fitness_data['Category']
X_train_fitness, X_test_fitness, y_train_fitness, y_test_fitness = train_test_split(
    X_fitness, y_fitness, test_size=0.3, random_state=1)
vectorizer_fitness = TfidfVectorizer(max_features=1000)
X_train_fitness_tfidf = vectorizer_fitness.fit_transform(X_train_fitness)
X_test_fitness_tfidf = vectorizer_fitness.transform(X_test_fitness)
classifier_fitness = LinearSVC()
classifier_fitness.fit(X_train_fitness_tfidf, y_train_fitness)
y_pred_fitness_clf = classifier_fitness.predict(X_test_fitness_tfidf)
# print("Fitness Sub Classifier Accuracy:", accuracy_score(y_test_fitness, y_pred_fitness_clf))

fat_loss_qanda = pd.read_csv('fat_loss_answers.csv')
muscle_building_qanda = pd.read_csv('muscle_building_answers.csv', engine='python', quoting=csv.QUOTE_ALL, on_bad_lines='skip')
powerlifting_qanda = pd.read_csv('powerlifting_answers.csv', engine='python', quoting=csv.QUOTE_ALL, on_bad_lines='skip')

fitness_answers_data = {
    'fat_loss': fat_loss_qanda,
    'muscle_building': muscle_building_qanda,
    'powerlifting': powerlifting_qanda
}


fitness_data_score = {}
for category, df in fitness_answers_data.items():
    df.dropna(subset=['Questions', 'Answers'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    fitness_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    fitness_vectorizer.fit(df['Questions'])
    question_vectors = fitness_vectorizer.transform(df['Questions'])
    fitness_data_score[category] = {
        'vectorizer': fitness_vectorizer,
        'question_vectors': question_vectors,
        'questions': df['Questions'],
        'answers': df['Answers']
    }

#load Q&A data
qanda_data = pd.read_csv("qanda_answers_dataset.csv")
qanda_data['ProcessedQuestions'] = qanda_data['Questions'].apply(preprocess_input)

#vectorize processed qanda questions
vectorizer_questions = TfidfVectorizer(ngram_range=(1, 2))
question_vectors = vectorizer_questions.fit_transform(qanda_data['ProcessedQuestions'])

def find_best_answer(user_input, qanda_data, question_vectors, vectorizer_questions):
    processed_input = preprocess_input(user_input)
    input_vector = vectorizer_questions.transform([processed_input])
    similarities = cosine_similarity(input_vector, question_vectors)
    max_similarity = similarities.max()
    if max_similarity < 0.2:
        return None
    best_match_idx = similarities.argmax()
    best_match_answer = qanda_data['Answers'].iloc[best_match_idx]
    return best_match_answer


#small talk subclassifier
small_talk_data = pd.read_csv('small_talk_classified.csv', on_bad_lines='skip')
small_talk_data['ProcessedQuestions'] = small_talk_data['Questions'].apply(preprocess_input)
vectorize_small_talk = TfidfVectorizer()
X_small_talk = vectorize_small_talk.fit_transform(small_talk_data['ProcessedQuestions'])
y_small_talk = small_talk_data['Category']
X_train_small_talk, X_test_small_talk, y_train_small_talk, y_test_small_talk = train_test_split(
    X_small_talk, y_small_talk, test_size=0.3, random_state=1)
small_talk_classifier = LinearSVC()
small_talk_classifier.fit(X_train_small_talk, y_train_small_talk)
responses_data = pd.read_csv("small_talk_responses.csv")
responses_data['ProcessedQuestions'] = responses_data['Questions'].apply(preprocess_input)
vectorizer_responses = TfidfVectorizer()
response_vectors = vectorizer_responses.fit_transform(responses_data['ProcessedQuestions'])
y_pred_small_talk = small_talk_classifier.predict(X_test_small_talk)
accuracy_small = accuracy_score(y_test_small_talk, y_pred_small_talk)
# print(f"Accuracy of the small talk classifier: {accuracy_small}")

responses = {
    "greeting": ["Hi there!", "Hello!", "Hey! How can I help you?"],
    "farewell": ["Goodbye!", "See you later!", "Take care!"],
    "insults": ["I'm here to help, not to argue.", "Let's keep this conversation positive."],
    "personal_interaction": ["I am here to help you with your fitness goals whether it is making fitness plans or diet plans", "I am here to answer any questions regarding your fitness goals"],
    "contextual_generic": ["Could you please elaborate?", "I'm not sure I understand. Can you provide more details?"],
    "casual_conversation": ["That's interesting!", "Tell me more about that."]
}



def confirm_name():
    joke_phrases = [
        'Why don’t scientists trust atoms? Because they make up everything!',
        'Why was Cinderella so bad at football? She kept running away from the ball',
        'Where do the fish keep their money? In the river bank!'
    ]

    last_user = get_last_active_user()
    if last_user:
        confirmation = input(f"Am I still talking to {last_user}? (yes/no): ").strip().lower()
        if confirmation == 'yes':
            user_mood = input("John: How are you feeling today? ").strip()
            sentiment = analyze_sentiment_initial(user_mood)
            if sentiment == "negative":
                print("I am sorry to hear that, I hope you feel better later")
                tell_joke = input("John: Would you like me to tell you a joke in the mean time: ").strip().lower()
                if tell_joke == "yes":
                    random_joke = random.choice(joke_phrases)
                    print(f"John: {random_joke}")
                else:
                    print("Well I hope you generally feel better then")
            else:
                print("I am glad to hear that")
            return last_user
        else:
            #if user says no prompt a new name
            username = input("What's your name: ").strip()
            create_user(username)
            set_last_active_user(username)
            print(f"Nice to meet you {username}")
            return username
    else:
        #if user not found
        username = input("What's your name: ").strip()
        create_user(username)
        set_last_active_user(username)
        print(f"Nice to meet you {username}")
        return username

def get_user_info(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT weight, height, age, gender, maintenance_calories FROM users WHERE username = ?',
                   (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        weight, height, age, gender, maintenance_calories = result
        return {
            'weight': weight,
            'height': height,
            'age': age,
            'gender': gender,
            'maintenance_calories': maintenance_calories
        }
    return None

#calculates maintenance calories
def calc_maintenance_calories(weight, height, age, gender):
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    maintenance_cal = bmr * 1.55
    return round(maintenance_cal)


def analyze_sentiment(user_input, classifier, count_vector, tf_idf):
    processed_input = count_vector.transform([user_input])
    input_tf = tf_idf.transform(processed_input)
    return classifier.predict(input_tf)[0]


def mood_response(username, user_input, classifier, count_vector, tf_idf):
    jokes = ["Why don’t scientists trust atoms? Because they make up everything!", "Why did the chicken cross the road? I don't know", "What would bears be without bees? Ears", "Why was cinderella so bad at football? she kept running away from the ball", "What did the tomato say to the other tomato during a race? Ketchup"]
    sentiment = analyze_sentiment(user_input, classifier, count_vector, tf_idf)
    if sentiment == "positive":
        print(f"John: Great to hear you're doing well, {username}! How can I assist you today?")
    elif sentiment == 'negative':
        print(f"John: I'm sorry to hear that, {username}. Would a motivational tip or a joke help?")
        next_action = input("John: Would you like a motivational tip or a joke? (tip/joke/no): ").strip().lower()
        if next_action == "tip":
            print("John: Stay consistent, and you'll achieve your fitness goals!")
        elif next_action == "joke":
            random_joke = random.choice(jokes)
            print(f"John: {random_joke}")
        else:
            print("John: Let me know how else I can assist.")
    else:
        print(f"John: Thanks for sharing, {username}. How can I help you further?")

#if answer includes time, date or user name
def preprocess_response(response, user_name):
    if "{current_time}" in response:
        current_time = datetime.now().strftime("%H:%M:%S")
        response = response.replace("{current_time}", current_time)
    if "{current_date}" in response:
        current_date = datetime.now().strftime("%Y-%m-%d")
        response = response.replace("{current_date}", current_date)
    if "{user_name}" in response and user_name:
        response = response.replace("{user_name}", user_name)
    return response


def find_best_response(user_input, user_name, category=None):
    processed_input = preprocess_input(user_input)
    #filter responses by category
    if category:
        filtered_responses = responses_data[responses_data['Category'] == category].reset_index(drop=True)
    else:
        filtered_responses = responses_data

    if filtered_responses.empty:
        return None
    #vectorize filtered responses and the user input
    response_vectors_filtered = vectorizer_responses.transform(filtered_responses['ProcessedQuestions'])
    user_vector = vectorizer_responses.transform([processed_input])
    similarities = cosine_similarity(user_vector, response_vectors_filtered)
    max_similarity = similarities.max()
    #threshold
    if max_similarity < 0.2:
        return None
    best_match_idx = similarities.argmax()
    matched_response = filtered_responses['Responses'].iloc[best_match_idx]
    return preprocess_response(matched_response, user_name)


def ensure_maintenance_calories(username):
    user_info = get_user_info(username)
    #check if user details are missing
    if user_info is None or any(user_info[k] is None for k in ['weight', 'height', 'age', 'gender']):
        print("John: I need your details to calculate your maintenance calories.")
        weight, height, age, gender = collect_details()
        update_user(username, weight, height, age, gender)
        user_info = get_user_info(username)

    #if still missing details after attempting to collect
    if user_info is None or any(user_info[k] is None for k in ['weight', 'height', 'age', 'gender']):
        print("John: Sorry, I still don't have all your details. Cannot compute maintenance calories.")
        return None

    #check if maintenance_calories is already stored
    maintenance_calories = user_info['maintenance_calories']
    if maintenance_calories is None:
        maintenance_calories = calc_maintenance_calories(
            user_info['weight'], user_info['height'], user_info['age'], user_info['gender']
        )
        store_maintenance_calories(username, maintenance_calories)
    return maintenance_calories


def collect_fitness_details(missing_info):
    #start with defaults as None
    weight = None
    height = None
    age = None
    gender = None

    if 'weight' in missing_info:
        while True:
            try:
                weight = float(input("John: Please enter your weight in kilograms (kg): ").strip())
                if weight <= 0:
                    print("John: Please enter a positive number for weight.")
                    continue
                break
            except ValueError:
                print("John: Invalid input, please enter a numeric value for weight.")

    if 'height' in missing_info:
        while True:
            try:
                height = float(input("John: Please enter your height in centimeters (cm): ").strip())
                if height <= 0:
                    print("John: Please enter a positive number for height.")
                    continue
                break
            except ValueError:
                print("John: Invalid input, please enter a numeric value for height.")

    if 'age' in missing_info:
        while True:
            try:
                age = int(input("John: Please enter your age in years: ").strip())
                if age <= 0:
                    print("John: Please enter a positive number for age.")
                    continue
                break
            except ValueError:
                print("John: Invalid input, please enter a numeric value for age.")

    if 'gender' in missing_info:
        while True:
            gender_input = input("John: Please enter your gender (male/female): ").strip().lower()
            if gender_input not in ['male', 'female']:
                print("John: Invalid gender. Please enter 'male' or 'female'.")
            else:
                gender = gender_input
                break

    # Return all collected details (some might have been None if they weren't missing)
    return weight, height, age, gender

def set_fitness_goals(username):
    print("John: What is your primary fitness goal?")
    print("1. Fat Loss")
    print("2. Muscle Gain")
    print("3. Powerlifting")
    goal_map = {"1": "Fat Loss", "2": "Muscle Gain", "3": "Powerlifting"}
    choice = input("John: Please select an option (1/2/3): ").strip()

    if choice in goal_map:
        fitness_goal = goal_map[choice]
        conn = sqlite3.connect('user_data.db')
        cursor = conn.cursor()
        cursor.execute('''
                INSERT INTO user_goals (username, fitness_goal)
                VALUES (?, ?)
                ON CONFLICT(username) DO UPDATE SET fitness_goal=excluded.fitness_goal
            ''', (username, fitness_goal))
        conn.commit()
        conn.close()
        print(f"John: Great! Your fitness goal is set to '{fitness_goal}'.")
        return fitness_goal
    else:
        print("John: Invalid choice. Please try again.")
        return set_fitness_goals(username)

def start_fitness(username):
    user_info = get_user_info(username)


    if user_info is None or any(user_info[k] is None for k in ['weight', 'height', 'age', 'gender']):
        print("John: I need your details to start your fitness journey.")
        weight, height, age, gender = collect_details()
        update_user(username, weight, height, age, gender)
        user_info = get_user_info(username)


    if user_info is None or any(user_info[k] is None for k in ['weight', 'height', 'age', 'gender']):
        print("John: Sorry, I still don't have all your details.")
        return

    #maintenance calories
    maintenance_calories = user_info['maintenance_calories']
    if maintenance_calories is None:
        maintenance_calories = calc_maintenance_calories(
            user_info['weight'], user_info['height'], user_info['age'], user_info['gender']
        )
        store_maintenance_calories(username, maintenance_calories)

    #suggests calories and set goal
    print(f"John: Your maintenance calories are approximately {maintenance_calories} kcal per day.")
    goal = set_fitness_goals(username)

    # Suggest plans based on goals
    if goal == "Fat Loss":
        print("John: Based on your goal, I recommend our Fat Loss Plan.")
    elif goal == "Muscle Gain":
        print("John: Based on your goal, I recommend our Muscle Building Plan.")
    elif goal == "Powerlifting":
        print("John: Based on your goal, I recommend our Powerlifting Plan.")

    #offer plans after goal setting
    plan_selected = offer_fitness_plans()
    confirm_plan_purchase(username, plan_selected)


def get_maintenance_calories_suggestion(maintenance_calories):
    if maintenance_calories < 2000:
        return "Your maintenance calories are on the lower side. You might consider a muscle building phase to increase strength and metabolism."
    elif 2000 <= maintenance_calories <= 2500:
        return "Your maintenance calories are moderate. You could maintain or choose a plan to reach a specific goal, like muscle building or fat loss."
    else:
        return "You have a relatively high maintenance calorie baseline, which might make powerlifting or building strength more suited to you."

def offer_fitness_plans():
    print("John: I have three plans for you to choose from:")
    print("1. Powerlifting")
    print("2. Muscle Building")
    print("3. Fat Loss")
    choice = input("John: Which plan would you like to choose? (1/2/3): ").strip()
    if choice == '1':
        return "Powerlifting"
    elif choice == '2':
        return "Muscle Building"
    elif choice == '3':
        return "Fat Loss"
    else:
        print("John: Invalid choice. Please try again.")
        return offer_fitness_plans()

def display_csv_plan(csv_filename):
    #reads CSV file and prints it out in a formatted manner
    try:
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # First row is headers
            print("John: Here is your plan:")
            print(", ".join(headers))
            for row in reader:
                print(", ".join(row))
    except FileNotFoundError:
        print(f"John: Sorry, I couldn't find the file {csv_filename}. Please contact support.")

def prompt_plan_purchase():
    plan_prices = {
        "1": {"name": "Powerlifting", "price": 49.99},
        "2": {"name": "Muscle Building", "price": 39.99},
        "3": {"name": "Fat Loss", "price": 29.99}
    }

    print("John: You haven't purchased a plan yet. Here are our options:")
    for number, details in plan_prices.items():
        print(f"{number}. {details['name']}: ${details['price']}")

    #prompt
    choice = input("John: Please select a plan (1/2/3) or type 'no' to cancel: ").strip()

    if choice.lower() == 'no':
        print("John: No problem. Let me know if you change your mind.")
        return None
    elif choice in plan_prices:
        selected_plan = plan_prices[choice]
        confirm = input(
            f"John: The {selected_plan['name']} plan costs ${selected_plan['price']}. Confirm purchase? (yes/no): ").strip().lower()
        if confirm == 'yes':
            return selected_plan['name']
        else:
            print("John: Purchase canceled.")
            return None
    else:
        print("John: Invalid choice. Please try again.")
        return prompt_plan_purchase()  #retry if invalid choice

def analyze_sentiment_initial(user_input):
    negative_keywords = ["bad", "sad", "terrible", "awful", "depressed", "unhappy", "horrible", "sick"]
    if any(word in user_input.lower() for word in negative_keywords):
        return "negative"
    else:
        return "positive"


def get_plan_files(plan_name):
    plan_files = {
        "Powerlifting": {
            "diet": "powerlifting_diet.csv",
            "workout": "powerlifting_split_paid.csv"
        },
        "Muscle Building": {
            "diet": "bodybuilding_diet.csv",
            "workout": "bodybuilding_workoutplan.csv"
        },
        "Fat Loss": {
            "diet": "fat_loss_dietplan.csv",
            "workout": "fat_loss_workout.csv"
        }
    }
    return plan_files.get(plan_name, None)


def handle_plan_request(username, plan_type):
    user_plans = user_has_plan(username)
    if user_plans:
        #is user has more than one plan
        if len(user_plans) == 1:
            #show the single plan they have
            plan_name = user_plans[0]
            files = get_plan_files(plan_name)
            if files and plan_type in files:
                display_csv_plan(files[plan_type])
            else:
                print(f"John: Sorry, I don't have a {plan_type} for the {plan_name} plan.")
        else:
            #multiple plans
            print("John: You have multiple plans. Which one would you like to see?")
            for i, p in enumerate(user_plans, start=1):
                print(f"{i}. {p}")
            choice = input("John: Please choose a plan number: ").strip()
            try:
                choice = int(choice)-1
                if 0 <= choice < len(user_plans):
                    selected_plan = user_plans[choice]
                    files = get_plan_files(selected_plan)
                    if files and plan_type in files:
                        display_csv_plan(files[plan_type])
                    else:
                        print(f"John: Sorry, I don't have a {plan_type} for the {selected_plan} plan.")
                else:
                    print("John: Invalid choice.")
            except ValueError:
                print("John: Please enter a valid number.")
    else:
        #user has no plans
        goal = get_fitness_goal(username)
        if goal:
            print(f"John: Based on your goal, I recommend the {goal} plan.")
        plan_selected = offer_fitness_plans()
        confirm_plan_purchase(username, plan_selected)
        #show the selected plan if requested
        user_plans = user_has_plan(username)
        if user_plans and plan_selected in user_plans:
            files = get_plan_files(plan_selected)
            if files and plan_type in files:
                display_csv_plan(files[plan_type])
            else:
                print(f"John: Sorry, I don't have a {plan_type} plan for {plan_selected}.")
        else:
            print("John: Let me know if you change your mind!")


def confirm_plan_purchase(username, plan_name):
    plan_prices = {
        "Powerlifting": 49.99,
        "Muscle Building": 39.99,
        "Fat Loss": 29.99
    }

    user_plans = user_has_plan(username)
    if user_plans:
        #user already has at least one plan
        print(f"John: You already have the following plan(s): {', '.join(user_plans)}.")
        #ask if they want another one
        confirmation = input(f"John: You already have a plan. Would you like to purchase the '{plan_name}' plan as well? (yes/no): ").strip().lower()
        if confirmation != 'yes':
            print("John: Purchase canceled. You can continue with your current plan(s).")
            return


    price = plan_prices.get(plan_name, 19.99)
    print(f"John: The {plan_name} plan costs ${price}. Would you like to confirm your purchase?")
    confirmation = input("Type 'yes' to confirm, or 'no' to cancel: ").strip().lower()

    if confirmation == 'yes':
        record_purchase(username, plan_name)
    else:
        print("John: Purchase canceled. Let me know if you change your mind.")

def show_plan_details(plan_name):
    plan_files = {
        "Powerlifting": {
            "diet": "powerlifting_diet.csv",
            "workout": "powerlifting_split_paid.csv"
        },
        "Muscle Building": {
            "diet": "bodybuilding_diet.csv",
            "workout": "bodybuilding_workoutplan.csv"
        },
        "Fat Loss": {
            "diet": "fat_loss_dietplan.csv",
            "workout": "fat_loss_workout.csv"
        }
    }

    if plan_name not in plan_files:
        print(f"John: Sorry, I don't have details for the '{plan_name}' plan.")
        return

    choice = input("John: Would you like to see the 'diet' or 'workout' plan? ").strip().lower()
    if choice in plan_files[plan_name]:
        display_csv_plan(plan_files[plan_name][choice])
    else:
        print("John: Invalid choice. Please specify 'diet' or 'workout'.")



def chat():
    sentiment_keywords = ["happy", "sad", "great", "terrible", "good", "bad", "blessed", "thankful"]
    transaction_phrases = ["i would like to make a purchase", "i would like to purchase a plan", "i want to start my fitness journey", "help me start my fitness journey", "what can i do to start my fitness journey", "begin my fitness journey", 'start my fitness journey', 'i would like to purchase a fitness plan', 'i would like to purchase a diet plan', 'i want to purchase a diet plan', 'i want to purchase a fitness plan', 'create a workout plan for me']
    user_name = confirm_name()
    print("Hello, type 'exit' or 'bye' to end the conversation.")
    while True:
        user_input = input(f"{user_name}: ")
        if user_input.lower() in ['exit', 'bye', 'goodbye']:
            response = random.choice(responses['farewell'])
            print(f"John: {response}")
            break

        if is_sentiment_related(user_input, sentiment_keywords):
            mood_response(user_name, user_input, sentiment_clf, count_vect, tf_idf_transform_sentiment)
            continue


        phrases = ["maintenance calories", 'maintenance calorie', 'how many calories should i eat', 'what are my maintenance calories', 'calculate my maintenance calories', 'what is my maintenance calories', 'calculate my maintenance calories']
        if user_input.lower() in phrases:
            maintenance = ensure_maintenance_calories(user_name)
            if maintenance is not None:
                print(f"John: Your maintenance calories are {maintenance} kcal per day.")
            continue
        #preprocess input for main classifier
        processed_input = preprocess_input(user_input)
        input_vector_main = vectorizer.transform([processed_input])
        predicted_main_category = svm_classifier.predict(input_vector_main)[0]


        if "is my name" in user_input.lower():
            print(f"Your name is {user_name}")
            continue

        if any(phrase in user_input.lower() for phrase in ["change my name", "i want to change my name", "call me"]):
            user_name = change_user_name(user_name)
            continue
        if "show me my plan" in user_input.lower() or "show my plan" in user_input.lower():

            plan_type = input("John: Which plan would you like to see, 'diet' or 'fitness'?: ").strip().lower()

            if plan_type == "diet":
                handle_plan_request(user_name, "diet")
            elif plan_type == "fitness":
                handle_plan_request(user_name, "workout")
            else:
                print("John: I'm sorry, I didn't understand that. Please specify 'diet' or 'fitness'.")
            continue
        if "show me my diet plan" in user_input.lower() or "show my diet plan" in user_input.lower():
            handle_plan_request(user_name, "diet")
            continue
        elif "show me my workout plan" in user_input.lower() or "show my workout plan" in user_input.lower() or "show my fitness plan" in user_input.lower() or "show me my fitness plan" in user_input.lower():
            handle_plan_request(user_name, "workout")
            continue

        if any(phrase in user_input.lower() for phrase in transaction_phrases):
            start_fitness(user_name)
            continue
        if "log my workout" in user_input:
            log_workout(user_name)
            continue
        elif "log my diet" in user_input:
            log_diet(user_name)
            continue
        elif "log my weight" in user_input:
            log_weight(user_name)
            continue
        elif "show my progress" in user_input:
            display_progress(user_name)
            continue

        if "delete workout" in user_input:
            delete_workout_entry(user_name)
            continue

        if "delete diet" in user_input or "delete meal" in user_input:
            delete_diet_entry(user_name)
            continue

        if predicted_main_category == 'small_talk':
            input_vector_small_talk = vectorize_small_talk.transform([processed_input])
            predicted_category = small_talk_classifier.predict(input_vector_small_talk)[0]



            best_match = find_best_response(user_input, user_name, category=predicted_category)
            if best_match is not None:
                print(f"John: {best_match}")
            else:
                #if no good match found go back to the predefined responses dictionary
                if predicted_category in responses:
                    response = random.choice(responses[predicted_category])
                    print(f"John: {response}")
                else:
                    print("John: I'm sorry, I didn't understand that. Could you please rephrase?")

        elif predicted_main_category == 'qanda':
            #qanda handling
            best_answer = find_best_answer(user_input, qanda_data, question_vectors, vectorizer_questions)
            if best_answer:
                print(f"John: {best_answer}")
            else:
                print("John: I'm sorry, I couldn't find an answer to that.")
        elif predicted_main_category == 'fitness_query':
            #fitness handling
            input_vector_fitness = vectorizer_fitness.transform([processed_input])
            predicted_fitness_category = classifier_fitness.predict(input_vector_fitness)[0]
            # print(f"Fitness Category Prediction: {predicted_fitness_category}")

            category_data = fitness_data_score.get(predicted_fitness_category)
            if category_data:
                vectorizer_category = category_data['vectorizer']
                question_vectors_category = category_data['question_vectors']
                questions = category_data['questions']
                answers = category_data['answers']

                user_input_vector_cs = vectorizer_category.transform([processed_input])
                similarities = cosine_similarity(user_input_vector_cs, question_vectors_category)
                max_similarity = similarities.max()
                # print(f"Similarities: {similarities}")
                # print(f"Max Similarity: {max_similarity}")

                if max_similarity < 0.1:
                    print("John: I'm sorry, I couldn't find a relevant answer. Could you try rephrasing?")
                else:
                    best_match_idx = similarities.argmax()
                    best_answer = answers.iloc[best_match_idx]
                    # print(f"Matched Question: {questions.iloc[best_match_idx]}")
                    print(f"John: {best_answer}")
            else:
                print("John: Sorry, I don't have information on that fitness category.")
        else:
            print("John: I'm sorry, I couldn't understand that. Could you please rephrase?")
# train_sentiment_model()
initialize_db()
chat()

