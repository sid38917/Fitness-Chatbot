import sqlite3
import csv

from nltk.downloader import update

#intializes the database, creates three tables - users, purchases, and user_goals

def initialize_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        weight REAL,
        height REAL,
        age INTEGER,
        gender TEXT,
        maintenance_calories REAL
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS purchases (
        username TEXT,
        plan_name TEXT,
        FOREIGN KEY(username) REFERENCES users(username)
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS plans (
        plan_name TEXT PRIMARY KEY,
        content TEXT
    )''')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_goals (
                username TEXT PRIMARY KEY,
                fitness_goal TEXT,
                purchased_plan TEXT,
                last_weight REAL,
                workout_logs TEXT,
                dietary_logs TEXT,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        ''')

    try:
        cursor.execute('ALTER TABLE users ADD COLUMN maintenance_calories REAL;')
    except sqlite3.OperationalError:
        pass
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
    conn.commit()
    conn.close()

def reset_fitness_details(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET weight = NULL,
            height = NULL,
            age = NULL,
            gender = NULL,
            maintenance_calories = NULL
        WHERE username = ?
    ''', (username,))
    conn.commit()
    conn.close()
    print("John: Your fitness journey details have been reset.")

#obtains the fitness goals of the users, either Powerlifting, Muscle Building or Fat Loss
def get_fitness_goal(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT fitness_goal FROM user_goals WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def create_user(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    existing = cursor.fetchone()

    if existing:
        #if username exists
        choice = input(f"John: The username '{username}' already exists. Would you like to (r)etry with another name or (o)verwrite the existing user? ").strip().lower()
        if choice == 'o':
            #delete old user data and create new user
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.execute("DELETE FROM user_goals WHERE username = ?", (username,))
            conn.execute("DELETE FROM purchases WHERE username = ?", (username,))
            conn.commit()
            print(f"John: The old user '{username}' has been removed. Creating a new user with the same name.")
            #insert new user
            cursor.execute(
                "INSERT INTO users (username, weight, height, age, gender) VALUES (?, NULL, NULL, NULL, NULL)",
                (username,))
            conn.commit()
            conn.close()
            print(f"John: Nice to meet you {username}")
            return username
        else:
            #retry with another name
            conn.close()
            new_name = input("John: Please enter a new username: ").strip()
            if not new_name:
                print("John: Invalid name. Please try again.")
                return create_user(username)
            return create_user(new_name)
    else:
        #create a new user
        cursor.execute(
            "INSERT INTO users (username, weight, height, age, gender) VALUES (?, NULL, NULL, NULL, NULL)",
            (username,))
        conn.commit()
        conn.close()
        print(f"John: Nice to meet you {username}")
        return username

def record_purchase(username, plan_name):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    #checks if user already purchased a plan
    cursor.execute('SELECT plan_name FROM purchases WHERE username = ? AND plan_name = ?', (username, plan_name))
    existing_plan = cursor.fetchone()
    if existing_plan:
        print(f"John: You already have the '{plan_name}' plan.")
        conn.close()
        return
    #inserts new purchase into purchases table
    cursor.execute('''
        INSERT INTO purchases (username, plan_name)
        VALUES (?, ?)
    ''', (username, plan_name))
    conn.commit()
    conn.close()
    print(f"John: The '{plan_name}' plan has been successfully purchased.")

#stores maintenance calories into the Users table
def store_maintenance_calories(username, maintenance_calories):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET maintenance_calories = ? WHERE username = ?', (maintenance_calories, username))
    conn.commit()
    conn.close()

#updates the user data, especially seen when the user requests to change name
def update_user(username, weight, height, age, gender):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET weight = ?, height = ?, age = ?, gender = ?
        WHERE username = ?
    ''', (weight, height, age, gender, username))
    conn.commit()
    conn.close()

#collects user details to idenitfy maintenance calories, and suggest plans
def collect_details():
    while True:
        try:
            weight = float(input("Please enter your weight in kilograms (kg): ").strip())
            height = float(input("Please enter your height in centimeters (cm): ").strip())
            age = int(input("Please enter your age in years: ").strip())
            gender = input("Please enter your gender (male or female): ").strip().lower()
            if gender not in ['male', 'female']:
                print("John: Invalid gender. Please enter 'male' or 'female'.")
                continue
            break
        except ValueError:
            print("John: invalid input, please enter the right values")

    return weight, height, age, gender


def change_user_name(old_username):
    import sqlite3

    new_name = input("John: What would you like to change your name to? ").strip()
    if not new_name:
        print("John: Invalid name. Please try again later.")
        return old_username

    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    try:
        #check if new_name exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (new_name,))
        existing = cursor.fetchone()

        if existing:
            choice = input(f"Assistant: The name '{new_name}' already exists. Would you like to (r)etry with another name or (o)verwrite this user? ").strip().lower()
            if choice == 'o':
                #delete all references to the existing `new_name`
                cursor.execute("DELETE FROM users WHERE username = ?", (new_name,))
                cursor.execute("DELETE FROM user_goals WHERE username = ?", (new_name,))
                cursor.execute("DELETE FROM purchases WHERE username = ?", (new_name,))
                conn.commit()


                cursor.execute("SELECT username FROM users WHERE username = ?", (new_name,))
                if cursor.fetchone():
                    raise Exception(f"Failed to fully delete old records for '{new_name}'.")


                cursor.execute("UPDATE users SET username = ? WHERE username = ?", (new_name, old_username))
                if cursor.rowcount == 0:
                    raise Exception(f"Failed to update username from '{old_username}' to '{new_name}'.")

                #update related tables
                cursor.execute("UPDATE user_goals SET username = ? WHERE username = ?", (new_name, old_username))
                cursor.execute("UPDATE purchases SET username = ? WHERE username = ?", (new_name, old_username))
                conn.commit()

                print(f"John: Your name has been successfully updated to {new_name}!")
                reset_fitness_details(new_name)
                set_last_active_user(new_name)
                return new_name
            elif choice == 'r':
                print("John: Let's try again.")
                return change_user_name(old_username)
            else:
                print("John: Invalid option. Cancelling the operation.")
                return old_username
        else:
            # update if no conflict
            cursor.execute("UPDATE users SET username = ? WHERE username = ?", (new_name, old_username))
            if cursor.rowcount == 0:
                raise Exception(f"Failed to update username from '{old_username}' to '{new_name}'.")

            #delete old user's plans before updating tables
            cursor.execute("DELETE FROM purchases WHERE username = ?", (old_username,))
            cursor.execute("DELETE FROM user_goals WHERE username = ?", (old_username,))

            #update tables
            cursor.execute("UPDATE user_goals SET username = ? WHERE username = ?", (new_name, old_username))
            cursor.execute("UPDATE purchases SET username = ? WHERE username = ?", (new_name, old_username))
            conn.commit()

            print(f"John: Your name has been successfully updated to {new_name}!")
            reset_fitness_details(new_name)
            set_last_active_user(new_name)
            return new_name

    except sqlite3.Error as db_error:
        print(f"John: Database error occurred: {db_error}")
    except Exception as e:
        print(f"John: An error occurred: {e}")
    finally:
        conn.close()

    return old_username

#gets the last active user
def get_last_active_user():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM app_state WHERE key = 'last_user'")
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return None
#
#sets the new last active user so in future conversations the chatbot continues conversing with this user
def set_last_active_user(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES ('last_user', ?)", (username,))
    conn.commit()
    conn.close()
#
#greeting, confirms if talking to same user
def confirm_name():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users')
    stored_users = [row[0] for row in cursor.fetchall()]
    conn.close()

    if stored_users:
        stored_user = stored_users[0]
        confirmation = input(f"Am I still talking to {stored_user}? (yes/no): ").strip().lower()
        if confirmation == 'yes':
            return stored_user
        else:
            username = input("What's your name: ").strip()
            create_user(username)
            print(f"Nice to meet you {username}")
            return username
    else:
        #if no users
        username = input("What's your name: ").strip()
        create_user(username)
        print(f"Nice to meet you {username}")
        return username

#checks if user has a plan, in case user requests to see plans without making a purchase
def user_has_plan(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT plan_name FROM purchases WHERE username = ?", (username,))
    result = cursor.fetchall()
    conn.close()
    #returns a list of purchased plans
    return [row[0] for row in result] if result else []

#gets maintenance calories
def get_maintenance_calories(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT maintenance_calories FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return[0]
    return None


