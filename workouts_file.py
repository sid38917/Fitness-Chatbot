import sqlite3


#allows user to log their workouts into the user_goals table
def log_workout(username):
    workout = input("John: What workout did you complete today? ").strip()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT workout_logs FROM user_goals WHERE username = ?', (username,))
    result = cursor.fetchone()
    if result and result[0]:
        existing_logs = result[0].split(',')
    else:
        existing_logs = []

    existing_logs.append(workout)
    updated_logs = ','.join(existing_logs)
    cursor.execute('UPDATE user_goals SET workout_logs = ? WHERE username = ?', (updated_logs, username))
    conn.commit()
    conn.close()
    print("John: Your workout has been logged!")

#allows user to log their diet into the user_goals table
def log_diet(username):
    meal = input("John: What meal did you eat today? ").strip()
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT dietary_logs FROM user_goals WHERE username = ?', (username,))
    result = cursor.fetchone()
    if result and result[0]:
        existing_logs = result[0].split(',')
    else:
        existing_logs = []

    existing_logs.append(meal)
    updated_logs = ','.join(existing_logs)
    cursor.execute('UPDATE user_goals SET dietary_logs = ? WHERE username = ?', (updated_logs, username))
    conn.commit()
    conn.close()
    print("John: Your dietary log has been updated!")

#allows users to log their weights, with the ability to consistently update it
def log_weight(username):
    try:
        weight = float(input("John: Please enter your current weight (kg): ").strip())
        conn = sqlite3.connect('user_data.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE user_goals SET last_weight = ? WHERE username = ?', (weight, username))
        conn.commit()
        conn.close()
        print(f"John: Your weight has been updated to {weight} kg.")
    except ValueError:
        print("John: Please enter a valid number.")
        log_weight(username)


#displays users progress, obtained from the user_goals table
def display_progress(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT last_weight, workout_logs, dietary_logs FROM user_goals WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        last_weight, workout_logs, dietary_logs = result
        print(f"John: Here is your progress summary:")
        print(f"- Last recorded weight: {last_weight} kg" if last_weight else "- No weight logged yet.")
        print(f"- Workouts logged: {workout_logs if workout_logs else 'None yet.'}")
        print(f"- Dietary logs: {dietary_logs if dietary_logs else 'None yet.'}")
    else:
        print("John: No progress data found. Let's get started!")

#allows user to delete workout entries from the database, in order to potentially add new ones
def delete_workout_entry(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT workout_logs FROM user_goals WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        workouts = result[0].split(',')
    else:
        print("John: You don't have any workout logs to delete.")
        return

    #display workouts
    print("John: Here are your logged workouts:")
    for i, w in enumerate(workouts, start=1):
        print(f"{i}. {w}")

    choice = input("John: Enter the number of the workout you want to delete (or 'cancel' to exit): ").strip()
    if choice.lower() == 'cancel':
        print("John: Deletion canceled.")
        return

    try:
        index = int(choice) - 1
        if 0 <= index < len(workouts):
            deleted = workouts.pop(index)
            updated_logs = ','.join(workouts)
            conn = sqlite3.connect('user_data.db')
            cursor = conn.cursor()
            cursor.execute('UPDATE user_goals SET workout_logs = ? WHERE username = ?', (updated_logs, username))
            conn.commit()
            conn.close()
            print(f"John: The workout '{deleted}' has been deleted.")
        else:
            print("John: Invalid selection. Please try again.")
    except ValueError:
        print("John: Please enter a valid number.")

#deletes diets entries, allowing the user to consistenly update their diets daily to track progress
def delete_diet_entry(username):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT dietary_logs FROM user_goals WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        meals = result[0].split(',')
    else:
        print("John: You don't have any dietary logs to delete.")
        return

    #display meals
    print("John: Here are your logged meals:")
    for i, meal in enumerate(meals, start=1):
        print(f"{i}. {meal}")

    choice = input("John: Enter the number of the meal you want to delete (or 'cancel' to exit): ").strip()
    if choice.lower() == 'cancel':
        print("John: Deletion canceled.")
        return

    try:
        index = int(choice) - 1
        if 0 <= index < len(meals):
            deleted = meals.pop(index)
            updated_logs = ','.join(meals)
            conn = sqlite3.connect('user_data.db')
            cursor = conn.cursor()
            cursor.execute('UPDATE user_goals SET dietary_logs = ? WHERE username = ?', (updated_logs, username))
            conn.commit()
            conn.close()
            print(f"John: The meal '{deleted}' has been deleted.")
        else:
            print("John: Invalid selection. Please try again.")
    except ValueError:
        print("John: Please enter a valid number.")
