# Fitness Chatbot ("John")

## Overview

"John" is a command-line fitness chatbot that helps users with goals like fat loss, muscle gain, and powerlifting. It combines intent classification, small-talk handling, Q&A retrieval, sentiment analysis, and user profile tracking to provide personalized recommendations, log workouts/diet/weight, and manage fitness plans and purchases.

## Features

- Top-level intent classification (small talk / Q&A / fitness queries)
- Sub-classifiers for:
  - Small talk with canned responses
  - Fitness-specific queries (fat loss, muscle building, powerlifting)
- Retrieval of best match answers from Q&A datasets
- Sentiment-aware interaction to adapt tone (e.g., offering tips or jokes)
- User profile and state stored in SQLite (weight, height, goals, purchases, logs)
- Logging: workouts, diet, weight
- Maintenance calorie calculation and goal setting
- Plan selection and purchase tracking
- Interactive conversational flow via terminal

## Requirements

- Python 3.9+
- The following Python packages (example `requirements.txt` can include):
  ```text
  pandas
  scikit-learn
  nltk

## Running

Start the chatbot with:
- python main3.py

This will:
- Initialize or open user_data.db
- Train/Load sentiment and intent models as required
- Enter the interactive chat loop in your terminal 

## Internal Architecture 

- Intent classification: Combines multiple dataset-trained classifiers:

- Main classifier distinguishes between small_talk, qanda, and fitness_query.

- Sub-classifiers handle finer granularity (e.g., fitness goals or small talk categories).

- Sentiment model: Logistic regression over TF-IDF features to decide positive/negative mood.

- Q&A retrieval: Cosine similarity on vectorized questions to find best match.

- User state: Stored in SQLite (user_data.db), tracking user info, goals, purchased plans, logs.

- Logging modules: Separate for workouts, diet, and weight history.

- Conversation flow: main3.py orchestrates the dialogue, handling user inputs, managing state, and routing to helpers.

## Assumptions

- User interacts via a single terminal session (no concurrency).

- All required CSV files are present, valid, and have expected columns.

- User input is reasonably formatted (e.g., gender is "male" or "female").

- Plan pricing and names are hardcoded (Powerlifting, Muscle Building, Fat Loss).

- The system is operating on a machine with sufficient permissions to write user_data.db.


## Troubleshooting

- Missing NLTK data: If you see errors regarding tokenization or wordnet, rerun the NLTK download commands from setup.

- Database errors: Delete user_data.db to reset (use with caution, this will lose all stored users/plans).

- Invalid CSV formats: Ensure no missing required columns and correct encoding; for problematic CSVs, consider opening in a spreadsheet editor.

- Unexpected predictions: The classifiers are simple (SVM, TF-IDF) and might need re-training or data cleaning for edge-case inputs.


## Extending / Next Steps
- Replace static classifiers with fine-tuned transformer models for better intent/sentiment understanding.

- Expose via REST API instead of CLI for integration with web/mobile clients.

- Add authentication per user instead of name prompts.

- Support persistence of session across devices or multi-user environments.

- Improve plan management (e.g., expiration, upgrades, discounts).

