import git
import random
import os
from datetime import datetime, timedelta
import textwrap

# Configuration
REPO_PATH = '.'
TARGET_FILES = [
    'data_preparation.py',
    'evaluate_mamo.py',
    'evaluation.py',
    'maml.py',
    'mamo.py'
]
START_DATE = datetime.now() - timedelta(days=150)  # 5 months ago
END_DATE = datetime.now()

# Initialize repo
repo = git.Repo(REPO_PATH)

def generate_python_code(file_name):
    """Generate plausible Python code changes based on file purpose"""
    base_content = ""
    
    if file_name == 'data_preparation.py':
        functions = [
            "def load_data(path):",
            "def preprocess_images(images):",
            "def split_dataset(data, test_size=0.2):",
            "def augment_data(samples):"
        ]
        base_content = textwrap.dedent(f'''\
            import numpy as np
            from sklearn.model_selection import train_test_split
            
            {random.choice(functions)}
                """{random.choice(['Add docstring', 'Update loading logic', 'Fix path handling'])}"""
                {random.choice([
                    'return np.load(path)',
                    'images = [img/255.0 for img in images]',
                    'return train_test_split(data, test_size=test_size)',
                    'return [np.fliplr(img) for img in samples]'
                ])}
            ''')
    
    elif file_name == 'evaluate_mamo.py':
        base_content = textwrap.dedent(f'''\
            from sklearn.metrics import accuracy_score, f1_score
            
            def evaluate_model(model, X_test, y_test):
                """{random.choice(['Update evaluation metrics', 'Add logging', 'Fix threshold'])}"""
                preds = model.predict(X_test)
                return {random.choice([
                    'accuracy_score(y_test, preds)',
                    '{"accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds)}',
                    'np.mean(preds == y_test)'
                ])}
            ''')
    
    # Similar blocks for other files...
    
    return base_content

def make_commit(file_name, commit_date, commit_num, total_commits):
    """Make a commit with realistic changes"""
    action = random.choice([
        'change', 'fix', 'refactor', 'trying something', 'testing'
    ])
    
    # Generate or modify file
    if not os.path.exists(file_name) or random.random() < 0.3:
        with open(file_name, 'w') as f:
            f.write(generate_python_code(file_name))
    else:
        with open(file_name, 'a') as f:
            f.write(f"\n# {action}: {commit_date.date()} - {commit_num}/{total_commits}\n")
            f.write(generate_python_code(file_name))
    
    repo.index.add([file_name])
    repo.index.commit(
        f"{action} {file_name}",
        author_date=commit_date.strftime("%Y-%m-%d %H:%M:%S"),
        commit_date=commit_date.strftime("%Y-%m-%d %H:%M:%S")
    )

# Main execution
current_date = START_DATE
while current_date < END_DATE:
    # Only weekdays (Mon-Fri)
    if current_date.weekday() < 5:
        # 1-3 commits per day
        commits_per_day = random.randint(1, 3)
        for i in range(commits_per_day):
            file_to_modify = random.choice(TARGET_FILES)
            commit_time = current_date.replace(
                hour=random.randint(9, 17),
                minute=random.randint(0, 59)
            )
            make_commit(file_to_modify, commit_time, i+1, commits_per_day)
    
    # Move to next day
    current_date += timedelta(days=1)

print("Fake commit history generated successfully!")