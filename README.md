# Student Performance Predictor (Pure Python AIML Project)

**Description**: A simple AI/ML project written in pure Python (no external libraries required).
It includes synthetic data generation, a rule-based predictor, a simple linear regression trained
with gradient descent, CSV I/O, model save/load (JSON), and an interactive CLI.

## Files
- `student_perf_project.py` — main project script (pure Python)
- `README.md` — this file
- `sample_data.csv` — example CSV you can create to test CSV prediction
- `model.json` — saved model file (after you save one)

## How to run
1. Save `student_perf_project.py` to your machine.
2. Run: `python student_perf_project.py`
3. Choose option 1 to generate data & train, or 2 to load a saved model, or 3 to predict manually.

## CSV format for batch predictions
CSV rows should contain: `study_hours,attendance,sleep_hours,previous_marks` (one student per row).
No header required but headers are fine — non-numeric rows are skipped.

## Example usage
- Train a linear model and save it.
- Create `test_students.csv` with rows like:
```
6,85,7,70
4,75,6,55
```
- Use option 4 to load CSV and output predictions to a CSV file.

## Notes
- This project is educational: models are simple and intended for learning how ML concepts work.
- You can improve the project by adding a GUI (tkinter) or implementing more advanced models.
