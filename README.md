# Navigate to your project directory
cd /path/to/your/project

# Step 1: Create a virtual environment
python -m venv venv

# Step 2: Activate the virtual environment
# On Windows:
venv\Scripts\activate

# Step 3: Install required dependencies
# pip install streamlit numpy joblib
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run app.py
