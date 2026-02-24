# resume_extractor

## 💻 How to use with VS Code
If you are using **Visual Studio Code**, follow these steps to get synced:

1. **Clone the Repo:** 
   - Open VS Code.
   - Press `Ctrl + Shift + P` and type `Git: Clone`.
   - Paste this repository's URL.
2. **Open Terminal:**
   - Go to `Terminal > New Terminal` in the top menu.
3. **Install Dependencies:**
   - In the terminal at the bottom, type: `pip install -r requirements.txt`
4. **Select Python Interpreter:**
   - If you see yellow squiggly lines under your imports, press `Ctrl + Shift + P`, type `Python: Select Interpreter`, and pick your installed Python version.
5. **Running the Script:**
   - Type `python parser.py your_file.pdf` directly in the VS Code terminal.



## 🖥️ Manual Setup (No VS Code)
If you prefer using your computer's native terminal:

1. **Download:** Click the green `<> Code` button above and select **Download ZIP**. Extract it to your Desktop.
2. **Open Terminal:**
   - **Windows:** Search for `cmd` or `PowerShell`.
   - **Mac/Linux:** Search for `Terminal`.
3. **Navigate to the folder:**
   Type `cd` followed by a space, then drag the folder into the terminal window and hit Enter.
   *(Example: `cd Desktop/resume-extractor-main`)*
4. **Install Engines:**
   Run `pip install -r requirements.txt`
5. **Run the Parser:**
   `python parser.py your_resume.pdf`



Google Colab (Cloud / No Install)
Open Google Colab.
Upload parser.py and requirements.txt.
In a code cell, run: !pip install -r requirements.txt
To parse, run: !python parser.py your_uploaded_resume.pdf

