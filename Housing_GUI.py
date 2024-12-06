from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time


class IMS:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1350x700+0+0")
        self.root.title("Bostin Housing prediction project")
        self.root.configure(bg="grey")

        # Title Label with an icon
        self.icon_title = None 
        title = Label(self.root, text="Machine Learning Diploma", compound=LEFT,
                      font=("times new roman", 30, "bold"), bg="#010c48", fg="white", anchor="w", padx=20)
        title.place(x=0, y=0, relwidth=1, height=70)

        # Logout Button
        btn_logout = Button(self.root, text="Logout", font=("times new roman", 15, "bold"), bg="yellow",
                            cursor="hand2", command=self.logout)
        btn_logout.place(x=1150, y=10, height=50, width=150)

        # Clock Label
        self.lbl_clock = Label(self.root, text="Welcome to Machine Learning Diploma \t\t Date: DD-MM-YYYY \t\t Time: HH:MM:SS",
                               font=("times new roman", 15), bg="#4d636d", fg="white")
        self.lbl_clock.place(x=0, y=70, relwidth=1, height=30)
        self.update_clock()


        # Machine Learning Section
        groupbox_ml = LabelFrame(self.root, text="Machine Learning", font=("Arial", 16), bg="#4d636d", fg="white", padx=10, pady=10)
        groupbox_ml.place(x=10, y=110, width=200, height=550)

        # Add buttons inside the Machine Learning section
        Button(groupbox_ml, text="Boston Housing Project", font=("times new roman", 13, "bold"),
               bg="#ffffff", fg="black", relief="solid", cursor="hand2", command=self.open_boston_project).place(x=10, y=10, width=180, height=70)

    def open_boston_project(self):
        # New Window for the Boston Housing Project
        project_window = Toplevel(self.root)
        project_window.title("Boston Housing Price Prediction")
        project_window.geometry("800x600")

        # Title for the project
        title = Label(project_window, text="Boston Housing Price Prediction", font=("times new roman", 20, "bold"), bg="#010c48", fg="white")
        title.pack(side=TOP, fill=X)

        # Load Dataset Button
        Button(project_window, text="Load Dataset", font=("times new roman", 15), command=self.load_dataset).pack(pady=20)

        # Train Model Button
        self.train_btn = Button(project_window, text="Train Model", font=("times new roman", 15), command=self.train_model, state=DISABLED)
        self.train_btn.pack(pady=20)

        # Predict Button
        self.predict_btn = Button(project_window, text="Make Prediction", font=("times new roman", 15), command=self.make_prediction, state=DISABLED)
        self.predict_btn.pack(pady=20)

        # Output Label
        self.output_label = Label(project_window, text="Output will be displayed here.", font=("times new roman", 15), bg="white", fg="black", wraplength=700)
        self.output_label.pack(pady=20)

        # Data and model placeholders
        self.dataset = None
        self.model = None

    def load_dataset(self):
        # Open file dialog to load dataset
        file_path = filedialog.askopenfilename(filetypes=[("housing", "*.csv")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
                self.train_btn.config(state=NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_model(self):
        if self.dataset is not None:
            # Train a simple linear regression model
            X = self.dataset.iloc[:, :-1]
            y = self.dataset.iloc[:, -1] 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            self.output_label.config(text=f"Model trained! MSE: {mse:.2f}")
            self.predict_btn.config(state=NORMAL)
            
            

    def make_prediction(self):
        if self.model is not None:
            try:
                # Input values for prediction
                input_window = Toplevel(self.root)
                input_window.title("Input Features")
                input_window.geometry("400x400")

                inputs = []
                for i in range(len(self.dataset.columns) - 1):  # Exclude the target column
                    label = Label(input_window, text=f"Feature {i+1}:")
                    label.pack()
                    entry = Entry(input_window)
                    entry.pack()
                    inputs.append(entry)

                def predict():
                    try:
                        features = [float(entry.get()) for entry in inputs]
                        prediction = self.model.predict([features])
                        messagebox.showinfo("Prediction", f"Predicted Price: {prediction[0]:.2f}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Prediction failed: {e}")

                predict_btn = Button(input_window, text="Predict", command=predict)
                predict_btn.pack(pady=10)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to make prediction: {e}")

    def update_clock(self):
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")
        self.lbl_clock.config(text=f"Welcome to Machine Learning Diploma \t\t Date: {current_time.split()[0]} \t\t Time: {current_time.split()[1]}")
        self.root.after(1000, self.update_clock)

    def logout(self):
        response = messagebox.askyesno("Logout", "Are you sure you want to logout?")
        if response:
            self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    obj = IMS(root)
    root.mainloop()
