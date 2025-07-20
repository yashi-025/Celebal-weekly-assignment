# ⭐ Celebal Weekly Assignment

Welcome to the **Celebal Weekly Assignments** repository. 
This repository will be updated weekly with tasks and solutions provided as part of the Celebal Internship Program.

## 📅 Week 7: Diamond Price Predictor

### 🎯 Objective
This web application predicts diamond prices using a trained regression model. Built with Streamlit, it allows users to input data, get instant predictions, and view related visualizations.

### Features
- User-friendly sidebar input
- Real-time predictions
- Data visualizations
- Clean and modular code

### Tech Stack
- Python
- Scikit-learn
- Streamlit
- Pandas, Seaborn, Matplotlib

### Setup Instructions
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`

### Directory Structure
week 7
├── app.py                         
├── model/
|   └── trained_model.py
│   └── trained_model.pkl          
├── utils/
│   └── preprocess.py              
└── dataset/
  └── Diamonds Prices2022.csv            

## 📅 Week 2: OOPs in Python

### ✅ Assignment Description

The goal of this assignment is to implement a Singly Linked List in Python using Object-Oriented Programming (OOP) principles. This is part of the weekly assignment series and follows the Week 1 submission already uploaded to this repository.

### Features Implemented

1) A Node class to represent each element in the linked list.
  
2) A LinkedList class to manage nodes with the following methods:
  - add_node(data) – Add a node with the given data to the end of the list.
  - print_list() – Display all nodes in the list.
  - delete_nth_node(n) – Delete the node at the nth position (1-based index).

3) Exception Handling for edge cases:
  - Deleting from an empty list.
  - Deleting a node with an out-of-range index.


## 📅 Week 1: Python Basics

### ✅ Assignment Description

In Week 1, the focus is on practicing basic Python programming concepts such as loops, conditionals, and pattern printing.

### 🧠 Problem Statement

Write Python programs to generate the following patterns using the `*` character:

    1. **Lower Triangular Pattern**
    2. **Upper Triangular Pattern**
    3. **Pyramid Pattern**

