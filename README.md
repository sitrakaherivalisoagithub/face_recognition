# Face Recognition Attendance System

This is a web-based application that uses face recognition for attendance tracking. The application is built with Flask and utilizes the `insightface` library for deep learning-based face analysis.

## Features

*   **User Registration:** Register new users by providing a name and a clear image of their face. The system extracts facial embeddings and stores them.
*   **Identity Verification:** Check a user's identity by uploading an image. The system compares the detected face against the registered users.
*   **Real-time Recognition:** A live video stream from a webcam performs real-time face detection and recognition, identifying known individuals.

## Project Structure

```
├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
├── templates/
│   ├── index.html      # Main page for registration and checking
│   └── realtime.html   # Page for real-time camera feed
├── static/
│   ├── images/         # Stores registered user images
│   └── ...
├── utils/
│   └── face_utils.py   # Utility functions for face model and comparison
└── employees.json      # Database storing names and facial embeddings
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd FaceAttendanceProject
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage

*   **To Register a Person:**
    1.  On the homepage, use the "Register" form.
    2.  Enter the person's name in the "Name" field.
    3.  Click "Choose File" to select an image of the person.
    4.  Click "Register".

*   **To Check a Person's Identity:**
    1.  On the homepage, use the "Check" form.
    2.  Click "Choose File" to select an image of the person to verify.
    3.  Click "Check". The result will show if the person is recognized.

*   **For Real-time Recognition:**
    1.  Navigate to the `/realtime` endpoint in your browser (`http://127.0.0.1:5000/realtime`).
    2.  The browser will request camera access. Allow it to start the video stream.
    3.  The application will draw boxes around detected faces and display the names of recognized individuals.
