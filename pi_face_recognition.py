from imutils.video import VideoStream, FPS
import dlib
import face_recognition
import argparse
import imutils
import pickle
import cv2
import time
import mysql.connector
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor

employee_id_counter = 0
recognized_persons = {}  # Initialize recognized persons dictionary
cooldown_period = 20  # 20 seconds cooldown period

# Replace these values with your database credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '19092003',
    'database': 'face_biometric',
}

# Connect to MySQL using connection pooling
conn_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **db_config)

# Explicitly specify the database to use
conn = conn_pool.get_connection()
cursor = conn.cursor()
cursor.execute("USE face_biometric_attendance;")

# Load the face recognition model from dlib
#face_detector = dlib.get_frontal_face_detector()
#face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def process_frame(frame):
    global recognized_persons, employee_id_counter

    orig = frame.copy()
    frame = imutils.resize(frame, width=500)

    # Find all face locations in the current frame using dlib
    face_locations = face_recognition.face_locations(frame)

    # Compute face encodings for all detected faces
    encodings = face_recognition.face_encodings(frame, face_locations)

    names = []

    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = None

        if True in matches:
            # Find the index of the first matched face
            matchedIdx = matches.index(True)
            #print("Matches:", matches)

            name = data["names"][matchedIdx]

            current_time = time.time()
            if name not in recognized_persons or (current_time - recognized_persons.get(name, 0)) >= cooldown_period:
                print("Recognized Name:", name)
                recognized_persons[name] = current_time

                employee_id_counter += 1

                employee_id = employee_id_counter
                insert_attendance_query = "INSERT INTO attendance (employee_id, name, date, time) VALUES (%s, %s, CURDATE(), CURTIME())"
                cursor.execute(insert_attendance_query, (employee_id, name))
                conn.commit()

                print("Employee ID:", employee_id)
                print("Date:", date.today())
                print("Time:", datetime.now().strftime("%H:%M:%S"))
    #cv2.putText(frame, f"Recognized: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    #return frame

        if name is not None:
            names.append(name)

    for ((top, right, bottom, left), name) in zip(face_locations, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
# loop over frames from the video file stream
with ThreadPoolExecutor() as executor:
    try:
        while True:
            frame = vs.read()

            # Submit each frame to the executor for parallel processing
            future = executor.submit(process_frame, frame)

            # Get the result from the processed frame
            frame = future.result()

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break  # exit the loop when 'q' is pressed

            fps.update()
    except KeyboardInterrupt:
        pass  # handle KeyboardInterrupt (Ctrl+C) gracefully

fps.stop()

# Close the MySQL connection
cursor.close()
conn.close()  # Close the individual connection

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()