import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from datetime import datetime
import base64
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import smtplib
import ssl
from email.message import EmailMessage
import mimetypes
from dotenv import load_dotenv

load_dotenv()

# Google Drive Setup
SCOPES = ['https://www.googleapis.com/auth/drive']
creds = service_account.Credentials.from_service_account_info(
    eval(os.getenv('GOOGLE_CREDENTIALS')),
    scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=creds)

# Email Setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

def download_images_from_drive(folder_id, local_folder="downloaded_images"):
    """Downloads images from Google Drive"""
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    existing_files = set(os.listdir(local_folder))
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    image_paths = []
    total_files = len(files)
    
    for i, file in enumerate(files):
        file_name = file['name']
        file_path = os.path.join(local_folder, file_name)

        if file_name in existing_files:
            st.write(f"🔹 {file_name} already exists. Skipping download.")
            image_paths.append(file_path)
            continue

        file_id = file['id']
        request = drive_service.files().get_media(fileId=file_id)

        with open(file_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        st.write(f"✅ Downloaded {file_name}")
        image_paths.append(file_path)
        # Update progress
        st.progress((i + 1) / total_files)

    return image_paths

def detect_faces(image):
    """Detects faces in an image using OpenCV"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return img, faces

def match_faces(reference_image, group_photos, model_name="Facenet"):
    """Matches faces using DeepFace and returns matched & unmatched images"""
    if isinstance(reference_image, str):
        ref_img = cv2.imread(reference_image)
    else:
        ref_img = reference_image
        
    ref_embedding = DeepFace.represent(ref_img, model_name=model_name)[0]["embedding"]
    
    matched_images = []
    unmatched_images = []
    total_photos = len(group_photos)
    
    for idx, img_path in enumerate(group_photos):
        img, faces = detect_faces(img_path)
        matched = False

        for (x, y, w, h) in faces:
            face_crop = img[y:y+h, x:x+w]
            try:
                group_embedding = DeepFace.represent(face_crop, model_name=model_name)[0]["embedding"]
                similarity = np.dot(ref_embedding, group_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(group_embedding))

                if similarity > 0.6:  # Matching threshold
                    
                    matched_images.append(img)
                    matched = True
                    break
            except:
                continue

        if not matched:
            unmatched_images.append(img)
            
        # Update progress
        st.progress((idx + 1) / total_photos)

    return matched_images, unmatched_images

# Rest of the code remains the same...

def save_matched_images(images, output_folder="matched_images"):
    """Saves matched images locally before sending via email"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(output_folder, f"matched_{idx+1}.jpg")
        cv2.imwrite(img_path, img)
        saved_paths.append(img_path)

    return saved_paths

def send_email_with_images(receiver_email, matched_images):
    """Sends matched images via email as attachments"""
    if not matched_images:
        st.warning("No matched images to send.")
        return

    image_paths = save_matched_images(matched_images)

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = receiver_email
    msg["Subject"] = "Matched Face Images 📸"

    msg.set_content(f"Attached are {len(image_paths)} matched images.")

    for image_path in image_paths:
        with open(image_path, "rb") as f:
            file_data = f.read()
            file_type, _ = mimetypes.guess_type(image_path)
            msg.add_attachment(file_data, maintype="image", 
                             subtype=file_type.split("/")[1], 
                             filename=os.path.basename(image_path))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        st.success(f"✅ Email sent successfully to {receiver_email}!")

    except Exception as e:
        st.error(f"❌ Error sending email: {e}")

def process_uploaded_image(uploaded_file):
    """Convert uploaded file to OpenCV format"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def main():
    st.title("Face Matching Application 👤")
    st.write("Upload a reference image or take a photo to find matching faces in the group photos.")

    # Initialize session state
    if 'matched_images' not in st.session_state:
        st.session_state.matched_images = []

    # Sidebar settings
    st.sidebar.title("Settings")
    google_drive_folder_id = st.sidebar.text_input(
        "Google Drive Folder ID",
        value="1VlDby99dH2Sg-QH4yFkM_oIWAN_dFi1i",
        help="Enter your Google Drive folder ID containing the group photos"
    )

    # Main interface with tabs
    tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a reference image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            reference_img = process_uploaded_image(uploaded_file)
            st.image(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB), 
                    caption="Uploaded Image", use_container_width=True)
            
            if st.button("Find Matches", key="upload_match"):
                with st.spinner("Downloading group photos from Google Drive..."):
                    group_photos = download_images_from_drive(google_drive_folder_id)
                
                with st.spinner("Finding matches..."):
                    matched_images, _ = match_faces(reference_img, group_photos)
                    st.session_state.matched_images = matched_images

    with tab2:
        picture = st.camera_input("Take a picture")
        if picture is not None:
            reference_img = process_uploaded_image(picture)
            
            if st.button("Find Matches", key="camera_match"):
                with st.spinner("Downloading group photos from Google Drive..."):
                    group_photos = download_images_from_drive(google_drive_folder_id)
                
                with st.spinner("Finding matches..."):
                    matched_images, _ = match_faces(reference_img, group_photos)
                    st.session_state.matched_images = matched_images

    # Display matched images
    if st.session_state.matched_images:
        st.subheader(f"Matched Images ({len(st.session_state.matched_images)})")
        cols = st.columns(3)
        for idx, img in enumerate(st.session_state.matched_images):
            with cols[idx % 3]:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)

        # Email functionality
        st.subheader("Send Results via Email")
        email = st.text_input("Enter email address")
        if st.button("Send Email") and email:
            with st.spinner("Sending email..."):
                send_email_with_images(email, st.session_state.matched_images)

    # Instructions in sidebar
    with st.sidebar.expander("How to Use"):
        st.write("""
        1. Enter your Google Drive folder ID in the sidebar
        2. Upload a reference image or take a photo
        3. Click 'Find Matches' to see results
        4. Enter an email address to receive the matches
        
        Green rectangles indicate matched faces in the results.
        """)

if __name__ == "__main__":
    main()