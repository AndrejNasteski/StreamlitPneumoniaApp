import json
import os
import time
from datetime import datetime

import pyrebase
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from requests.exceptions import HTTPError

from model_aux import classify_image, retrain_model

ROLES = ["USER", "MODERATOR", "ADMIN"]

load_dotenv()
st.set_page_config(layout="wide")

firebaseConfig = {
    "apiKey": os.environ.get("apiKey"),
    "authDomain": os.environ.get("authDomain"),
    "projectId": os.environ.get("projectId"),
    "storageBucket": os.environ.get("storageBucket"),
    "messagingSenderId": os.environ.get("messagingSenderId"),
    "appId": os.environ.get("appId"),
    "measurementId": os.environ.get("measurementId"),
    "databaseURL": os.environ.get("databaseURL"),
    "serviceAccount": os.environ.get("serviceAccount"),
}


firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()


if "user" not in st.session_state:
    st.session_state["user"] = None
    st.session_state["user_role"] = None
    st.info("Log in to upload an image.")

if "label_button_enabled" not in st.session_state:
    st.session_state["label_button_enabled"] = False

if "image_id_db" not in st.session_state:
    st.session_state["image_id_db"] = None

with st.sidebar:
    st.header("Pneumonia Detection App")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form(key="Login", clear_on_submit=True):
            email_login = st.text_input("E-mail")
            password_login = st.text_input("Password")
            login_submit = st.form_submit_button("Login")

    with register_tab:
        with st.form(key="Register", clear_on_submit=True):
            email_register = st.text_input("Enter E-mail")
            username_register = st.text_input("Enter Username")
            password_register = st.text_input("Enter Password")
            confirm_password = st.text_input("Confirm Password")
            register_submit = st.form_submit_button("Register")

    if register_submit:
        if username_register == "":
            st.error("Please enter a username.")
        elif password_register != confirm_password:
            st.error("Passwords must match.")
        else:
            try:
                user = auth.create_user_with_email_and_password(
                    email_register, password_register
                )
                data = {
                    "ID": user["localId"],
                    "Username": username_register,
                    "Role": ROLES[2],
                }
                db.child("users").child(user["localId"]).set(data)
                st.success(
                    "Account registered successfully. You can now login into your account."
                )
                time.sleep(1)
            except HTTPError as e:
                error = json.loads(e.args[1])["error"]["message"]
                if error == "INVALID_LOGIN_CREDENTIALS":
                    st.error("Incorrect username or password.")
                elif error == "INVALID_EMAIL":
                    st.error("Invalid email address.")
                elif error == "MISSING_PASSWORD":
                    st.error("Missing password.")
                elif error == "MISSING_EMAIL":
                    st.error("Missing email address.")

    if login_submit:
        try:
            user = auth.sign_in_with_email_and_password(email_login, password_login)
            user_role = (
                db.child("users")
                .order_by_key()
                .equal_to(user["localId"])
                .limit_to_first(1)
                .get()
            )
            st.session_state["user"] = user
            st.session_state["user_role"] = user_role.val()[user["localId"]]["Role"]
            st.success("Logged in.")
            time.sleep(1)
        except HTTPError as e:
            error = json.loads(e.args[1])["error"]["message"]
            if error == "INVALID_LOGIN_CREDENTIALS":
                st.error("Incorrect username or password.")
            elif error == "INVALID_EMAIL":
                st.error("Invalid email address.")
            elif error == "MISSING_PASSWORD":
                st.error("Missing password.")


col1, col2 = st.columns([1, 2])
with col1:
    st.write(st.session_state.get("user_role"))  # ----------------------------- testing

    file = st.file_uploader(
        "Upload an image:",
        type=["png", "jpg"],
        disabled=(st.session_state["user"] is None),
    )
    btc1, btc2 = st.columns([1, 1])
    btc3, btc4 = st.columns([1, 1])

    with btc1:
        classify_button = st.button(
            "Classify Image",
            type="primary",
            disabled=st.session_state["user"] is None or not file,
        )
    with btc2:
        retrain_button = st.button(
            "Re-train Model",
            type="primary",
            disabled=st.session_state["user"] is None
            or st.session_state["user_role"] != ROLES[2],
        )

    if file and classify_button:
        with st.spinner("Running model..."):
            id = str(int(datetime.now().timestamp() * 1000))
            st.session_state["image_id_db"] = id

            storage.child("images").child(id).put(file)
            image = Image.open(file)
            y, y_prob = classify_image(image)
            data = {
                "Author_ID": st.session_state.get("user")["localId"],
                "Created_at": str(datetime.now()),
                "Model_label": y,
                "User_label": "",
            }
            db.child("images").child(id).set(data)

        image_classification = f"Image classified as: {y}."
        image_probability = f"Class prediction probability: {y_prob * 100:.2f}%."
        st.info(image_classification + "\n" + image_probability)

    with btc3:
        label_as_normal_button = st.button(
            "Label as Normal",
            disabled=st.session_state["user"] is None
            or st.session_state["user_role"] == ROLES[0]
            or not file
            or st.session_state["image_id_db"] is None,
        )
    with btc4:
        label_as_pneumonia_button = st.button(
            "Label as Pneumonia",
            disabled=st.session_state["user"] is None
            or st.session_state["user_role"] == ROLES[0]
            or not file
            or st.session_state["image_id_db"] is None,
        )

    if label_as_normal_button:
        if st.session_state["image_id_db"] is None:
            st.error("Error labeling image.")
        else:
            db.child("images").child(st.session_state["image_id_db"]).update(
                {"User_label": "NORMAL"}
            )
            st.session_state["image_id_db"] = None
            st.info("Image labeled as: Normal.")
            time.sleep(2)
            st.rerun()
    elif label_as_pneumonia_button:
        if st.session_state["image_id_db"] is None:
            st.error("Error labeling image.")
        else:
            db.child("images").child(st.session_state["image_id_db"]).update(
                {"User_label": "PNEUMONIA"}
            )
            st.session_state["image_id_db"] = None
            st.info("Image labeled as: Pneumonia.")
            time.sleep(2)
            st.rerun()

    if retrain_button:
        progress_bar = st.progress(0, "Downloading images...")
        images_db = db.child("images").get().val()  # Database entry
        blob_list = storage.list_files()  # Actual image
        for i, image_file in enumerate(blob_list):
            if i == 0:  # root directory, not image
                continue
            if i > 10:  # change this --------------------------------
                break
            image_file.download_to_filename(
                f"files/temp/{image_file.name.split('/')[1]}.jpeg"
            )
        progress_bar.progress(25, "Images downloaded, loading and training model...")
        return_message = retrain_model(images_db)
        time.sleep(2)
        progress_bar.progress(100, "Model successfully re-trained on new data")
        time.sleep(2)
        progress_bar.empty()
        st.info(return_message)


with col2:
    if file:
        image = Image.open(file)
        new_image = image.resize((1000, 700))
        st.image(new_image)
