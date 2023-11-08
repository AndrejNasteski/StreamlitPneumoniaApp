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
st.set_page_config(
    layout="wide", page_title="Pneumonia Detection App", page_icon=":lungs:"
)

firebaseConfig = {
    "apiKey": os.environ.get("apiKey"),
    "authDomain": os.environ.get("authDomain"),
    "projectId": os.environ.get("projectId"),
    "storageBucket": os.environ.get("storageBucket"),
    "messagingSenderId": os.environ.get("messagingSenderId"),
    "appId": os.environ.get("appId"),
    "measurementId": os.environ.get("measurementId"),
    "databaseURL": os.environ.get("databaseURL"),
}


firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()


def reset_user_session():
    st.session_state["user"] = None
    st.session_state["user_role"] = None
    st.session_state["username"] = None


if "user" not in st.session_state:
    reset_user_session()

if st.session_state["user"] == None:
    st.info("Log in to upload an image.")

if "label_button_enabled" not in st.session_state:
    st.session_state["label_button_enabled"] = False

if "image_id_db" not in st.session_state:
    st.session_state["image_id_db"] = None

with st.sidebar:
    st.title(":medical_symbol: Pneumonia Detection App")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form(key="Login", clear_on_submit=True):
            email_login = st.text_input("E-mail")
            password_login = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login", use_container_width=True)

    with register_tab:
        with st.form(key="Register", clear_on_submit=True):
            email_register = st.text_input("Enter E-mail")
            username_register = st.text_input("Enter Username")
            password_register = st.text_input("Enter Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button(
                "Register", use_container_width=True
            )

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
                    "Role": ROLES[0],  # user role
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
            user_db = (
                db.child("users")
                .order_by_key()
                .equal_to(user["localId"])
                .limit_to_first(1)
                .get()
            )
            st.session_state["user"] = user
            st.session_state["user_role"] = user_db.val()[user["localId"]]["Role"]
            st.session_state["username"] = user_db.val()[user["localId"]]["Username"]
            st.success("Logged in.")
            time.sleep(1)
            st.rerun()
        except HTTPError as e:
            error = json.loads(e.args[1])["error"]["message"]
            if error == "INVALID_LOGIN_CREDENTIALS":
                st.error("Incorrect username or password.")
            elif error == "INVALID_EMAIL":
                st.error("Invalid email address.")
            elif error == "MISSING_PASSWORD":
                st.error("Missing password.")

    with st.container():  # display username and logout button
        if st.session_state["user"] == None:
            display_username = "Not logged in."
        else:
            display_username = "Logged in as: " + st.session_state["username"]

        c1, c2 = st.columns(2)
        with c1:
            st.write(display_username)
        with c2:
            if st.session_state["user"] is not None:
                logout_placeholder = st.button(
                    "Log out", on_click=reset_user_session, use_container_width=True
                )

col1, col2 = st.columns([1, 2])
with col1:
    file = st.file_uploader(
        "Upload an image:",
        type=["png", "jpg"],
        disabled=(st.session_state["user"] is None),
    )

    if not file:
        st.session_state["image_id_db"] = None

    btc1, btc2 = st.columns([1, 1])
    btc3, btc4 = st.columns([1, 1])

    with btc1:
        classify_button = st.button(
            "Classify Image",
            type="primary",
            disabled=st.session_state["user"] is None or not file,
            use_container_width=True,
        )
    with btc2:
        retrain_button = st.button(
            "Re-train Model",
            type="primary",
            disabled=st.session_state["user"] is None
            or st.session_state["user_role"] != ROLES[2],
            use_container_width=True,
        )

    if file and classify_button:
        with st.spinner("Running model..."):
            id = str(int(datetime.now().timestamp() * 1000))
            st.session_state["image_id_db"] = id

            upload = storage.child("images").child(id).put(file)
            image_url = (
                storage.child("images").child(id).get_url(upload["downloadTokens"])
            )
            image = Image.open(file)
            y, y_prob = classify_image(image)
            data = {
                "Author_ID": st.session_state.get("user")["localId"],
                "Created_at": str(datetime.now()),
                "Model_label": y,
                "User_label": "",
                "Image_URL": image_url,
            }
            db.child("images").child(id).set(data)

        image_classification = f"Image classified as: {y}."
        image_probability = f"Class prediction certainty: {y_prob * 100:.2f}%."
        st.info(image_classification + "\n" + image_probability)

    with btc3:
        label_as_normal_button = st.button(
            "Label as Normal",
            disabled=st.session_state["user"] is None
            or st.session_state["user_role"] == ROLES[0]
            or not file
            or st.session_state["image_id_db"] is None,
            use_container_width=True,
        )
    with btc4:
        label_as_pneumonia_button = st.button(
            "Label as Pneumonia",
            disabled=st.session_state["user"] is None
            or st.session_state["user_role"] == ROLES[0]
            or not file
            or st.session_state["image_id_db"] is None,
            use_container_width=True,
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
        images_db = db.child("images").get().val()  # Database entry
        image_keys = images_db.keys()
        image_list = []
        key_count = len(image_keys)
        progress_bar = st.progress(0, "Fetching images...")

        for i, key in enumerate(image_keys):
            image_list.append(
                [
                    images_db[key]["Image_URL"],
                    images_db[key]["Model_label"],
                    images_db[key]["User_label"],
                ]
            )
            progress_bar.progress(int((i / key_count) * 100), "Fetching images...")
        progress_bar.empty()

        with st.spinner("Images downloaded, loading and training model..."):
            return_message = retrain_model(image_list)

        st.info(return_message)
        time.sleep(2)


with col2:
    if file:
        image = Image.open(file)
        new_image = image.resize((1000, 650))
        st.image(new_image)
