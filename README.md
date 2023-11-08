# StreamlitPneumoniaApp

Pneumonia Detection App using Streamlit.
After registering and logging in, the user can upload a image of a lung x-ray.
The image is then classified as normal (healthy) or lungs with pneumonia.
Users with 'moderator' privileges can label uploaded images.
Users with 'admin' privileges can choose to retrain the model using the newly acquired data.

Install dependencies:
``` pip install pip-tools
    pip-sync requirements.txt
```

Update dependencies:
```
    pip-compile requirements.in   // pip-compile
    pip-compile --upgrade requirements.in
```

To run app:
```
streamlit run app.py        (must be in same directory)
```

The .env file contains the Firebase configuration and the databaseURL.
