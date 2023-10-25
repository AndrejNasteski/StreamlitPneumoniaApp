# StreamlitPneumoniaApp

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

Steps to configure Firebase service Account:

    YourFirebaseProject -> Project settings -> Service accounts -> Generate new Private keys
    Download that .json file to the root app directory and name it Firebase_Service_Account_Keys.json
    In .env file add the key "serviceAccount" and set the value as "Firebase_Service_Account_Keys.json"