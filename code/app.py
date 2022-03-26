import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import (
    application,
    loadData,
)  # import your pages here

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("Data Storyteller Application")

# Add all your applications (pages) here
app.add_page("Upload Data", application)
app.add_page("execution", loadData)


# The main app
app.run()
