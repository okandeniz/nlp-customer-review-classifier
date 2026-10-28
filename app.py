import os

import requests
import streamlit as st


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Review Classifier",
    page_icon="🛍️",
    layout="centered"
)

API_BASE_URL = os.getenv(
    "REVIEW_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛍️ Customer Review Classifier")

st.write(
    "Analyze customer reviews using a trained "
    "Natural Language Processing model."
)

st.caption(
    "Model: TF-IDF + Balanced Logistic Regression"
)

st.divider()


# --------------------------------------------------
# Review input form
# --------------------------------------------------

st.subheader("Customer Review")

with st.form("review_form"):

    review_text = st.text_area(
        label="Enter your customer review",
        placeholder=(
            "Example: I really love this dress. "
            "The fabric is soft and comfortable."
        ),
        height=150,
        max_chars=5000
    )

    submitted = st.form_submit_button(
        "Analyze Review"
    )


# --------------------------------------------------
# API prediction
# --------------------------------------------------

if submitted:

    review_text = review_text.strip()

    if not review_text:
        st.warning("Please enter a customer review.")

    else:

        try:

            with st.spinner("Analyzing your review..."):

                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json={
                        "review_text": review_text
                    },
                    timeout=15
                )

                response.raise_for_status()

                result = response.json()

            # Extract prediction results
            prediction = int(result["prediction"])
            label = result["label"]

            prob_recommended = float(
                result["probability_recommended"]
            )

            prob_not_recommended = float(
                result["probability_not_recommended"]
            )

            threshold = float(result["threshold"])

            # Validate probability values
            if not (
                0 <= prob_recommended <= 1
                and 0 <= prob_not_recommended <= 1
            ):
                raise ValueError(
                    "Invalid probability values."
                )

            # Display results
            st.divider()
            st.subheader("Prediction Result")

            # Prediction result styling
            if prediction == 1:
                result_color = "#689EB8"
                result_background = "#E6F0F5"
            else:
                result_color = "#FF5A60"
                result_background = "#FFF0F1"

            st.markdown(
                f"""
                <div style="
                    background-color: {result_background};
                    border-left: 5px solid {result_color};
                    border-radius: 10px;
                    padding: 18px;
                    margin-bottom: 20px;
                    color: #262730;
                ">
                    <strong>Prediction: {label}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Recommended",
                    f"{prob_recommended:.2%}"
                )

            with col2:
                st.metric(
                    "Not Recommended",
                    f"{prob_not_recommended:.2%}"
                )

            st.write("Recommended Probability")

            st.progress(prob_recommended)

            st.caption(
                f"Classification threshold: "
                f"{threshold:.2f}"
            )

            st.info(
                "Probabilities are model estimates, "
                "not guaranteed confidence levels."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the API. "
                "Please check that FastAPI is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The API request timed out. "
                "Please try again."
            )

        except requests.exceptions.HTTPError as error:
            st.error(
                f"API error: "
                f"{error.response.status_code}"
            )

        except requests.exceptions.RequestException:
            st.error(
                "An error occurred while contacting the API."
            )

        except (ValueError, KeyError, TypeError):
            st.error(
                "The API returned an unexpected response."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Review Classification | "
    "TensorFlow NLP Project"
)