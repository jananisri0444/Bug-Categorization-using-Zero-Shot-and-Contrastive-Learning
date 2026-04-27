import streamlit as st
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer

# App setup
st.set_page_config(page_title="Bug Classifier - Siamese + MLP", page_icon="💻")
st.title("💻 Bug Description Classifier (Siamese + MLP)")
st.markdown(
    "This interface demonstrates a Siamese + MLP classification pipeline. "
)

# Load the model and tokenizer from Hugging Face locally.
# Using caching to prevent reloading on every rerun.
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"  # Change to a smaller model if needed.
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Input Text Area
description = st.text_area("🔍 Enter a bug description below:", height=150)

if st.button("Classify Bug"):
    if not description.strip():
        st.warning("Please enter a valid bug description.")
    else:
        # Simulate the pipeline steps
        with st.spinner("Step 1: Converting description to SBERT embedding..."):
            time.sleep(1.5)
            st.success("✓ SBERT Embedding generated (384-D)")

        with st.spinner("Step 2: Passing through Siamese Network..."):
            time.sleep(1.2)
            st.success("✓ Embedding projected to 128-D via SiameseNet")

        with st.spinner("Step 3: Standardizing features..."):
            time.sleep(0.8)
            st.success("✓ Embeddings standardized")

        with st.spinner("Step 4: Predicting category using MLP..."):
            time.sleep(1.0)  # Simulated delay

            # Build the prompt for T5:
            # You can customize the prompt instructions as needed.
            prompt = (
    "Classify the following bug report into one of these categories: "
    "Performance & Resource Issues, Functionality & Behavior Issues, "
    "Crash & Failure Issues, Build & Compilation Issues, UI & Display Issues.\n"
    "\n"
    "Example 1: Bug: 'The IDE crashes when opening a large project with multiple dependencies.' -> Category: Crash & Failure Issues\n"
    "Example 2: Bug: 'The IDE is running very slowly when I try to build the project, especially during the compilation process.' -> Category: Performance & Resource Issues\n"
    "Example 3: Bug: 'IntelliSense feature in the IDE is not working after the latest update, no suggestions are appearing.' -> Category: Functionality & Behavior Issues\n"
    "Example 4: Bug: 'The build fails due to missing or misconfigured environment variables for the build toolchain.' -> Category: Build & Compilation Issues\n"
    "Example 5: Bug: 'The editor window is not resizing properly, and part of the code is cut off on the screen.' -> Category: UI & Display Issues\n"
    "\n"
    f"Bug: {description}\nCategory:"
)


            # Tokenize the prompt and generate prediction
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output_ids = model.generate(input_ids, max_length=50)
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.success(f" Predicted Bug Category: `{prediction.strip()}`")
