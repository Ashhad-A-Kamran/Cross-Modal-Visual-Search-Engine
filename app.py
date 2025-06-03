import streamlit as st
import os # For os.path.exists, os.remove

from config import (
    device, MODEL_PATH, CSV_PATH,
    IMAGE_INDEX_FEATURES_PATH, IMAGE_INDEX_METADATA_PATH,
    DEFAULT_TOP_K_RESULTS, DEFAULT_RESULTS_COLS
)
from initialize import initialize_system

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="centered", page_title="Visual Search")

# --- Main Streamlit App UI ---
st.title("🖼️ Visual Item Search")
st.markdown("Describe an item or upload an image, and we'll find similar items.")

app = initialize_system()

st.sidebar.success("System Ready!")
st.sidebar.caption(f"Using device: {device}")
st.sidebar.metric("Items in Index:", len(app.image_urls) if app and hasattr(app, 'image_urls') else 0)


# --- Search Inputs ---
search_type = st.radio("Search by:", ("Text Description", "Uploaded Image", "Image URL"), horizontal=True, key="search_type_radio")

search_query_text = ""
uploaded_query_image = None
search_query_image_url = ""

if search_type == "Text Description":
    search_query_text = st.text_input("Enter item description:", key="search_query_text", placeholder="e.g., red apple with a leaf")
elif search_type == "Uploaded Image":
    uploaded_query_image = st.file_uploader("Upload an image to search for similar items:", type=["jpg", "jpeg", "png"], key="uploaded_query_image")
elif search_type == "Image URL":
    search_query_image_url = st.text_input("Enter image URL to search for similar items:", key="search_query_image_url", placeholder="https://...")

# Initialize session state variables if they don't exist
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'last_search_message' not in st.session_state: st.session_state.last_search_message = ""
if 'last_query_display_type' not in st.session_state: st.session_state.last_query_display_type = None
if 'last_query_content' not in st.session_state: st.session_state.last_query_content = None


search_button = st.button("Search 🔎", use_container_width=True, type="primary")

if search_button:
    query_valid = False
    results, message = [], "No query provided or query type error."
    # Reset previous query display info
    st.session_state.last_query_display_type = None
    st.session_state.last_query_content = None


    with st.spinner("Searching..."):
        if search_type == "Text Description" and search_query_text:
            results, message = app.search_by_text(search_query_text, top_k=st.session_state.get("top_k_results", DEFAULT_TOP_K_RESULTS))
            st.session_state.last_query_display_type = "text"
            st.session_state.last_query_content = search_query_text
            query_valid = True
        elif search_type == "Uploaded Image" and uploaded_query_image:
            # To display the uploaded image after search, store its bytes
            # Be cautious with large files in session state
            uploaded_image_bytes = uploaded_query_image.getvalue()
            results, message = app.search_by_image(uploaded_image_bytes, top_k=st.session_state.get("top_k_results", DEFAULT_TOP_K_RESULTS))
            st.session_state.last_query_display_type = "uploaded_image"
            st.session_state.last_query_content = uploaded_image_bytes # Store bytes for display
            query_valid = True
        elif search_type == "Image URL" and search_query_image_url:
            results, message = app.search_by_image(search_query_image_url, top_k=st.session_state.get("top_k_results", DEFAULT_TOP_K_RESULTS))
            st.session_state.last_query_display_type = "image_url"
            st.session_state.last_query_content = search_query_image_url
            query_valid = True
        else:
            st.toast("⚠️ Please provide a query (text, upload, or URL).", icon="📝")
            message = "Query was empty or invalid."

    if query_valid:
        st.session_state.search_results = results
        st.session_state.last_search_message = message
    else:
        st.session_state.search_results = []
        st.session_state.last_search_message = message


# --- Display Query and Results ---
if st.session_state.last_query_display_type:
    st.markdown("---")
    if st.session_state.last_query_display_type == "text":
        st.markdown(f"**Searched for text:** \"{st.session_state.last_query_content}\"")
    elif st.session_state.last_query_display_type == "image_url":
        st.markdown(f"**Query Image (from URL):**")
        try:
            st.image(st.session_state.last_query_content, width=150)
        except Exception as e:
            st.error(f"Could not display query image from URL: {e}")
    elif st.session_state.last_query_display_type == "uploaded_image":
        st.markdown(f"**Query Image (Uploaded):**")
        try:
            st.image(st.session_state.last_query_content, width=150, caption="Your Query Image")
        except Exception as e:
            st.error(f"Could not display uploaded query image: {e}")


if st.session_state.search_results:
    st.markdown("---") # Separator only if there are results or a query was made
    st.markdown(f"Found **{len(st.session_state.search_results)}** image(s):")
    num_cols = st.session_state.get("results_cols", DEFAULT_RESULTS_COLS)
    cols = st.columns(num_cols)
    for i, result in enumerate(st.session_state.search_results):
        with cols[i % num_cols]:
            try:
                st.image(result['image_url'], caption=f"Score: {result['score']:.3f}", use_container_width=True)
                with st.expander("Details", expanded=False):
                    st.caption(f"Original: {result.get('original_text', 'N/A')}")
                    st.caption(f"URL: {result['image_url']}")
            except Exception as e:
                st.error(f"Err displaying: {result['image_url'][:30]}...")
elif st.session_state.last_search_message: # If no results but there was a message from search attempt
    st.markdown("---")
    msg = st.session_state.last_search_message
    if "No matches found" in msg: st.info("🤔 No matches found for your query. Try a different description or image!")
    elif "Index is empty" in msg: st.warning("⚠️ The image index is currently empty. Please add items or check dataset setup.")
    elif "Query was empty" not in msg and "Found" not in msg and "match(es)" not in msg: # Avoid showing generic success/found messages here
        st.info(msg)


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls & Info")
    st.subheader("Add Item to Index")
    with st.expander("Add New Item", expanded=False):
        add_url = st.text_input("Image URL:", key="add_item_url")
        add_title = st.text_input("Item Title (optional):", key="add_item_title", placeholder="e.g., Blue Backpack")
        if st.button("Add to Index", key="add_item_button"):
            if add_url:
                with st.spinner("Adding..."):
                    title_to_add = add_title if add_title else "N/A (Newly Added via URL)"
                    success, msg = app.add_new_item(add_url, item_title=title_to_add)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please enter an image URL to add.")

    st.subheader("Search Options")
    st.slider("Max Results (Top K):", 1, 20, DEFAULT_TOP_K_RESULTS, key="top_k_results")
    st.number_input("Result Columns:", 1, 5, DEFAULT_RESULTS_COLS, key="results_cols")

    st.markdown("---")
    if st.button("Clear Search Results & Query"):
        st.session_state.search_results = []
        st.session_state.last_search_message = ""
        st.session_state.last_query_display_type = None
        st.session_state.last_query_content = None
        # Clear text input fields too by resetting their keys in session_state
        if 'search_query_text' in st.session_state: st.session_state.search_query_text = ""
        if 'search_query_image_url' in st.session_state: st.session_state.search_query_image_url = ""
        # For file uploader, clearing is trickier, often involves rerunning or a None value with a new key
        st.rerun()

    if st.button("Rebuild Index & Retrain Model"):
        if os.path.exists(IMAGE_INDEX_FEATURES_PATH): os.remove(IMAGE_INDEX_FEATURES_PATH)
        if os.path.exists(IMAGE_INDEX_METADATA_PATH): os.remove(IMAGE_INDEX_METADATA_PATH)
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        # Clear the Streamlit cache for the initialize_system function
        st.cache_resource.clear() # Clears all @st.cache_resource
        # If you want to clear specific function: initialize_system.clear() but must be defined first
        st.success("Cleared cached model, index, and retrain flags. App will rebuild on next interaction or reload.")
        st.rerun()