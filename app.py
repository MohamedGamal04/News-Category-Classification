import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from nltk.corpus import wordnet
import contractions
from nltk.corpus import stopwords
import re 
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#Preprocess
def preprocess_text(text):
    text = contractions.fix(text)
    text = text.replace('.', ' . ')
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = "".join(num2words(int(word)) if word.isdigit() else word for word in text)
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.tag.pos_tag(text)
    lemmatized_words = []

    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_words)

# Streamlit App
import pickle
@st.cache_resource
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, clf = load_model()
st.set_page_config(page_title="News Category Classification", page_icon="📰", layout="wide")

st.title("📰 News Category Classification")
st.markdown("### Classify news articles into different categories using Machine Learning")

# Sidebar for model info
with st.sidebar:
    st.header("Model Information")
    st.info("""
    **Model:** Logistic Regression
    
    **Accuracy:** ~90%

    **Dataset:** AG News Classification Dataset

    **Classes:**
    - World (1)
    - Sports (2)
    - Business (3)
    - Sci/Tech (4)
    """)
    
    st.header("About")
    st.markdown("""
    This app uses a trained Logistic Regression model with TF-IDF vectorization 
    to predict category from text.
    
    The preprocessing includes:
    - Contraction expansion
    - HTML tag removal
    - Stopword removal
    - Lemmatization
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Insights"])

with tab1:
    st.header("Single News Article Analysis")
    
    # Text input
    user_input = st.text_area(
        "Enter your news article:",
        height=150,
        placeholder="Type or paste your news article here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🔍 Classify a News Article", type="primary")
    
    if predict_button and user_input:
        with st.spinner("Processing..."):
            # Preprocess
            processed_text = preprocess_text(user_input)
            
            # For demo purposes - you'll need to load your actual trained model
            # vectorizer and clf should be loaded from saved files
            st.success("✅ Analysis Complete!")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                prediction = clf.predict(vectorizer.transform([processed_text]))[0]
                confidence = clf.predict_proba(vectorizer.transform([processed_text]))[0]

                if  prediction == 1:
                    st.metric("Class", "🌍 Category: World", delta= "High Confidence" if confidence.max() > 0.5 else "Low Confidence")
                elif prediction == 2:
                    st.metric("Class", "⚽ Category: Sports", delta= "High Confidence" if confidence.max() > 0.5 else "Low Confidence")
                elif prediction == 3:
                    st.metric("Class", "💼 Category: Business", delta= "High Confidence" if confidence.max() > 0.5 else "Low Confidence")
                elif prediction == 4:
                    st.metric("Class", "🔬 Category: Sci/Tech", delta= "High Confidence" if confidence.max() > 0.5 else "Low Confidence")
                else:
                    st.metric("Class", "❓ Category: Unknown", delta="N/A")

            with col2:
                st.metric("Confidence Score", f"{confidence.max()*100:.2f}%")
            
            # Show processed text
            with st.expander("🔎 View Preprocessed Text"):
                st.text(processed_text)
    
    elif predict_button and not user_input:
        st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.header("Batch News Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file with news articles", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("🚀 Classify All News Articles") and df['Title'].notna().any() and df['Description'].notna().any():
            with st.spinner("Processing batch..."):
                # Process all reviews
                df['processed_description'] = df['Description'].apply(preprocess_text)
                df['processed_title'] = df['Title'].apply(preprocess_text)
                df['Text'] = df['Title'] + ' ' + df['Description']

                predictions = clf.predict(vectorizer.transform(df['Text']))
                df['class'] = predictions
                
                st.success("✅ Batch analysis complete!")
                st.dataframe(df)
                
                # Show statistics
                col1, col2, col3 , col4,col5 = st.columns(5)
                with col1:
                    st.metric("Total Reviews", len(df))
                with col2:
                    st.metric("🌍 Category: World", len(df[df['class'] == 1]))
                with col3:
                    st.metric("⚽ Category: Sports", len(df[df['class'] == 2]))
                with col4:
                    st.metric("💼 Category: Business", len(df[df['class'] == 3]))
                with col5:
                    st.metric("🔬 Category: Sci/Tech", len(df[df['class'] == 4]))
        else:
            st.info("ℹ️ Please upload a CSV file with a 'Title' and 'Description' column to analyze.")

with tab3:
    st.header("Model Insights & Feature Importance")
    
    st.markdown("""
    ### Top Features in Each Category
    These words have the strongest influence on class prediction.
    """)
    
    # Example feature importance (replace with actual from your model)
    top_world = ['Say', 'Iraq', 'Kill', 'Threenine', 'Ap']
    top_sports  = ['Threenine', 'Ap', 'Game', 'One', 'Win']
    top_business = ['Oil', 'Threenine', 'Reuters', 'Say', 'Us']
    top_scitech = ['Microsoft', 'Threenine', 'New', 'The', 'Software']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success("**Top 'World' Words**")
        for i, word in enumerate(top_world, 1):
            st.write(f"{i}. {word}")
    
    with col2:
        st.error("**Top 'Sports' Words**")
        for i, word in enumerate(top_sports, 1):
            st.write(f"{i}. {word}")

    with col3:
        st.warning("**Top 'Business' Words**")
        for i, word in enumerate(top_business, 1):
            st.write(f"{i}. {word}")

    with col4:
        st.info("**Top 'Sci/Tech' Words**")
        for i, word in enumerate(top_scitech, 1):
            st.write(f"{i}. {word}")

    st.markdown("---")
    st.info("""
    **To use your trained model:**
    
    ```
    # Save your model and vectorizer:
    import pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    # Load in Streamlit:
    @st.cache_resource
    def load_model():
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    
    vectorizer, clf = load_model()
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Built with Streamlit | Powered by Logistic Regression & TF-IDF</p>"
    "</div>",
    unsafe_allow_html=True

)


