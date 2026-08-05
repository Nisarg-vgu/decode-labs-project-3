import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="DecodeLabs AI Career Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Career & Tech Stack Advisor")
st.subheader("DecodeLabs Project 3: AI Recommendation Engine")
st.markdown("Enter your skills/interests below (e.g., *Python, Cloud, Automation*) to get personalized career path recommendations based on similarity matching!")

# ==========================================
# DATASET & RECOMMENDATION ENGINE
# ==========================================
@st.cache_data
def load_dataset():
    data = {
        'Job_Role': [
            'Data Scientist',
            'DevOps Engineer',
            'Backend Developer',
            'Frontend Developer',
            'Machine Learning Engineer',
            'Cloud Architect',
            'Cybersecurity Analyst',
            'Data Engineer'
        ],
        'Required_Skills': [
            'Python SQL Machine Learning Data Analysis Statistics Pandas NumPy',
            'AWS Docker Kubernetes CI/CD Automation Linux Cloud Terraform',
            'Java Python SQL APIs Microservices Node.js Backend Postgres',
            'HTML CSS JavaScript React Web Design Frontend UI UX Next.js',
            'Python Machine Learning TensorFlow PyTorch Algorithms Data Science Deep Learning',
            'AWS Cloud Infrastructure Security Docker Systems Architecture Azure Network',
            'Cybersecurity Networking Linux Cryptography Security Firewalls Ethical Hacking',
            'Python SQL Spark ETL Data Lakes Warehousing Airflow Pipeline'
        ]
    }
    return pd.DataFrame(data)

class TechStackRecommender:
    def __init__(self, dataset):
        self.df = dataset
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Vectorize job role skills into TF-IDF Matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Required_Skills'])

    def recommend(self, user_skills, top_n=3):
        # Validation: Require at least 3 skills to ensure data density
        if len(user_skills) < 3:
            return "⚠️ Please enter **at least 3 skills** so I can calculate accurate similarity scores!"

        # Combine input list into string
        user_profile_str = " ".join(user_skills)

        # Vectorize user input
        user_vector = self.vectorizer.transform([user_profile_str])

        # Calculate Cosine Similarity
        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        results = self.df.copy()
        results['Similarity_Score'] = similarity_scores

        # Sort and take Top-N matches
        ranked_results = results.sort_values(by='Similarity_Score', ascending=False)
        top_recommendations = ranked_results.head(top_n)

        # Check for User Cold Start
        if top_recommendations['Similarity_Score'].iloc[0] == 0:
            return "🔍 **Cold Start Encountered:** I couldn't find any matching skills in my database. Try entering tech keywords like *Python, AWS, React, SQL, Linux, or Machine Learning*!"

        return top_recommendations[['Job_Role', 'Required_Skills', 'Similarity_Score']]

# Initialize Engine
df = load_dataset()
recommender = TechStackRecommender(df)

# ==========================================
# CHAT SESSION STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am your **AI Tech Stack Advisor**. Tell me at least **3 skills or technologies** you know or want to learn, separated by commas (e.g., `Python, AWS, Automation`)."
        }
    ]

# Display existing chat message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# CHAT INTERACTION LOGIC
# ==========================================
if user_input := st.chat_input("Type your skills here (comma separated)..."):
    # Render User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process Input
    raw_skills = [skill.strip() for skill in user_input.split(",") if skill.strip()]
    
    # Generate Bot Response
    with st.chat_message("assistant"):
        with st.spinner("Calculating TF-IDF vectors & Cosine Similarity..."):
            recommendations = recommender.recommend(raw_skills, top_n=3)

            if isinstance(recommendations, str):
                bot_response = recommendations
                st.markdown(bot_response)
            else:
                bot_response = "Here are your **Top 3 Recommended Tech Career Paths** based on your inputs:\n\n"
                for idx, row in recommendations.iterrows():
                    match_percentage = round(row['Similarity_Score'] * 100, 2)
                    bot_response += f"### 🎯 **{row['Job_Role']}** ({match_percentage}% Match)\n"
                    bot_response += f"- **Key Skills:** `{row['Required_Skills']}`\n\n"
                
                st.markdown(bot_response)

    # Save Assistant Response to Session History
    st.session_state.messages.append({"role": "assistant", "content": bot_response})