import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import random
import os
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from faker import Faker

# Set page configuration
st.set_page_config(
    page_title="National Poster Presentation Event Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #f7f7f7;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Generate a synthetic dataset if it doesn't exist
@st.cache_data
def generate_dataset():
    fake = Faker()
    
    # Define constants
    num_participants = 400
    tracks = ["Engineering", "Medical Sciences", "Business & Economics", "Arts & Humanities"]
    states = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat", "Uttar Pradesh", 
              "West Bengal", "Rajasthan", "Telangana", "Kerala"]
    colleges = [
        "IIT Bombay", "IIT Delhi", "AIIMS Delhi", "NIT Trichy", "BITS Pilani", 
        "St. Xavier's College", "Christ University", "Manipal Institute of Technology",
        "Symbiosis International University", "Delhi University", "Anna University",
        "VIT Vellore", "JNTU Hyderabad", "Amity University", "SRM University",
        "Osmania University", "PSG College of Technology", "IIIT Hyderabad",
        "NIFT Mumbai", "NID Ahmedabad"
    ]
    
    feedback_templates = [
        "The event was {quality}. The organization was {org_quality} and the content was {content_quality}.",
        "I found the poster presentations {quality}. The venue was {venue_quality}.",
        "Overall, this was a {quality} experience. The speakers were {speaker_quality}.",
        "The event was {quality} organized. I particularly enjoyed the {aspect} aspect.",
        "A {quality} platform for networking. The {aspect} could be improved."
    ]
    
    qualities = ["excellent", "good", "average", "outstanding", "superb", "decent", "remarkable"]
    org_qualities = ["well-organized", "structured", "meticulous", "professional", "smooth"]
    content_qualities = ["informative", "insightful", "educational", "enlightening", "comprehensive"]
    venue_qualities = ["comfortable", "convenient", "appropriate", "well-equipped", "accessible"]
    speaker_qualities = ["knowledgeable", "engaging", "articulate", "experienced", "inspiring"]
    aspects = ["technical", "networking", "scheduling", "catering", "presentation", "Q&A", "poster layout"]
    
    # Generate data
    data = []
    
    for i in range(num_participants):
        participant_id = f"P{i+1:03d}"
        name = fake.name()
        track = random.choice(tracks)
        day = random.randint(1, 4)
        date = (datetime.datetime(2024, 4, 1) + datetime.timedelta(days=day-1)).strftime("%Y-%m-%d")
        state = random.choice(states)
        college = random.choice(colleges)
        age = random.randint(20, 35)
        
        # Generate feedback
        template = random.choice(feedback_templates)
        feedback = template.format(
            quality=random.choice(qualities),
            org_quality=random.choice(org_qualities),
            content_quality=random.choice(content_qualities),
            venue_quality=random.choice(venue_qualities),
            speaker_quality=random.choice(speaker_qualities),
            aspect=random.choice(aspects)
        )
        
        # Add track-specific keywords to feedback
        if track == "Engineering":
            track_keywords = ["innovation", "technology", "design", "prototype", "engineering", "technical"]
        elif track == "Medical Sciences":
            track_keywords = ["healthcare", "patient", "clinical", "medical", "treatment", "diagnosis"]
        elif track == "Business & Economics":
            track_keywords = ["market", "strategy", "economics", "finance", "business", "entrepreneurship"]
        else:  # Arts & Humanities
            track_keywords = ["culture", "society", "history", "literature", "arts", "creative"]
            
        feedback += f" The {random.choice(track_keywords)} aspect was particularly noteworthy."
        
        # Score from 1-5
        score = random.randint(3, 5)  # Slightly skewed towards positive reviews
        
        data.append({
            "Participant_ID": participant_id,
            "Name": name,
            "Track": track,
            "Day": day,
            "Date": date,
            "State": state,
            "College": college,
            "Age": age,
            "Feedback": feedback,
            "Rating": score
        })
    
    df = pd.DataFrame(data)
    return df

# Function to apply image filters
def apply_image_filter(image, filter_name):
    if filter_name == "Grayscale":
        return image.convert("L").convert("RGB")
    elif filter_name == "Sepia":
        grayscale = image.convert("L")
        sepia = Image.merge("RGB", [
            grayscale.point(lambda x: min(255, x * 1.1)),
            grayscale.point(lambda x: min(255, x * 0.9)),
            grayscale.point(lambda x: min(255, x * 0.7))
        ])
        return sepia
    elif filter_name == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_name == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_name == "Enhance":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    else:
        return image

def text_similarity_analysis(df, track):
    # Filter data for the selected track
    track_data = df[df['Track'] == track]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Check if there are any feedback entries
    if track_data.shape[0] == 0:
        return "No feedback data available for this track."
    
    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(track_data['Feedback'].tolist())
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create a heatmap of the similarity matrix
    plt.figure(figsize=(8, 6))  # Reduced from (10, 8)
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title(f'Feedback Similarity Matrix for {track}')
    plt.xlabel('Feedback Index')
    plt.ylabel('Feedback Index')
    
    # Return the plot for displaying in Streamlit
    return plt

def main():
    # Initialize session state for storing images
    if 'track_day_images' not in st.session_state:
        st.session_state.track_day_images = {
            "Engineering": {1: [], 2: [], 3: [], 4: []},
            "Medical Sciences": {1: [], 2: [], 3: [], 4: []},
            "Business & Economics": {1: [], 2: [], 3: [], 4: []},
            "Arts & Humanities": {1: [], 2: [], 3: [], 4: []}
        }
    
    # Load the dataset
    df = generate_dataset()
    
    # Header
    st.markdown('<div class="main-header">National Poster Presentation Event Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for filters
    st.sidebar.title("Filters")
    
    # Track filter
    selected_track = st.sidebar.selectbox("Select Track", ["All"] + list(df['Track'].unique()))
    
    # State filter
    selected_state = st.sidebar.selectbox("Select State", ["All"] + list(df['State'].unique()))
    
    # College filter
    selected_college = st.sidebar.selectbox("Select College", ["All"] + list(df['College'].unique()))
    
    # Day filter
    selected_day = st.sidebar.selectbox("Select Day", ["All"] + [f"Day {i}" for i in range(1, 5)])
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_track != "All":
        filtered_df = filtered_df[filtered_df['Track'] == selected_track]
    
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    if selected_college != "All":
        filtered_df = filtered_df[filtered_df['College'] == selected_college]
    
    if selected_day != "All":
        day_num = int(selected_day.split(" ")[1])
        filtered_df = filtered_df[filtered_df['Day'] == day_num]
    
    # Dashboard Tabs
    tabs = st.tabs(["Overview", "Participation Analysis", "Feedback Analysis", "Image Gallery", "Image Processing"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown('<div class="sub-header">Event Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Total Participants", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Number of Tracks", len(df['Track'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Number of Colleges", len(df['College'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Average Rating", round(df['Rating'].mean(), 2))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display filtered data
        st.markdown('<div class="sub-header">Filtered Participant Data</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df, use_container_width=True)
    
    # Tab 2: Participation Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">Participation Analysis</div>', unsafe_allow_html=True)
        
        # Chart 1: Track-wise Participation
        st.markdown("### Track-wise Participation")
        fig1, ax1 = plt.subplots(figsize=(7, 4))  # Reduced from (10, 6)
        track_counts = df['Track'].value_counts()
        ax1.pie(track_counts, labels=track_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        
        # Chart 2: Day-wise Participation
        st.markdown("### Day-wise Participation")
        fig2, ax2 = plt.subplots(figsize=(7, 4))  # Reduced from (10, 6)
        day_counts = df['Day'].value_counts().sort_index()
        sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax2)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Number of Participants')
        ax2.set_title('Participation by Day')
        st.pyplot(fig2)
        
        # Chart 3: College-wise Participation (Top 10)
        st.markdown("### College-wise Participation (Top 10)")
        fig3, ax3 = plt.subplots(figsize=(8, 4))  # Reduced from (12, 6)
        college_counts = df['College'].value_counts().nlargest(10)
        sns.barplot(x=college_counts.values, y=college_counts.index, ax=ax3)
        ax3.set_xlabel('Number of Participants')
        ax3.set_ylabel('College')
        ax3.set_title('Top 10 Colleges by Participation')
        plt.tight_layout()  # Added to ensure labels are visible
        st.pyplot(fig3)
        
        # Chart 4: State-wise Participation
        st.markdown("### State-wise Participation")
        fig4, ax4 = plt.subplots(figsize=(8, 4))  # Reduced from (12, 6)
        state_counts = df['State'].value_counts()
        sns.barplot(x=state_counts.values, y=state_counts.index, ax=ax4)
        ax4.set_xlabel('Number of Participants')
        ax4.set_ylabel('State')
        ax4.set_title('Participation by State')
        plt.tight_layout()  # Added to ensure labels are visible
        st.pyplot(fig4)
        
        # Chart 5: Age Distribution
        st.markdown("### Age Distribution of Participants")
        fig5, ax5 = plt.subplots(figsize=(7, 4))  # Reduced from (10, 6)
        sns.histplot(df['Age'], bins=15, kde=True, ax=ax5)
        ax5.set_xlabel('Age')
        ax5.set_ylabel('Count')
        ax5.set_title('Age Distribution of Participants')
        st.pyplot(fig5)
        
        # Chart 6: Track Distribution by Day
        st.markdown("### Track Distribution by Day")
        fig6, ax6 = plt.subplots(figsize=(8, 5))  # Reduced from (12, 8)
        track_day_counts = pd.crosstab(df['Track'], df['Day'])
        track_day_counts.plot(kind='bar', stacked=True, ax=ax6)
        ax6.set_xlabel('Track')
        ax6.set_ylabel('Number of Participants')
        ax6.set_title('Track Distribution by Day')
        plt.tight_layout()  # Added to ensure labels are visible
        st.pyplot(fig6)
    
    # Tab 3: Feedback Analysis
    with tabs[2]:
        st.markdown('<div class="sub-header">Feedback Analysis</div>', unsafe_allow_html=True)
        
        # Select track for feedback analysis
        track_for_feedback = st.selectbox("Select Track for Feedback Analysis", list(df['Track'].unique()))
        
        # Generate WordCloud for the selected track
        track_feedback = df[df['Track'] == track_for_feedback]['Feedback'].str.cat(sep=' ')
        
        st.markdown("### Feedback Word Cloud")
        wordcloud = WordCloud(width=600, height=300, background_color='white', max_words=100).generate(track_feedback)  # Reduced from (800, 400)
        
        fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(7, 4))  # Reduced from (10, 5)
        ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
        ax_wordcloud.axis('off')
        st.pyplot(fig_wordcloud)
        
        # Text Similarity Analysis
        st.markdown("### Feedback Similarity Analysis")
        similarity_plot = text_similarity_analysis(df, track_for_feedback)
        if isinstance(similarity_plot, str):
            st.write(similarity_plot)
        else:
            st.pyplot(similarity_plot)
        
        # Rating Distribution
        st.markdown("### Rating Distribution")
        fig_rating, ax_rating = plt.subplots(figsize=(7, 4))  # Reduced from (10, 6)
        sns.countplot(x='Rating', data=df[df['Track'] == track_for_feedback], ax=ax_rating)
        ax_rating.set_xlabel('Rating')
        ax_rating.set_ylabel('Count')
        ax_rating.set_title(f'Rating Distribution for {track_for_feedback} Track')
        st.pyplot(fig_rating)
    
    # Tab 4: Image Gallery with manual upload
    with tabs[3]:
        st.markdown('<div class="sub-header">Image Gallery</div>', unsafe_allow_html=True)
        
        # Image upload section
        st.markdown("### Add Images to Gallery")
        
        upload_col1, upload_col2 = st.columns(2)
        
        with upload_col1:
            upload_track = st.selectbox("Select Track", list(df['Track'].unique()), key="upload_track")
        
        with upload_col2:
            upload_day = st.selectbox("Select Day", [1, 2, 3, 4], key="upload_day")
        
        uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        
        if uploaded_files:
            if st.button("Add Images to Gallery"):
                for uploaded_file in uploaded_files:
                    # Read and process the image
                    image = Image.open(uploaded_file)
                    
                    # Convert to base64 for storage
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Add to the session state
                    st.session_state.track_day_images[upload_track][upload_day].append({
                        "image": img_str,
                        "caption": uploaded_file.name
                    })
                
                st.success(f"{len(uploaded_files)} images added to {upload_track} - Day {upload_day}!")
        
        # Image gallery view
        st.markdown("### Browse Gallery")
        
        gallery_col1, gallery_col2 = st.columns(2)
        
        with gallery_col1:
            view_track = st.selectbox("Select Track to View", list(df['Track'].unique()), key="view_track")
        
        with gallery_col2:
            view_day = st.selectbox("Select Day to View", [1, 2, 3, 4], key="view_day")
        
        # Display images for the selected track and day
        images = st.session_state.track_day_images[view_track][view_day]
        
        if not images:
            st.info(f"No images available for {view_track} - Day {view_day}. Upload some images first!")
        else:
            st.markdown(f"### Images for {view_track} - Day {view_day}")
            
            # Create a grid for images (3 columns)
            cols = st.columns(3)
            
            for i, img_data in enumerate(images):
                with cols[i % 3]:
                    st.image(
                        f"data:image/png;base64,{img_data['image']}",
                        caption=img_data['caption'],
                        use_column_width=True
                    )
                    
                    # Add delete button for each image
                    if st.button(f"Delete", key=f"delete_{view_track}_{view_day}_{i}"):
                        st.session_state.track_day_images[view_track][view_day].pop(i)
                        st.experimental_rerun()
    
    # Tab 5: Image Processing
    with tabs[4]:
        st.markdown('<div class="sub-header">Image Processing</div>', unsafe_allow_html=True)
        
        # Upload image for processing
        st.markdown("### Process Images")
        uploaded_process_file = st.file_uploader("Upload an image for processing", type=["jpg", "jpeg", "png"], key="process_image")
        
        if uploaded_process_file is not None:
            # Load the image
            image = Image.open(uploaded_process_file)
            
            # Display the original image
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image processing options
            st.markdown("### Apply Image Processing")
            
            # Select filter
            filter_name = st.selectbox("Select Filter", ["None", "Grayscale", "Sepia", "Blur", "Sharpen", "Enhance"])
            
            if filter_name != "None":
                # Apply the selected filter
                processed_image = apply_image_filter(image, filter_name)
                
                # Display the processed image
                st.image(processed_image, caption=f"Processed Image with {filter_name} Filter", use_column_width=True)
                
                # Option to download the processed image
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Download Processed Image",
                        data=byte_im,
                        file_name=f"processed_{uploaded_process_file.name}_{filter_name}.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Add processed image to a gallery
                    if st.button("Add to Gallery"):
                        # Select track and day
                        save_track = st.selectbox("Select Track to Save To", list(df['Track'].unique()))
                        save_day = st.selectbox("Select Day to Save To", [1, 2, 3, 4])
                        
                        # Convert to base64 for storage
                        buffered = io.BytesIO()
                        processed_image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Add to the session state
                        st.session_state.track_day_images[save_track][save_day].append({
                            "image": img_str,
                            "caption": f"Processed {uploaded_process_file.name} - {filter_name}"
                        })
                        
                        st.success(f"Image added to {save_track} - Day {save_day}!")
        else:
            st.info("Upload an image to apply processing filters.")
            
            # Show some image processing examples
            st.markdown("### Image Processing Examples")
            st.markdown("""
            Available filters:
            - **Grayscale**: Converts the image to black and white
            - **Sepia**: Applies a vintage sepia tone effect
            - **Blur**: Applies Gaussian blur to the image
            - **Sharpen**: Enhances the edges in the image
            - **Enhance**: Increases the contrast of the image
            """)

if __name__ == "__main__":
    main()