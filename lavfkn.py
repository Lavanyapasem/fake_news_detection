import streamlit as st
import google.generativeai as genai
import requests
from urllib.parse import urlparse
import re
import time
from datetime import datetime
import pandas as pd
import os
from bs4 import BeautifulSoup
import json

# Configure the page
st.set_page_config(
    page_title="Fake News Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure Gemini API with the loaded key
genai.configure(api_key=GEMINI_API_KEY)
# API Key configuration section
if not GEMINI_API_KEY:
    st.warning("🔑 Gemini API Key Required")
    st.markdown("""
    To use this application, you need to provide a Google Gemini API key. You can:
    
    1. **Set environment variable:** `export GEMINI_API_KEY=your_api_key_here`
    2. **Use Streamlit secrets:** Add to `.streamlit/secrets.toml`
    3. **Enter below for testing:** (Not recommended for production)
    """)
    
    # Allow manual input for testing
    manual_key = st.text_input(
        "Enter your Gemini API Key:", 
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    if manual_key:
        GEMINI_API_KEY = manual_key
        st.success("✅ API Key provided! You can now use the application.")
    else:
        st.info("👆 Please provide your API key to continue")
        st.stop()

# Configure Gemini with the API key
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {str(e)}")
    st.stop()

# Initialize the model


@st.cache_resource
def load_model():
    try:
        # Updated model names for current Gemini API
        # Try the latest models first, fallback to older ones if needed
        model_names = [
            'gemini-1.5-flash',      # Latest and fastest
            'gemini-1.5-pro',        # More capable but slower
            'gemini-1.0-pro',        # Fallback option
            'models/gemini-1.5-flash',  # With models/ prefix
            'models/gemini-1.5-pro',
            'models/gemini-1.0-pro'
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model with a simple request
                test_response = model.generate_content("Test")
                st.success(f"✅ Successfully loaded model: {model_name}")
                return model
            except Exception as e:
                st.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        # If all models fail, raise an error
        raise Exception("No available models found")
        
    except Exception as e:
        st.error(f"Error loading any model: {str(e)}")
        return None

# Also add this function to check available models
def check_available_models():
    """Check and display available models"""
    try:
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        return available_models
    except Exception as e:
        st.error(f"Error checking available models: {str(e)}")
        return []

# Add this to your sidebar for debugging
if st.sidebar.button("🔍 Check Available Models"):
    with st.spinner("Checking available models..."):
        models = check_available_models()
        if models:
            st.sidebar.success("Available models:")
            for model in models:
                st.sidebar.write(f"- {model}")
        else:
            st.sidebar.error("No models available or error occurred")

model = load_model()

if not model:
    st.error("Failed to load the Gemini model. Please check your API key.")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .result-box {
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
        border-left: 2px solid;
    }
    .fake-news {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .real-news {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
        color: #2e7d32;
    }
    .uncertain {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        color: #f57c00;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1em;
        border-radius: 8px;
        text-align: center;
    }
    .source-info {
        background-color: #f8f9fa;
        padding: 1em;
        border-radius: 8px;
        margin: 1em 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1em;
        margin: 1em 0;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1em;
        border-radius: 8px;
        margin: 1em 0;
    }
    .api-key-section {
        background-color: #e3f2fd;
        padding: 1em;
        border-radius: 8px;
        margin: 1em 0;
        border: 1px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Fake News Detective</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered news verification system</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🛠 Detection Tools")
detection_method = st.sidebar.radio(
    "Choose detection method:",
    ["📝 Analyze Text", "🔗 Check URL", "📊 Batch Analysis", "🎯 Advanced Analysis"]
)

# Enhanced helper functions
def extract_text_from_url(url):
    """Enhanced text extraction from URL using BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Use BeautifulSoup for better HTML parsing
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Extract text from common article containers
        article_selectors = [
            'article', '.article-content', '.post-content', 
            '.entry-content', '.content', '#content', 'main'
        ]
        
        text = ""
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                text = elements[0].get_text()
                break
        
        # Fallback to body text if no article container found
        if not text:
            text = soup.get_text()
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Get page title
        title = soup.title.string if soup.title else "No title"
        
        return {
            'text': text[:8000],  # Increased limit
            'title': title,
            'length': len(text)
        }
    except Exception as e:
        return {'error': f"Error extracting text: {str(e)}"}

def analyze_news_enhanced(text, title="", url=""):
    """Enhanced news analysis with more detailed prompts"""
    try:
        prompt = f"""
        As an expert fact-checker and media analyst, please analyze this news content comprehensively:
        
        Title: {title}
        URL: {url}
        Content: {text[:4000]}
        
        Please provide a detailed analysis including:
        
        1. CREDIBILITY SCORE (0-100): Where 100 is completely credible
        2. CLASSIFICATION: Real News, Fake News, Misleading, or Uncertain
        3. BIAS ANALYSIS: Political lean, emotional manipulation, loaded language
        4. FACTUAL ASSESSMENT: Verifiable claims, sources cited, evidence provided
        5. WRITING QUALITY: Grammar, style, professionalism
        6. RED FLAGS: Clickbait headlines, sensationalism, lack of sources
        7. POSITIVE INDICATORS: Credible sources, balanced reporting, fact-checking
        8. RECOMMENDATIONS: Actions readers should take
        
        Be specific and provide concrete examples from the text.
        Format your response with clear section headers.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing text: {str(e)}"
def parse_analysis(analysis):
    """Extract the credibility score from section 1 and classification from section 2."""
    try:
        # Extract section 1 where credibility score is typically placed
        section_1_match = re.search(r'1\..*?(?=\n\d+\.)', analysis, re.DOTALL)
        section_1 = section_1_match.group(0).strip() if section_1_match else analysis

        # Look for credibility score in section 1
        score_match = re.search(r'(?i)credibility\s*score\s*\(?.*?\)?:?\s*(\d{1,3})', section_1)
        score = int(score_match.group(1)) if score_match and 0 <= int(score_match.group(1)) <= 100 else 50

        # Extract classification from the full analysis
        classification_match = re.search(r'(?i)(?:classification|category)\s*[:\-]?\s*(.*)', analysis)
        classification_raw = classification_match.group(1).strip().split('\n')[0] if classification_match else "Uncertain"

        # Normalize classification
        if 'fake' in classification_raw.lower():
            classification = "Fake News"
        elif 'real' in classification_raw.lower():
            classification = "Real News"
        elif 'misleading' in classification_raw.lower():
            classification = "Misleading"
        else:
            classification = "Uncertain"

        return {
            'score': score,
            'classification': classification,
            'full_analysis': analysis
        }

    except Exception as e:
        return {
            'score': 50,
            'classification': "Uncertain",
            'full_analysis': analysis
        }


def display_analysis_results(analysis_data, source_info=None):
    """Display analysis results in a structured format"""
    score = analysis_data['score']
    classification = analysis_data['classification']
    full_analysis = analysis_data['full_analysis']
    
    # Display source information if provided
    if source_info:
        st.subheader("📰 Source Information")
        st.markdown(f"""
        <div class="source-info">
            <strong>Title:</strong> {source_info.get('title', 'N/A')}<br>
            <strong>URL:</strong> <a href="{source_info.get('url', '#')}" target="_blank">{source_info.get('url', 'N/A')}</a><br>
            <strong>Content Length:</strong> {source_info.get('length', 'N/A')} characters<br>
            <strong>Analysis Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """, unsafe_allow_html=True)
    
    # Display metrics
    st.subheader("📊 Analysis Results")
    
    
    # Result box with appropriate styling
    if classification == "Fake News":
        st.markdown(f'<div class="result-box fake-news"><h3>⚠ {classification}</h3><p>This content shows significant indicators of being unreliable or false.</p></div>', unsafe_allow_html=True)
    elif classification == "Real News":
        st.markdown(f'<div class="result-box real-news"><h3>✅ {classification}</h3><p>This content appears to be credible and well-sourced.</p></div>', unsafe_allow_html=True)
    elif classification == "Misleading":
        st.markdown(f'<div class="result-box uncertain"><h3>⚠ {classification}</h3><p>This content may contain misleading information or bias.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box uncertain"><h3>❓ {classification}</h3><p>The authenticity of this content cannot be determined with confidence.</p></div>', unsafe_allow_html=True)
    
    # Detailed analysis sections
    st.subheader("🔍 Detailed Analysis")
    
    # Split analysis into sections
    sections = re.split(r'\n(?=\d+\.|\*\*|\#)', full_analysis)
    
    for i, section in enumerate(sections):
        if section.strip():
            # Create expandable sections for different analysis parts
            section_title = section.split('\n')[0][:50] + "..." if len(section.split('\n')[0]) > 50 else section.split('\n')[0]
            
            with st.expander(f"Analysis Section {i+1}: {section_title}"):
                st.write(section)

# Main content based on detection method
if detection_method == "📝 Analyze Text":
    st.header("📝 Text Analysis")
    
    # Text input with better UX
    news_text = st.text_area(
        "Enter the news text you want to analyze:",
        height=200,
        placeholder="Paste your news article text here...",
        help="You can paste the full article text or just the main content"
    )
    
    # Optional title input
    news_title = st.text_input(
        "Article Title (optional):",
        placeholder="Enter the headline or title of the article"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    with col2:
        quick_check = st.button("⚡ Quick Check")
    
    if (analyze_button or quick_check) and news_text:
        analysis_type = "quick" if quick_check else "full"
        
        with st.spinner(f"Performing {'quick' if quick_check else 'comprehensive'} analysis..."):
            if analysis_type == "quick":
                # Quick analysis with simpler prompt
                analysis = analyze_news_enhanced(news_text[:2000], news_title)
            else:
                # Full analysis
                analysis = analyze_news_enhanced(news_text, news_title)
            
            analysis_data = parse_analysis(analysis)
            display_analysis_results(analysis_data)

elif detection_method == "🔗 Check URL":
    st.header("🔗 URL Analysis")
    
    # URL input with validation
    url = st.text_input(
        "Enter the URL of the news article:",
        placeholder="https://example.com/news-article",
        help="Make sure the URL is complete and accessible"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        check_button = st.button("🔍 Check URL", type="primary")
    
    if check_button and url:
        if not url.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Extracting and analyzing content..."):
                # Extract content from URL
                extraction_result = extract_text_from_url(url)
                
                if 'error' in extraction_result:
                    st.error(extraction_result['error'])
                else:
                    # Analyze the extracted content
                    analysis = analyze_news_enhanced(
                        extraction_result['text'], 
                        extraction_result['title'], 
                        url
                    )
                    
                    analysis_data = parse_analysis(analysis)
                    
                    # Prepare source info
                    source_info = {
                        'title': extraction_result['title'],
                        'url': url,
                        'length': extraction_result['length']
                    }
                    
                    display_analysis_results(analysis_data, source_info)

elif detection_method == "📊 Batch Analysis":
    st.header("📊 Batch Analysis")
    
    st.info("Upload multiple news articles or URLs for comprehensive batch analysis")
    
    # File upload options
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload file (TXT, CSV, or PDF):",
            type=['txt', 'csv', 'pdf'],

            help="Upload a text file with articles or URLs (one per line)"
        )
    
    with col2:
        url_list = st.text_area(
            "Or enter URLs here:",
            height=100,
            placeholder="https://example1.com/article1\nhttps://example2.com/article2",
            help="Enter one URL per line"
        )
    
    # Analysis options
    st.subheader("Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Quick", "Standard", "Comprehensive"]
        )
    
    with col2:
        export_results = st.checkbox("Export results to CSV")
    
    if st.button("🔍 Analyze Batch", type="primary"):
        items_to_analyze = []
        
        # Process uploaded file
        import fitz  # PyMuPDF

        if uploaded_file:
            if uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode('utf-8', errors='replace')
                items_to_analyze.extend([line.strip() for line in content.split('\n') if line.strip()])
    
            elif uploaded_file.type == "application/pdf":
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    pdf_text = ""
                    for page in doc:
                        pdf_text += page.get_text()
                items_to_analyze.extend([line.strip() for line in pdf_text.split('\n') if line.strip()])
    
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                if 'url' in df.columns:
                    items_to_analyze.extend(df['url'].tolist())
                else:
                    st.error("CSV file must contain a 'url' column")

        
        # Process URL list
        if url_list:
            items_to_analyze.extend([line.strip() for line in url_list.split('\n') if line.strip()])
        
        if items_to_analyze:
            st.subheader(f"Processing {len(items_to_analyze)} items...")
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, item in enumerate(items_to_analyze):
                status_text.text(f"Processing item {i+1}/{len(items_to_analyze)}: {item[:50]}...")
                
                # Update progress
                progress_bar.progress((i + 1) / len(items_to_analyze))
                
                try:
                    # Determine if it's a URL or text
                    if item.startswith(('http://', 'https://')):
                        # Extract from URL
                        extraction_result = extract_text_from_url(item)
                        if 'error' in extraction_result:
                            results.append({
                                'Source': item,
                                'Status': 'Error',
                                'Error': extraction_result['error'],
                                'Classification': 'N/A',
                                'Score': 0
                            })
                            continue
                        
                        text = extraction_result['text']
                        title = extraction_result['title']
                        source = item
                    else:
                        # It's direct text
                        text = item
                        title = ""
                        source = "Direct text"
                    
                    # Analyze based on depth setting
                    if analysis_depth == "Quick":
                        analysis = analyze_news_enhanced(text[:1000], title)
                    elif analysis_depth == "Standard":
                        analysis = analyze_news_enhanced(text[:3000], title)
                    else:  # Comprehensive
                        analysis = analyze_news_enhanced(text, title)
                    
                    analysis_data = parse_analysis(analysis)
                    
                    results.append({
                        'Source': source,
                        'Title': title,
                        'Classification': analysis_data['classification'],
                        'Score': analysis_data['score'],
                        'Status': 'Analyzed',
                        'Analysis': analysis_data['full_analysis'][:500] + "..." if len(analysis_data['full_analysis']) > 500 else analysis_data['full_analysis']
                    })
                    
                except Exception as e:
                    results.append({
                        'Source': item,
                        'Status': 'Error',
                        'Error': str(e),
                        'Classification': 'N/A',
                        'Score': 0
                    })
                
                # Rate limiting
                time.sleep(2)
            
            # Display results
            status_text.text("Analysis complete!")
            st.subheader("📊 Batch Analysis Results")
            
            # Summary metrics
            successful_analyses = [r for r in results if r['Status'] == 'Analyzed']
            total_items = len(results)
            success_rate = len(successful_analyses) / total_items * 100 if total_items > 0 else 0
            
            fake_count = sum(1 for r in successful_analyses if r['Classification'] == 'Fake News')
            real_count = sum(1 for r in successful_analyses if r['Classification'] == 'Real News')
            misleading_count = sum(1 for r in successful_analyses if r['Classification'] == 'Misleading')
            uncertain_count = len(successful_analyses) - fake_count - real_count - misleading_count
            
            # Display summary
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Items", total_items)
            with col2:
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col3:
                st.metric("Fake News", fake_count)
            with col4:
                st.metric("Real News", real_count)
            with col5:
                st.metric("Misleading", misleading_count)
            
            # Create DataFrame for display
            df_results = pd.DataFrame(results)
            
            # Display interactive table
            st.subheader("Detailed Results")
            st.dataframe(df_results, use_container_width=True)
            
            # Export option
            if export_results and successful_analyses:
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Individual result expandable sections
            st.subheader("Individual Analysis Details")
            for i, result in enumerate(results):
                if result['Status'] == 'Analyzed':
                    with st.expander(f"Result {i+1}: {result['Classification']} (Score: {result['Score']})"):
                        st.write(f"**Source:** {result['Source']}")
                        st.write(f"**Title:** {result.get('Title', 'N/A')}")
                        st.write(f"**Classification:** {result['Classification']}")
                        st.write(f"**Score:** {result['Score']}/100")
                        st.write(f"**Analysis:** {result['Analysis']}")
                else:
                    with st.expander(f"Result {i+1}: ERROR"):
                        st.error(f"Failed to analyze: {result.get('Error', 'Unknown error')}")
        else:
            st.warning("Please upload a file or enter URLs to analyze")

elif detection_method == "🎯 Advanced Analysis":
    st.header("🎯 Advanced Analysis")
    st.info("Compare multiple sources and perform cross-verification")

    # Multiple source comparison
    st.subheader("Cross-Source Verification")
    st.markdown("Enter up to 3 source URLs for cross-verification:")

    source1 = st.text_input("Source 1 URL", key="source_1")
    source2 = st.text_input("Source 2 URL", key="source_2")
    source3 = st.text_input("Source 3 URL", key="source_3")

    sources = [url for url in [source1, source2, source3] if url.strip() != ""]

    if st.button("🔍 Compare Sources") and sources:
        st.subheader("Cross-Verification Results")
        source_results = []

        for i, source in enumerate(sources):
            with st.spinner(f"Analyzing source {i+1}..."):
                try:
                    extraction_result = extract_text_from_url(source)

                    if 'error' in extraction_result:
                        source_results.append({
                            'url': source,
                            'title': 'N/A',
                            'score': 0,
                            'classification': 'Error',
                            'analysis': extraction_result['error']
                        })
                        continue

                    analysis = analyze_news_enhanced(
                        extraction_result['text'],
                        extraction_result['title'],
                        source
                    )

                    analysis_data = parse_analysis(analysis)

                    source_results.append({
                        'url': source,
                        'title': extraction_result['title'],
                        'score': analysis_data['score'],
                        'classification': analysis_data['classification'],
                        'analysis': analysis_data['full_analysis']
                    })

                except Exception as e:
                    source_results.append({
                        'url': source,
                        'title': 'N/A',
                        'score': 0,
                        'classification': 'Error',
                        'analysis': str(e)
                    })

        # Display comparison
        if source_results:
            comparison_df = pd.DataFrame([{
                'Source': f"Source {i+1}",
                'Title': result['title'][:50] + "..." if result['title'] != 'N/A' and len(result['title']) > 50 else result['title'],
                'Classification': result['classification'],
                'Score': result['score'],
                'URL': result['url']
            } for i, result in enumerate(source_results)])

            st.dataframe(comparison_df, use_container_width=True)

            # Calculate consensus only for successful classifications
            successful = [r for r in source_results if r['classification'] not in ['Error']]
            if successful:
                avg_score = sum(r['score'] for r in successful) / len(successful)
                classifications = [r['classification'] for r in successful]
                consensus = max(set(classifications), key=classifications.count)
                agreement = classifications.count(consensus) / len(classifications) * 100
            else:
                avg_score, consensus, agreement = 0, "No consensus", 0

            st.subheader("Consensus Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{avg_score:.1f}/100")
            with col2:
                st.metric("Consensus", consensus)
            with col3:
                st.metric("Agreement", f"{agreement:.1f}%")

            # Individual source details
            for i, result in enumerate(source_results):
                with st.expander(f"Source {i+1} Details"):
                    st.write(f"**URL:** {result['url']}")
                    st.write(f"**Title:** {result['title']}")
                    st.write(f"**Classification:** {result['classification']}")
                    st.write(f"**Score:** {result['score']}/100")
                    st.write("**Analysis:**")
                    st.write(result['analysis'])


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔍 Enhanced Fake News Detective | Powered by Google Gemini AI</p>
    <p><small>⚠ This tool provides AI-assisted analysis. Always verify important news through multiple credible sources.</small></p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ Analysis Methods")
st.sidebar.markdown("""
**Text Analysis:** Direct text input with optional title
**URL Check:** Automatic content extraction and analysis
**Batch Analysis:** Process multiple items with export options
**Advanced Analysis:** Cross-source verification and comparison

**AI Analysis Factors:**
- Source credibility and reputation
- Writing quality and professionalism
- Factual accuracy and citations
- Emotional manipulation detection
- Bias and political lean analysis
- Clickbait and sensationalism indicators
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Settings")
api_status = "✅ Connected" if GEMINI_API_KEY else "❌ Not configured"
st.sidebar.text(f"API Status: {api_status}")

# Display API key setup instructions
if st.sidebar.button("🔑 API Key Setup"):
    st.sidebar.markdown("""
    **How to get your Gemini API Key:**
    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Create a new API key
    3. Copy the key
    
    **How to set it up:**
    - **Environment variable:** `export GEMINI_API_KEY=your_key`
    - **Streamlit secrets:** Add to `.streamlit/secrets.toml`
    - **Manual input:** Use the input field above
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("⚠ Important Notes")
st.sidebar.markdown("""
- This tool uses AI analysis and should not be the sole determinant of news authenticity
- Always cross-reference with multiple reliable sources
- Be aware of your own biases when interpreting results
- Consider the context and timing of news articles
- Report suspicious content to appropriate authorities
""")

# Add analysis history (optional - would require database integration)
if st.sidebar.button("📊 View Analysis History"):
    st.sidebar.info("Analysis history feature would require database integration")