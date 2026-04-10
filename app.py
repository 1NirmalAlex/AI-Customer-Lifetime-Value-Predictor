import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="AI CLV Predictor", page_icon="💎", layout="wide")

st.markdown("""
    <style>
        /* Main background and font adjustments */
        .main {
            background-color: #0A0A0A;
            color: #FFFFFF;
        }
        
        /* Custom styling for the main prediction card */
        .pred-card {
            background: linear-gradient(135deg, #1E1E1E 0%, #121212 100%);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid #333333;
            box-shadow: 0px 10px 30px rgba(0, 255, 128, 0.1);
            text-align: center;
        }
        
        .pred-title {
            color: #888888;
            font-size: 1.2rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .pred-value {
            color: #00FFAA;
            font-size: 4rem;
            font-weight: 800;
            margin-top: 10px;
            text-shadow: 0px 0px 20px rgba(0, 255, 170, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD THE ML MODEL
# ==========================================
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('clv_linear_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'clv_linear_model.pkl' not found. Please run the Jupyter Notebook first.")
        st.stop()

model = load_ml_components()

# ==========================================
# 3. HEADER SECTION
# ==========================================
st.title("💎 Smart CLV Predictor")
st.markdown("### AI-Powered Customer Lifetime Value Analytics")
st.write("Enter the customer's behavioral metrics below to predict their future value to the business.")
st.divider()

# ==========================================
# 4. USER INPUT SECTION (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Input Parameters")
st.sidebar.markdown("Adjust the sliders to simulate customer behavior.")

# Inputs mapped to RFM concepts
recency = st.sidebar.slider("Recency (Days since last purchase)", min_value=0, max_value=365, value=15, step=1)
frequency = st.sidebar.slider("Frequency (Total purchases made)", min_value=1, max_value=100, value=10, step=1)
monetary = st.sidebar.number_input("Monetary (Average spend per order in Rs.)", min_value=100, max_value=100000, value=5000, step=500)

# ==========================================
# 5. PREDICTION LOGIC & DASHBOARD LAYOUT
# ==========================================
# Predict button in sidebar
if st.sidebar.button("Predict Lifetime Value", type="primary", use_container_width=True):
    
    # Format data for the model
    input_df = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    # Get prediction
    prediction = model.predict(input_df)[0]
    predicted_clv = max(0, prediction) # Ensure CLV is not negative
    
    # Layout columns
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: Value Card
    with col1:
        st.markdown(f"""
            <div class="pred-card">
                <div class="pred-title">Predicted Lifetime Value</div>
                <div class="pred-value">Rs. {predicted_clv:,.2f}</div>
                <br>
                <span style="color:#AAAAAA; font-size:0.9rem;">
                    Based on machine learning regression analysis.
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        # Business Action recommendation based on value
        st.markdown("<br>", unsafe_allow_html=True)
        if predicted_clv > 50000:
            st.success("🌟 **High-Value Customer:** Assign a VIP manager and send exclusive offers.")
        elif predicted_clv > 15000:
            st.info("📈 **Growth Potential:** Send loyalty program invites and bundle discounts.")
        else:
            st.warning("⚠️ **Low-Value Customer:** Do not spend excessive marketing budget here.")

    # RIGHT COLUMN: Plotly Gauge Chart
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_clv,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CLV Score Health", 'font': {'color': 'white', 'size': 20}},
            number = {'font': {'color': '#00FFAA'}, 'prefix': "Rs "},
            gauge = {
                'axis': {'range': [None, 100000], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00FFAA"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 15000], 'color': "rgba(255, 75, 75, 0.3)"},
                    {'range': [15000, 50000], 'color': "rgba(255, 204, 0, 0.3)"},
                    {'range': [50000, 100000], 'color': "rgba(0, 255, 170, 0.3)"}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # Default state before button is pressed
    st.info("👈 Please adjust the customer metrics in the sidebar and click **'Predict Lifetime Value'** to view the AI analysis.")

# Footer
st.divider()
st.caption("🚀 Developed for Business Analytics | Powered by Scikit-Learn & Streamlit")