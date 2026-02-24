cat > streamlit_app.py << 'EOF'
import streamlit as st
st.set_page_config(page_title="CORSIA Dashboard", layout="wide")
st.title("CORSIA Dashboard")
st.write("Streamlit running ✅")
EOF
