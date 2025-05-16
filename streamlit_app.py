import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pos_analysis_model import POSAnalysisModel
import base64
from io import BytesIO

# Store analysis results that will be used in the UI
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

def get_image_download_link(img_path, filename="image.png"):
    """Generate a download link for an image file"""
    with open(img_path, "rb") as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def display_visualization(img_path, caption=None, download=True):
    """Display a visualization image with optional caption and download link"""
    st.image(img_path, caption=caption, use_container_width=True)
    if download:
        st.markdown(get_image_download_link(img_path, os.path.basename(img_path)), unsafe_allow_html=True)

# Add this new function to ensure proper data types for PyArrow compatibility
def fix_dataframe_for_arrow(df):
    """
    Fix common issues with DataFrame types that cause PyArrow conversion errors.
    """
    df_fixed = df.copy()
    
    # Ensure all date columns are properly formatted as datetime objects
    date_columns = [col for col in df_fixed.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in df_fixed.columns:
            # Convert to datetime with a standard format
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col])
            except:
                # If conversion fails, keep as string/object
                df_fixed[col] = df_fixed[col].astype(str)
    
    # Ensure ID columns are strings, not mixed types
    id_columns = [col for col in df_fixed.columns if 'id' in col.lower()]
    for col in id_columns:
        if col in df_fixed.columns:
            df_fixed[col] = df_fixed[col].astype(str)
    
    # Convert any columns with mixed types to strings
    for col in df_fixed.columns:
        if df_fixed[col].dtype.name == 'object':
            # Check if the column contains mixed types
            try:
                pd.to_numeric(df_fixed[col])
            except:
                # If conversion to numeric fails, it's likely mixed - convert to string
                df_fixed[col] = df_fixed[col].astype(str)
    
    return df_fixed

def main():
    # Set page config - this must be the first Streamlit command
    st.set_page_config(
        page_title="POS Terminal Analysis Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: #ff7800;
        border-radius: 4px 4px 0px 0px;
        gap: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff7800;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("POS Terminal Analysis Dashboard")
    st.markdown("Interactive dashboard for analyzing POS terminal transactions across customer segments")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File uploader
    st.sidebar.header("Upload Data")
    transactions_file = st.sidebar.file_uploader("Transaction Data (CSV)", type=['csv'])
    customers_file = st.sidebar.file_uploader("Customer Data (CSV)", type=['csv'])
    
    # Save uploaded files temporarily
    temp_transactions_path = None
    temp_customers_path = None
    
    if transactions_file is not None:
        temp_transactions_path = "temp_transactions.csv"
        with open(temp_transactions_path, "wb") as f:
            f.write(transactions_file.getbuffer())
    
    if customers_file is not None:
        temp_customers_path = "temp_customers.csv"
        with open(temp_customers_path, "wb") as f:
            f.write(customers_file.getbuffer())
    
    # Model parameters
    st.sidebar.header("Analysis Parameters")
    output_dir = st.sidebar.text_input("Output Directory", value="output")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "📈 Transaction Analysis", 
        "👥 Customer Segmentation", 
        "🏢 Division Analysis", 
        "🔄 POS Adoption", 
        "📝 Reports"
    ])
    
    # Initialize session state if not already
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Run Analysis Button
    if st.sidebar.button("Run Analysis"):
        if transactions_file is None or customers_file is None:
            st.sidebar.error("Please upload both transaction and customer data files.")
        else:
            with st.spinner("Running analysis..."):
                try:
                    # Create output directories
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
                    
                    # Create and run model
                    model = POSAnalysisModel(
                        transactions_path=temp_transactions_path,
                        customers_path=temp_customers_path,
                        output_dir=output_dir
                    )
                    
                    # Store model in session state
                    st.session_state.model = model
                    
                    # Run analysis
                    result = model.run_analysis()
                    if result == "Analysis completed successfully":
                        st.session_state.analysis_complete = True
                        st.sidebar.success("Analysis completed successfully!")
                    else:
                        st.sidebar.error(f"Error: {result}")
                
                except Exception as e:
                    st.sidebar.error(f"An error occurred: {str(e)}")
    
    # Fill in content for each tab
    if not st.session_state.analysis_complete:
        # Display placeholder content if analysis is not yet complete
        with tab1:
            st.header("Dashboard Overview")
            st.info("Please upload data files and run the analysis to view dashboard content.")
            
            # Display placeholder metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Transactions", value="0")
            with col2:
                st.metric(label="Total Customers", value="0")
            with col3:
                st.metric(label="Total Revenue", value="₦0")
        
        with tab2:
            st.header("Transaction Analysis")
            st.info("Run analysis to view transaction insights.")
        
        with tab3:
            st.header("Customer Segmentation")
            st.info("Run analysis to view customer segments.")
        
        with tab4:
            st.header("Division Analysis")
            st.info("Run analysis to view division data.")
        
        with tab5:
            st.header("POS Adoption")
            st.info("Run analysis to view POS adoption metrics.")
        
        with tab6:
            st.header("Reports")
            st.info("Reports will be generated after analysis completes.")
    
    else:
        # Display actual analysis results if analysis is complete
        model = st.session_state.model
        
        # Get key metrics from the analysis
        adoption_data = model.calculate_pos_adoption()
        subsector_volume = model.analyze_value_by_category('subsector_name')
        segment_volume = model.analyze_value_by_category('segment')
        division_data = model.get_top_bottom_divisions()
        
        with tab1:
            st.header("Dashboard Overview")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Transactions", value=f"{len(model.transactions_df):,}")
            with col2:
                st.metric(label="Total Customers", value=f"{len(model.customers_df):,}")
            with col3:
                total_revenue = model.transactions_df['amount'].sum()
                st.metric(label="Total Revenue", value=f"₦{total_revenue:,.0f}")
            
            # Adoption rate
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="POS Adoption Rate", 
                         value=f"{adoption_data['adoption_percentage']:.1f}%",
                         delta=f"{adoption_data['customers_with_pos']:,} customers")
            
            # Top subsector by value
            top_subsector = subsector_volume.sort_values('amount', ascending=False).iloc[0]
            with col2:
                st.metric(label="Top Subsector by Value", 
                         value=top_subsector['subsector_name'],
                         delta=f"₦{top_subsector['amount']:,.0f}")
            
            # Main visualizations
            st.subheader("Key Visualizations")
            
            col1, col2 = st.columns(2)
            with col1:
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'volume_by_subsector.png'),
                    "Transaction Volume by Subsector"
                )
            
            with col2:
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'value_by_subsector.png'),
                    "Transaction Value by Subsector"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'pos_adoption_rate.png'),
                    "POS Terminal Adoption Rate"
                )
            
            with col2:
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'customer_value_segmentation.png'),
                    "Customer Value Segmentation"
                )
        
        with tab2:
            st.header("Transaction Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Volume Analysis")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'volume_by_subsector.png'),
                    "By Subsector"
                )
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'volume_by_segment.png'),
                    "By Segment"
                )
            
            with col2:
                st.subheader("Value Analysis")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'value_by_subsector.png'),
                    "By Subsector"
                )
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'value_by_segment.png'),
                    "By Segment"
                )
            
            st.subheader("Transaction Data Summary")
            # Apply fix to ensure Arrow compatibility
            fixed_transactions_df = fix_dataframe_for_arrow(model.transactions_df)
            st.dataframe(fixed_transactions_df.describe())
        
        with tab3:
            st.header("Customer Segmentation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Customer Value Segments")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'customer_value_segmentation.png'),
                    "Customer Value Segmentation"
                )
                
                # Show segment counts
                if model.monthly_customer_values is not None:
                    segment_counts = model.monthly_customer_values.groupby('segment').size().reset_index(name='count')
                    # Apply fix for Arrow compatibility
                    segment_counts = fix_dataframe_for_arrow(segment_counts)
                    st.dataframe(segment_counts)
            
            with col2:
                st.subheader("High Value Customers by Location")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'high_value_by_location.png'),
                    "High Value Customers by Location"
                )
                
                st.subheader("Low Value Customers by Location")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'low_value_by_location.png'),
                    "Low Value Customers by Location"
                )
            
            # Show customer data summary
            st.subheader("Customer Data Summary")
            # Apply fix for Arrow compatibility
            fixed_customers_df = fix_dataframe_for_arrow(model.customers_df)
            st.dataframe(fixed_customers_df.describe(include='all'))
        
        with tab4:
            st.header("Division Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Divisions by Volume")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'top10_divisions_volume.png'),
                    "Top 10 Divisions by Volume"
                )
                # Apply fix for Arrow compatibility
                top_volume = fix_dataframe_for_arrow(division_data['top_volume'])
                st.dataframe(top_volume)
            
            with col2:
                st.subheader("Top Divisions by Value")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'top10_divisions_value.png'),
                    "Top 10 Divisions by Value"
                )
                # Apply fix for Arrow compatibility
                top_value = fix_dataframe_for_arrow(division_data['top_value'])
                st.dataframe(top_value)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Bottom Divisions by Volume")
                # Apply fix for Arrow compatibility
                bottom_volume = fix_dataframe_for_arrow(division_data['bottom_volume'])
                st.dataframe(bottom_volume)
            
            with col2:
                st.subheader("Bottom Divisions by Value")
                # Apply fix for Arrow compatibility
                bottom_value = fix_dataframe_for_arrow(division_data['bottom_value'])
                st.dataframe(bottom_value)
        
        with tab5:
            st.header("POS Adoption")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Overall Adoption")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'pos_adoption_rate.png'),
                    "POS Terminal Adoption Rate"
                )
                
                st.metric(label="Adoption Rate", 
                         value=f"{adoption_data['adoption_percentage']:.1f}%",
                         delta=f"{adoption_data['customers_with_pos']:,}/{adoption_data['total_customers']:,} customers")
            
            with col2:
                st.subheader("Adoption by Location")
                display_visualization(
                    os.path.join(output_dir, 'visualizations', 'pos_adoption_by_location.png'),
                    "POS Adoption by Location"
                )
            
            st.subheader("Adoption in Top Divisions")
            display_visualization(
                os.path.join(output_dir, 'visualizations', 'pos_adoption_top_divisions.png'),
                "POS Adoption in Top 10 Divisions by Volume"
            )
        
        with tab6:
            st.header("Reports")
            
            # Show generated reports
            st.subheader("Generated Reports")
            
            # Interactive dashboard
            st.markdown("### Interactive Dashboard")
            st.markdown(f"""
            [Open Interactive Dashboard]({os.path.join(output_dir, 'reports', 'interactive_dashboard.html')})
            """)
            
            # Comprehensive report
            st.markdown("### Comprehensive Insights Report")
            st.markdown(f"""
            [Open Comprehensive Report]({os.path.join(output_dir, 'reports', 'comprehensive_insights_report.html')})
            """)
            
            # Data exports
            st.subheader("Data Exports")
            
            if st.button("Export High Value Customers"):
                high_value_path = os.path.join(output_dir, 'data', 'high_value_customers.csv')
                with open(high_value_path, "rb") as f:
                    st.download_button(
                        label="Download High Value Customers CSV",
                        data=f,
                        file_name="high_value_customers.csv",
                        mime="text/csv"
                    )
            
            if st.button("Export Low Value Customers"):
                low_value_path = os.path.join(output_dir, 'data', 'low_value_customers.csv')
                with open(low_value_path, "rb") as f:
                    st.download_button(
                        label="Download Low Value Customers CSV",
                        data=f,
                        file_name="low_value_customers.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()