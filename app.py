import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Traffic Prediction Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    """Load the XGBoost model"""
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_sample_data_from_timestamp(timestamp_obj):
    """Create sample input based on a timestamp from the dataset"""
    # Handle both string and Timestamp objects
    if isinstance(timestamp_obj, str):
        dt = datetime.strptime(timestamp_obj, '%d/%m/%Y %H:%M')
    else:
        # timestamp_obj is already a pandas Timestamp, use it directly
        dt = timestamp_obj
    
    return {
        'month': dt.month,
        'day': dt.day,
        'minute': dt.minute,
        'hour': dt.hour,
        'day_of_week': dt.strftime('%A')
    }

# Load the dataset
@st.cache_data
def load_data():
    """Load the traffic dataset"""
    try:
        df = pd.read_csv('traffic_dataset1.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
        
        # Encode day_of_week to numeric for consistency
        df['day_of_week_encoded'] = df['day_of_week'].map({
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_model_input(vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute):
    """Prepare input data in the format expected by the model"""
    # Create base dataframe
    input_data = pd.DataFrame({
        'vehicle_count': [vehicle_count],
        'lag_1': [lag_1],
        'lag_2': [lag_2], 
        'lag_3': [lag_3],
        'hour': [hour],
        'month': [month],
        'day': [day],
        'minute': [minute],
    })
    
    # One-hot encode day_of_week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_name in days:
        input_data[f'day_of_week_{day_name}'] = 1 if day_of_week == day_name else 0
    
    # Reorder columns to match model expectations
    expected_columns = [
        'vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'month', 'day', 'minute',
        'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday', 
        'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday'
    ]
    
    # Ensure we have all expected columns
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Select and reorder columns
    input_data = input_data[expected_columns]
    
    return input_data

def create_sample_data_from_timestamp(timestamp_obj):
    """Create sample input based on a timestamp from the dataset"""
    # Handle both string and Timestamp objects
    if isinstance(timestamp_obj, str):
        from datetime import datetime
        dt = datetime.strptime(timestamp_obj, '%d/%m/%Y %H:%M')
    else:
        # timestamp_obj is already a pandas Timestamp
        dt = timestamp_obj
    
    return {
        'month': dt.month,
        'day': dt.day,
        'minute': dt.minute,
        'hour': dt.hour,
        'day_of_week': dt.strftime('%A')
    }

def encode_day_of_week(day_name):
    """Convert day name to numeric value"""
    day_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    return day_mapping.get(day_name, 1)  # Default to Tuesday if not found

def main():
    st.title("🚗 Traffic Prediction Demo")
    st.markdown("---")
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    if model is None or data is None:
        st.error("Failed to load model or data. Please check the files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "Model Info"])
    
    if page == "Prediction":
        prediction_page(model, data)
    elif page == "Data Analysis":
        analysis_page(data)
    else:
        model_info_page(model, data)

def prediction_page(model, data):
    """Page for making predictions"""
    st.header("🔮 Traffic Prediction")
    
    # Sample data selector
    st.subheader("📋 Quick Start with Sample Data")
    col_sample1, col_sample2 = st.columns([2, 1])
    
    with col_sample1:
        # Get sample records from dataset
        sample_records = data.sample(5).reset_index(drop=True)
        selected_idx = st.selectbox(
            "Choose a sample record from the dataset:",
            range(len(sample_records)),
            format_func=lambda x: f"Sample {x+1}: {sample_records.iloc[x]['timestamp']} - {sample_records.iloc[x]['vehicle_count']} vehicles"
        )
    
    with col_sample2:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎯 Use Sample Data"):
                st.session_state.use_sample = True
                st.session_state.selected_sample = sample_records.iloc[selected_idx]
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset"):
                st.session_state.use_sample = False
                if 'selected_sample' in st.session_state:
                    del st.session_state.selected_sample
                st.rerun()
    
    # Show selected sample info
    if hasattr(st.session_state, 'use_sample') and st.session_state.use_sample:
        st.info(f"📍 **Using Sample Data:** {st.session_state.selected_sample['timestamp']} | Target: {st.session_state.selected_sample['target_next_1h']} vehicles")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Features")
        
        # Get default values
        if hasattr(st.session_state, 'use_sample') and st.session_state.use_sample:
            sample = st.session_state.selected_sample
            default_vehicle_count = int(sample['vehicle_count'])
            default_lag_1 = int(sample['lag_1'])
            default_lag_2 = int(sample['lag_2'])
            default_lag_3 = int(sample['lag_3'])
            default_day_of_week = sample['day_of_week']
            default_hour = int(sample['hour'])
            
            # Extract additional info from timestamp
            timestamp_info = create_sample_data_from_timestamp(sample['timestamp'])
            default_month = timestamp_info['month']
            default_day = timestamp_info['day']
            default_minute = timestamp_info['minute']
        else:
            default_vehicle_count = 5
            default_lag_1 = 4
            default_lag_2 = 3
            default_lag_3 = 2
            default_day_of_week = "Tuesday"
            default_hour = 12
            default_month = 9
            default_day = 13
            default_minute = 5
        
        # Input form
        with st.form("prediction_form"):
            vehicle_count = st.number_input(
                "Current Vehicle Count",
                min_value=0,
                max_value=100,
                value=default_vehicle_count,
                help="Number of vehicles currently observed"
            )
            
            lag_1 = st.number_input(
                "Lag 1 (Previous 5 min)",
                min_value=0,
                max_value=100,
                value=default_lag_1,
                help="Vehicle count from 5 minutes ago"
            )
            
            lag_2 = st.number_input(
                "Lag 2 (Previous 10 min)",
                min_value=0,
                max_value=100,
                value=default_lag_2,
                help="Vehicle count from 10 minutes ago"
            )
            
            lag_3 = st.number_input(
                "Lag 3 (Previous 15 min)",
                min_value=0,
                max_value=100,
                value=default_lag_3,
                help="Vehicle count from 15 minutes ago"
            )
            
            day_of_week = st.selectbox(
                "Day of Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(default_day_of_week)
            )
            
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=default_hour,
                help="Hour in 24-hour format"
            )
            
            # Additional fields for model compatibility
            col_a, col_b = st.columns(2)
            with col_a:
                month = st.number_input(
                    "Month",
                    min_value=1,
                    max_value=12,
                    value=default_month,
                    help="Month (1-12)"
                )
                
                day = st.number_input(
                    "Day of Month",
                    min_value=1,
                    max_value=31,
                    value=default_day,
                    help="Day of the month"
                )
            
            with col_b:
                minute = st.selectbox(
                    "Minute",
                    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                    index=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].index(default_minute) if default_minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] else 1,
                    help="Minute in 5-minute intervals"
                )
            
            submit_button = st.form_submit_button("🚀 Predict Traffic")
    
    with col2:
        st.subheader("Prediction Result")
        
        if submit_button:
            try:
                # Prepare input data with proper encoding and features
                input_data = prepare_model_input(
                    vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute
                )
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.success("Prediction Complete!")
                
                # Show actual vs predicted if using sample data
                if hasattr(st.session_state, 'use_sample') and st.session_state.use_sample:
                    actual_value = st.session_state.selected_sample['target_next_1h']
                    st.info(f"**Actual Value from Dataset:** {actual_value} vehicles")
                    
                    # Calculate accuracy metrics
                    error = abs(prediction - actual_value)
                    accuracy_percentage = max(0, (1 - error / max(actual_value, 1)) * 100)
                    
                    col_metrics = st.columns(3)
                    with col_metrics[0]:
                        st.metric("Predicted", f"{prediction:.1f}", f"{prediction - vehicle_count:.1f} from current")
                    with col_metrics[1]:
                        st.metric("Actual", f"{actual_value}", f"{actual_value - vehicle_count:.1f} from current")
                    with col_metrics[2]:
                        st.metric("Error", f"{error:.1f}", f"Accuracy: {accuracy_percentage:.1f}%")
                else:
                    # Create a nice display for the result
                    col2_1, col2_2 = st.columns([1, 1])
                    
                    with col2_1:
                        st.metric(
                            label="Predicted Traffic (Next Hour)",
                            value=f"{prediction:.1f} vehicles",
                            delta=f"{prediction - vehicle_count:.1f} from current"
                        )
                
                # Create gauge chart
                if not (hasattr(st.session_state, 'use_sample') and st.session_state.use_sample):
                    with col2_2:
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prediction,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Traffic Level"},
                            delta = {'reference': vehicle_count},
                            gauge = {
                                'axis': {'range': [None, 50]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 10], 'color': "lightgray"},
                                    {'range': [10, 25], 'color': "gray"},
                                    {'range': [25, 50], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 30
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Show gauge chart for sample data comparison
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Predicted vs Actual"},
                        delta = {'reference': st.session_state.selected_sample['target_next_1h']},
                        gauge = {
                            'axis': {'range': [None, 50]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 10], 'color': "lightgray"},
                                {'range': [10, 25], 'color': "gray"},
                                {'range': [25, 50], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': st.session_state.selected_sample['target_next_1h']
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show input summary
                st.subheader("Input Summary")
                display_data = pd.DataFrame({
                    'Feature': ['Vehicle Count', 'Lag 1', 'Lag 2', 'Lag 3', 'Day of Week', 'Hour', 'Month', 'Day', 'Minute'],
                    'Value': [str(vehicle_count), str(lag_1), str(lag_2), str(lag_3), str(day_of_week), str(hour), str(month), str(day), str(minute)]
                })
                st.dataframe(display_data, hide_index=True)
                
                # Show model input format
                st.subheader("Model Input Features")
                st.dataframe(input_data, hide_index=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

def analysis_page(data):
    """Page for data analysis and visualization"""
    st.header("📊 Data Analysis")
    
    # Basic statistics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        date_range = f"{data['timestamp'].dt.date.min()} to {data['timestamp'].dt.date.max()}"
        st.metric("Date Range", date_range)
    with col3:
        st.metric("Avg Vehicle Count", f"{data['vehicle_count'].mean():.1f}")
    with col4:
        st.metric("Max Vehicle Count", data['vehicle_count'].max())
    
    # Additional statistics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Min Vehicle Count", data['vehicle_count'].min())
    with col6:
        st.metric("Std Deviation", f"{data['vehicle_count'].std():.1f}")
    with col7:
        unique_days = data['day_of_week'].nunique()
        st.metric("Days Covered", unique_days)
    with col8:
        unique_hours = data['hour'].nunique()
        st.metric("Hours Covered", unique_hours)
    
    # Time series plot
    st.subheader("Traffic Patterns Over Time")
    
    # Allow user to select sample size
    sample_size = st.slider("Sample size for time series plot", 100, 5000, 1000, 100)
    sample_data = data.head(sample_size).copy()
    
    fig = px.line(
        sample_data, 
        x='timestamp', 
        y='vehicle_count',
        title=f"Vehicle Count Over Time (First {sample_size} records)",
        labels={'vehicle_count': 'Vehicle Count', 'timestamp': 'Time'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction accuracy visualization
    if 'target_next_1h' in data.columns:
        st.subheader("Historical Prediction vs Actual Analysis")
        
        # Create lag-based simple prediction for comparison
        sample_for_analysis = data.head(1000).copy()
        sample_for_analysis['simple_prediction'] = sample_for_analysis['lag_1']  # Simple: assume next hour = current
        sample_for_analysis['error'] = abs(sample_for_analysis['target_next_1h'] - sample_for_analysis['simple_prediction'])
        
        col_acc1, col_acc2 = st.columns(2)
        
        with col_acc1:
            # Error distribution
            fig_error = px.histogram(
                sample_for_analysis,
                x='error',
                title="Distribution of Prediction Errors (Simple Model)",
                labels={'error': 'Absolute Error', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col_acc2:
            # Actual vs Predicted scatter
            fig_scatter = px.scatter(
                sample_for_analysis.head(200),
                x='target_next_1h',
                y='simple_prediction',
                title="Actual vs Simple Prediction (First 200 records)",
                labels={'target_next_1h': 'Actual Next Hour', 'simple_prediction': 'Predicted Next Hour'}
            )
            # Add perfect prediction line
            max_val = max(sample_for_analysis['target_next_1h'].max(), sample_for_analysis['simple_prediction'].max())
            fig_scatter.add_shape(
                type="line", line=dict(dash="dash"),
                x0=0, x1=max_val, y0=0, y1=max_val
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Hourly patterns
    col1, col2 = st.columns(2)
    
    with col1:
        hourly_avg = data.groupby('hour')['vehicle_count'].mean().reset_index()
        fig = px.bar(
            hourly_avg,
            x='hour',
            y='vehicle_count',
            title="Average Traffic by Hour of Day",
            labels={'vehicle_count': 'Avg Vehicle Count', 'hour': 'Hour'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        daily_avg = data.groupby('day_of_week')['vehicle_count'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg['day_of_week'] = pd.Categorical(daily_avg['day_of_week'], categories=day_order, ordered=True)
        daily_avg = daily_avg.sort_values('day_of_week')
        
        fig = px.bar(
            daily_avg,
            x='day_of_week',
            y='vehicle_count',
            title="Average Traffic by Day of Week",
            labels={'vehicle_count': 'Avg Vehicle Count', 'day_of_week': 'Day'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = ['vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week_encoded', 'target_next_1h']
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Numeric Features"
    )
    st.plotly_chart(fig, use_container_width=True)

def model_info_page(model, data):
    """Page showing model information"""
    st.header("🤖 Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Details")
        st.write("**Model Type:** XGBoost Regressor")
        st.write("**Objective:** Predict traffic count for next hour")
        st.write("**Input Features:**")
        st.write("- vehicle_count: Current vehicle count")
        st.write("- lag_1, lag_2, lag_3: Historical traffic counts")
        st.write("- day_of_week_*: One-hot encoded day of week")
        st.write("- hour: Hour of the day (0-23)")
        st.write("- month: Month of the year (1-12)")
        st.write("- day: Day of the month (1-31)")
        st.write("- minute: Minute in 5-minute intervals")
        
        st.subheader("Dataset Information")
        st.write(f"**Total Records:** {len(data):,}")
        st.write(f"**Features:** {len(data.columns)}")
        st.write(f"**Date Range:** {data['timestamp'].dt.date.min()} to {data['timestamp'].dt.date.max()}")
    
    with col2:
        st.subheader("Feature Statistics")
        feature_stats = data[['vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week_encoded', 'target_next_1h']].describe()
        st.dataframe(feature_stats)
        
        st.subheader("Sample Data")
        st.dataframe(data.head(10))

if __name__ == "__main__":
    main()
