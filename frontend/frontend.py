import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime


from app import SurveillanceSystem

def initialize_surveillance_system():
    if 'surveillance_system' not in st.session_state:
        st.session_state.surveillance_system = SurveillanceSystem()
    return st.session_state.surveillance_system

def create_metrics_chart(risk_scores, timestamps):
    df = pd.DataFrame({
        'timestamp': timestamps,
        'risk_score': risk_scores
    })
    
    fig = px.line(df, x='timestamp', y='risk_score',
                  title='Risk Score Over Time',
                  labels={'risk_score': 'Risk Score', 'timestamp': 'Time'})
    return fig

def create_heatmap(environment_metrics):
    metrics_df = pd.DataFrame([environment_metrics])
    
    fig = go.Figure(data=go.Heatmap(
        z=[list(metrics_df.values[0])],
        x=list(metrics_df.columns),
        y=['Metrics'],
        colorscale='RdYlBu_r'
    ))
    
    fig.update_layout(
        title='Environment Metrics Heatmap',
        xaxis_title='Metric Type',
        yaxis_title='Current State'
    )
    return fig

def main():
    st.set_page_config(page_title="Video Surveillance Analysis", layout="wide")
    
    st.title("Video Surveillance Analysis System")
    
    
    surveillance_system = initialize_surveillance_system()
    
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        
        cap = cv2.VideoCapture(tfile.name)
        
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        
        col1, col2 = st.columns([2, 1])
        
        
        progress_bar = st.progress(0)
        
        
        risk_scores = []
        timestamps = []
        
        
        with col1:
            st.subheader("Video Analysis")
            video_placeholder = st.empty()
            
        with col2:
            st.subheader("Real-time Metrics")
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            heatmap_placeholder = st.empty()
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            
            results, vit_outputs, alerts = surveillance_system.process_frame(frame)
            
            
            current_state = surveillance_system.digital_twin.current_state
            
            
            risk_scores.append(current_state['risk_assessment']['overall_risk_score'])
            timestamps.append(datetime.now())
            
            
            annotated_frame = results[0].plot()
            
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            
            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", current_state['risk_assessment']['risk_level'])
                m2.metric("Risk Score", f"{current_state['risk_assessment']['overall_risk_score']:.2f}")
                m3.metric("Processing FPS", f"{surveillance_system.get_processing_fps():.1f}")
                
                
                if alerts:
                    st.warning(f"Alert: {alerts[0]['description']}")
            
            
            chart_placeholder.plotly_chart(create_metrics_chart(risk_scores, timestamps),
                                             key=f"chart_{frame_count}")
            heatmap_placeholder.plotly_chart(create_heatmap(current_state['environment_metrics']),
                                             key=f"heatmap_{frame_count}")
            
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            
            
            time.sleep(1/fps)
        
        cap.release()
        
        
        st.subheader("Analysis Summary")
        st.write({
            "Total Frames Processed": frame_count,
            "Average Risk Score": np.mean(risk_scores),
            "Max Risk Score": np.max(risk_scores),
            "Average Processing FPS": surveillance_system.get_processing_fps()
        })
        
        
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'risk_score': risk_scores
        })
        
        st.download_button(
            label="Download Analysis Results",
            data=results_df.to_csv().encode('utf-8'),
            file_name='surveillance_analysis.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
