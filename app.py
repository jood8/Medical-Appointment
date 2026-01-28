
import streamlit as st 
import pandas as pd 
import numpy as np
import random    
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2
if "data_cleaned" not in st.session_state:
    st.session_state.data_cleaned = False

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

@st.cache_data
def load_data():
    return pd.read_csv("Medical Appointment.csv")
df = load_data()
if st.session_state.cleaned_df is None:
    clean = df.copy()
    clean['ScheduledDay'] = pd.to_datetime(clean['ScheduledDay'], errors='coerce')
    clean['AppointmentDay'] = pd.to_datetime(clean['AppointmentDay'], errors='coerce')
    clean['ScheduledDay'] = clean['ScheduledDay'].dt.tz_localize(None)
    clean['AppointmentDay'] = clean['AppointmentDay'].dt.tz_localize(None)
    st.session_state.cleaned_df = clean

st.set_page_config(page_title="Medical Visit Flow",layout="wide")
st.title("Medical Appointment No Shows ")
st.subheader("Why do 30% of patients miss their scheduled appointments?")
st.markdown(
            "🔗 Data Source:" "[Medical Appointment No Shows on kaggle]" 
            "(https://www.kaggle.com/datasets/joniarroba/noshowappointments)")
img=Image.open("No Shows.png")
st.image(img,use_container_width=True)
st.sidebar.title("Sections")
sections = [
    "Overview",
    "Data Quality",
    "Cleaning",
    "Exploratory Analysis",
    "Feature Engineering",
    "Final Visualization"
]

if "current_section" not in st.session_state:
    st.session_state.current_section = sections[0]
if "data_cleaned" not in st.session_state:
    st.session_state.data_cleaned = False

section = st.sidebar.radio(
    "Go to:",
    sections,
    index=sections.index(st.session_state.current_section))

show_raw=st.sidebar.checkbox("Show raw data",value=True)
rows_to_show=st.sidebar.slider(
  "Number of rows to display",min_value=10,max_value=110527 ,value=20)

  #-----Overview section-----
if section == "Overview":
    st.subheader("Dataset Overview")
    st.write(f"Number of rows: {df.shape[0]} , Number of columns: {df.shape[1]}")
    col_types=pd.DataFrame({
        'Column Name':df.columns,"Data Type":[str(df[col].dtype)for col in df.columns]})
    st.dataframe(col_types)
    st.subheader("Data Preview")
    if show_raw:
        st.dataframe(df.head(rows_to_show))
    else:
        st.info("Raw data preview hidden, use the checkbox to show it ")
        
  #-----Data Quality-----
if section == "Data Quality":
    st.subheader("Data Quality checks")
    clean = st.session_state.cleaned_df
    num,cat,mis,date=st.tabs(["Numeric Issues","Categorical Issues", "Missing Values","Dates"])
    with num:
        st.write("Check numeric columns for logical inconsistencies:")
        num_cols = clean.select_dtypes(include=np.number).columns.tolist()
        issues_summary = {}
        for col in num_cols:
            if col in clean.columns:
                count_invalid = 0
                if col == "Age":
                    count_invalid = (clean[col] <= 0).sum()
                elif col == "Handcap":
                    count_invalid = (clean[col] > 1).sum()
                elif col == "PatientId":
                    count_invalid = (clean[col] > 1e12).sum()
                else:
                    count_invalid =0
            issues_summary[col] = count_invalid
        issues=pd.DataFrame(list(issues_summary.items()),columns=["Column","Number of Issuse"])
        st.dataframe(issues.head(rows_to_show))
    with cat:
        st.write("Check categorical columns for unexpected values.")
        categorical_cols = ["Gender","No-show","Neighbourhood" ]
        status=["✅ All values are logical"] *len (categorical_cols)
        status_df=pd.DataFrame({
            "Column":categorical_cols,"Status":status})
        st.table(status_df)
    with mis:
        st.write("Check for missing values in all columns")        
        missing=clean.isnull().sum()
        missing =missing[missing>0]
        if not missing.empty:
            st.dataframe(missing)
        else:
            st.success("No missing values found in the dataset")
    with date:
        st.write("Check date columns for logical inconsistencies and extract components:")
        invalid_dates = clean[clean['ScheduledDay'].dt.date > clean['AppointmentDay']]
        if not invalid_dates.empty:
            st.write(f"Found {invalid_dates.shape[0]} rows where ScheduledDay > AppointmentDay (logical inconsistency):")
            st.dataframe(invalid_dates[['ScheduledDay','AppointmentDay']].head())
        else:
            st.write("✅ All dates are logically consistent")
#-----Cleaning----

if section == "Cleaning" :
    st.subheader("Apply Data Cleaning")
    st.write("Click the button  below to apply automated data clean")
    if st.button("🧹Clean Dataset"):
        progress = st.progress(0)
        status = st.empty()
        clean = st.session_state.cleaned_df
        st.session_state.data_cleaned = True
        #rename
        clean.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived', 'No-show': 'NoShow'},inplace=True)
        progress.progress(15)
        #age
        median_age = clean.loc[clean["Age"]>0 ,"Age"].median()
        clean.loc[clean["Age"] <=0 ,"Age"]= median_age    
        progress.progress(33)
        #Handicap
        clean.loc[clean["Handicap"]>1,"Handicap"]=1
        progress.progress(50)
        #drop
        clean.drop(columns=["PatientId", "AppointmentID"], inplace=True)
        progress.progress(70)
        #
        mask = clean["ScheduledDay"] > clean["AppointmentDay"]
        clean.loc[mask,"ScheduledDay"] = (
            clean.loc[mask,"AppointmentDay"] - pd.Timedelta(days=1))
        progress.progress(100)        
        st.session_state.cleaned_df = clean
        st.success("✅ Data Cleaning completed successfully")
        st.session_state.data_cleaned = True
        st.dataframe(clean.head(rows_to_show))
#----Exploratory Analysis----

if section == "Exploratory Analysis":
    st.subheader("Appointment Attendance Distribution")
    clean = st.session_state.cleaned_df
    clean["WaitingDays"] = (
    clean["AppointmentDay"] - clean["ScheduledDay"]).dt.days
    st.subheader("Exploratory Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Appointment Attendance Distribution")
        no_show = clean["NoShow"].value_counts()
        fig1 = px.pie(names=no_show.index,values=no_show.values,title="Show vs No-show",color=no_show.index,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("---")
    with col2:
        st.write("Age Distribution by Appointment Status")
        fig2 = px.box(clean,x="NoShow",y="Age",color="NoShow",
            title="Age vs No-show", color_discrete_sequence=["#636EFA", "#EF553B"])
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Effect of SMS Reminder on Attendance")
        sms_counts = clean.groupby(["SMSReceived", 'NoShow']).size().unstack()
        fig3 = px.bar(sms_counts,barmode="group",title="SMS Reminder vs Appointment Attendance",color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("---")
    with col4:
        st.write("Effect of Alcoholism on Appointment Attendance")
        alcohol_counts = clean.groupby(["Alcoholism", "NoShow"]).size().unstack()
        alcohol_counts.index = alcohol_counts.index.map({0: "Non-Alcoholic",1: "Alcoholic"})
        fig4 = px.bar(
            alcohol_counts,
            barmode="group",
            title="Alcoholism vs Appointment Attendance",
            labels={"value": "Number of Patients", "Alcoholism": "Alcoholism Status"})
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        st.write("Heatmap: Correlation Between Factors")
        heatmap_data = clean[["Age", "WaitingDays", "SMSReceived", "Hypertension", "Diabetes"]].corr()
        fig5 = px.imshow(heatmap_data, text_auto=True, title="Correlation Heatmap",color_continuous_scale="Viridis")
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        st.write("No-show Behavior vs Waiting Time")
        fig6 = px.box(
            clean,
            x="NoShow",
            y="WaitingDays",
            color="NoShow",
            title="Waiting Time Distribution by Appointment Outcome",
            labels={
                "NoShow": "Appointment Outcome",
                "WaitingDays": "Waiting Days"},
            hover_data=["Age", "Gender", "SMSReceived"],
            color_discrete_sequence=px.colors.qualitative.Set2)
        fig6.update_layout(height=450)
        st.plotly_chart(fig6, use_container_width=True)
#------Feature------

if section =="Feature Engineering":
    st.subheader("Feature Engineering")
    clean = st.session_state.cleaned_df

    #استخراج الوقت 
    clean["ScheduledDay"] = pd.to_datetime(clean["ScheduledDay"])
    clean["AppointmentDay"] = pd.to_datetime(clean["AppointmentDay"])
    for col in ["ScheduledDay", "AppointmentDay"]:
        clean[f"{col}_Year"] = clean[col].dt.year
        clean[f"{col}_Month"] = clean[col].dt.month
        clean[f"{col}_Day"] = clean[col].dt.day
        clean[f"{col}_Hour"] = clean[col].dt.hour
        clean[f"{col}_Minute"] = clean[col].dt.minute
   # encoding
    clean['Gender'] = clean['Gender'].replace({'M': 1, 'F': 0})
    clean["NoShow"]=clean["NoShow"].apply(lambda x:0 if str (x).lower()=="no"else 1 )
    le = LabelEncoder()
    clean['Neighbourhood_encode'] = le.fit_transform(clean['Neighbourhood'])
    display_cols = [c for c in clean.columns if 'ScheduledDay' in c or 'AppointmentDay' in c] + ['Gender', 'NoShow' ,"Neighbourhood_encode"]
    st.dataframe(clean[display_cols].head(rows_to_show))
   #scaling
    st.subheader("Scaling")     
    ft=st.session_state.cleaned_df.copy()
    ft['WaitingDays'] = (ft['AppointmentDay'] - ft['ScheduledDay']).dt.days
    scaler=StandardScaler()
    ft_scaled = ft[['Age','WaitingDays']].copy()
    ft_scaled[['Age','WaitingDays']] = scaler.fit_transform(ft_scaled[['Age','WaitingDays']])
    st.dataframe(ft_scaled.head(rows_to_show))

    if "pca_df" not in st.session_state:
        pca = PCA(n_components=2)
        principle = pca.fit_transform(ft_scaled)
        pca_df = pd.DataFrame(principle, columns=["PC1", "PC2"])
        pca_df["NoShow"] = clean["NoShow"].values
        st.session_state.pca_df = pca_df
        st.session_state.explained_variance = pca.explained_variance_ratio_
        #pca
        st.subheader("PCA")
        explained = st.session_state.explained_variance
        st.write("Explained Variance Ratio:")
        st.write(f"PC1: {explained[0]:.2f} | PC2: {explained[1]:.2f}")
        fig7 = px.scatter(
            st.session_state.pca_df,
            x="PC1",
            y="PC2",
            title="PCA Visualization"
        )
        st.plotly_chart(fig7, use_container_width=True)
    with st.expander("🖼 Image Preprocessing"):
        img = Image.open("No Shows.png")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224,224))
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Original Image")
            st.image(img, use_container_width=True)
        with col2:
            st.caption("Processed Image")
            st.image(resized, use_container_width=True, clamp=True)
            
  #-----Final----
            
if section == "Final Visualization":
        st.subheader("Final Dataset Visualization using PCA")
        fig8 = px.scatter(
        st.session_state.pca_df,
        x="PC1",y="PC2",color="NoShow",
        title="PCA Projection of Medical Appointment Data",
        labels={
                "PC1": "Principal Component 1",
                "PC2": "Principal Component 2",
                "NoShow": "Appointment Status"})
        fig8.update_traces(
        marker=dict(size=6, opacity=0.6, line=dict(width=0)))
        fig8.update_layout(
            height=500,template="simple_white",showlegend=True)
        st.plotly_chart(fig8, use_container_width=True)
        
# ------ Navigation  ------

current_idx = sections.index(section)
col1, col2 = st.columns(2)

with col1:
    if current_idx > 0:
        if st.button("⬅ Previous"):
            st.session_state.current_section = sections[current_idx - 1]
            st.rerun()

with col2:
    if current_idx < len(sections) - 1:
        if st.button("Next ➡"):
            if section == "Cleaning" and not st.session_state.data_cleaned:
                st.warning("⚠ Please clear the data first before proceeding to the next step")
                st.stop()

            st.session_state.current_section = sections[current_idx + 1]
            st.rerun()

