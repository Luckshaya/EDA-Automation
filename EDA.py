import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis 📋📊")

uploaded_file = st.file_uploader("Upload your CSV file here...", type=["csv"])

def find_cat_cont_columns(df):
    cont_columns, cat_columns = [],[]
    for col in df.columns:        
        if len(df[col].unique()) <= 25 or df[col].dtype == np.object_: 
            cat_columns.append(col.strip())
        else:
            cont_columns.append(col.strip())
    return cont_columns, cat_columns

def create_correlation_chart(correlation):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    return fig


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
    
    with tab1:
        cont_columns, cat_columns = find_cat_cont_columns(df)

        st.subheader("1. Dataset")
        st.write("Structure of Data", df.head())
        st.write("Last some entries", df.tail())
        st.write("Shape of the Dataset",df.shape)
        st.write("Datatypes of Features", df.dtypes)
        
        Duplicates = df.nunique()
        missing_values_count = df.isna().sum()
        missing_values_percentage = (df.isnull().sum() / len(df)) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Count of Missing Values",missing_values_count)
            
        with col2:
            st.write("Missing Values Percentage",missing_values_percentage)
        
        with col3:
            st.write("Duplicates", Duplicates)

        st.write("Detailed Description",df.describe())
        st.write("Features", df.columns.tolist())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<span style='color: black; font-weight: bold;'>Categorical Columns</span>", unsafe_allow_html=True)
            st.write(cat_columns)
        
        with col2:
            st.markdown("<span style='color: black; font-weight: bold;'>Categorical Columns</span>", unsafe_allow_html=True)
            st.write(cont_columns)


    with tab2:
        df_descr = df.describe()
        cat_cols=df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        print("Categorical Variables:")
        print(cat_cols)
        print("Numerical Variables:")
        print(num_cols)

        for col in num_cols:
            st.subheader(f"Column: {col}")
            st.write(f"Skew: {round(df[col].skew(), 2)}")
            
            plt.figure(figsize=(30, 8))
            plt.subplot(1, 2, 1)
            plt.hist(df[col], bins=20,color = 'red', edgecolor='black')
            plt.xlabel(col)
            plt.ylabel('Count')
            st.pyplot(plt)

            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.xlabel(col)
            st.pyplot(plt)
        

    with tab3:
            df1 = df.copy()
            st.title("Scatter Plot Visualization")
            st.write("Explore relationships between variables using a scatter plot.")
            st.write("Select any two columns")
            sns.set(style="ticks")
            selected_columns = st.multiselect("Select columns", df1.columns)
            for i, var1 in enumerate(selected_columns):
                for j, var2 in enumerate(selected_columns):
                    if i < j:
                        plt.figure(figsize=(8, 6))
                        sns.scatterplot(data=df1, x=var1, y=var2)
                        plt.title(f"{var1} vs {var2}")
                        st.pyplot(plt)  # Display the plot in Streamlit
    
    with tab4:

        correlation = df[cont_columns].corr()
        corr_fig = create_correlation_chart(correlation)
        
        st.subheader("3. Correlation Chart")
        st.pyplot(corr_fig, use_container_width=True)