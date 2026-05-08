import pandas as pd
import numpy as np

def load_and_merge_data(data_dir='Milestone 1 Data/'):
    
    # Load datasets 
    assessments = pd.read_csv(f'{data_dir}assessments.csv')
    courses = pd.read_csv(f'{data_dir}courses.csv')
    student_assessments = pd.read_csv(f'{data_dir}StudentAssesments.csv')
    student_info = pd.read_csv(f'{data_dir}studentInfo.csv')
    student_reg = pd.read_csv(f'{data_dir}studentRegistration.csv')
    student_vle = pd.read_csv(f'{data_dir}studentVle.csv')
    vle = pd.read_csv(f'{data_dir}vle.csv') 

    datasets = [assessments, courses, student_assessments, student_info, student_reg, student_vle, vle]

    # 2. Standardize Column Names
    column_mapping = {
        'id_assessment': 'id_assess',
        'code_module': 'code_mod',
        'code_presentation': 'code_pres',
        'code_presentat': 'code_pres'
    }
    
    for df in datasets:
        df.rename(columns=column_mapping, inplace=True)
        # Replace the Open University '?' with standard NaN
        df.replace('?', np.nan, inplace=True)

    # Merge Strategy
    df = student_assessments.copy()

    # Merge assessments 
    df = pd.merge(df, assessments, on='id_assess', how='left')

    # Merge student_info 
    df = pd.merge(df, student_info, on=['id_student', 'code_mod', 'code_pres'], how='left')

    # Aggregate VLE clicks to see total engagement per student per module/presentation
    vle_agg = student_vle.groupby(['id_student', 'code_mod', 'code_pres'])['sum_click'].sum().reset_index()
    df = pd.merge(df, vle_agg, on=['id_student', 'code_mod', 'code_pres'], how='left')

    # Clean Target Variable
    df.dropna(subset=['score'], inplace=True)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    return df