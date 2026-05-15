import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "Milestone 2 Data", "extracted")

# Load raw CSV tables from extracted dataset
def load_raw_data():
    
    courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    assessments = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"))
    student_info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"))
    student_reg = pd.read_csv(os.path.join(DATA_DIR, "studentRegistration.csv"))
    student_assess = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"))
    print("Loading studentVle (large file)...")
    student_vle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"))
    vle = pd.read_csv(os.path.join(DATA_DIR, "vle.csv"))
    return courses, assessments, student_info, student_reg, student_assess, student_vle, vle


def create_assessment_class(score):
    """Map numeric score to assessmentClass categories."""
    if pd.isna(score):
        return np.nan
    if score < 40:
        return "Fail"
    elif score < 60:
        return "Good"
    elif score < 80:
        return "Very Good"
    else:
        return "Excellent"


def build_dataset():
    """
    Build final merged dataset with engineered features & assessmentClass target.
    """
    courses, assessments, student_info, student_reg, student_assess, student_vle, vle = load_raw_data()

    # --- Merge studentAssessment with assessments ---
    sa = student_assess.merge(assessments, on="id_assessment", how="left")

    # --- Create assessmentClass from score ---
    sa["assessmentClass"] = sa["score"].apply(create_assessment_class)
    sa = sa.dropna(subset=["assessmentClass"])

    # --- Aggregate VLE interactions per student per module-presentation ---
    print("Aggregating VLE interactions...")
    vle_agg = student_vle.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_clicks=("sum_click", "sum"),
        num_vle_interactions=("sum_click", "count"),
        avg_clicks_per_day=("sum_click", "mean"),
        unique_days_active=("date", "nunique"),
    ).reset_index()

    # --- Aggregate VLE activity types ---
    vle_with_type = student_vle.merge(
        vle[["id_site", "activity_type"]], on="id_site", how="left"
    )
    activity_counts = vle_with_type.groupby(
        ["id_student", "code_module", "code_presentation", "activity_type"]
    )["sum_click"].sum().unstack(fill_value=0).reset_index()
    activity_counts.columns.name = None
    act_cols = [c for c in activity_counts.columns if c not in ["id_student", "code_module", "code_presentation"]]
    activity_counts = activity_counts.rename(columns={c: f"act_{c}" for c in act_cols})

    # --- Merge everything together ---
    # Merge sa with student_info
    df = sa.merge(student_info, on=["id_student", "code_module", "code_presentation"], how="left")

    df = df.merge(courses, on=["code_module", "code_presentation"], how="left")

    df = df.merge(student_reg, on=["id_student", "code_module", "code_presentation"], how="left")

    df = df.merge(vle_agg, on=["id_student", "code_module", "code_presentation"], how="left")

    df = df.merge(activity_counts, on=["id_student", "code_module", "code_presentation"], how="left")

    # --- Feature Engineering ---
    # Submission delay
    df["submission_delay"] = df["date_submitted"] - df["date"]
    
    # Registration earliness
    df["registration_earliness"] = -df["date_registration"]

    # Is unregistered flag
    df["is_unregistered"] = df["date_unregistration"].notna().astype(int)

    # --- Select final columns ---
    # Drop score & identifiers that shouldn't be features
    drop_cols = ["score", "id_assessment", "id_student", "id_site",
                 "date_unregistration", "date_registration", "final_result"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # --- Save ---
    output_path = os.path.join(os.path.dirname(__file__), "processed_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['assessmentClass'].value_counts()}")
    print(f"\nColumns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    build_dataset()
