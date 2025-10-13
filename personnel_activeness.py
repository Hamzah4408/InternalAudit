#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import gradio as gr
import os
import tempfile
import re
from fastapi import FastAPI
from starlette.responses import PlainTextResponse

# =====================================================
# Utility: Find Email Column
# =====================================================
def find_email_column(df):
    for col in df.columns:
        if 'email' in col.lower():
            return col
    raise ValueError("No email column found in the uploaded file.")

# =====================================================
# Staff Verification Logic
# =====================================================
def verify_access_by_sheet(access_file, hr_file):
    ext = os.path.splitext(access_file.name)[-1].lower()

    if ext in ['.xls', '.xlsx']:
        xls = pd.ExcelFile(access_file)
        sheet_dfs = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    elif ext == '.csv':
        df = pd.read_csv(access_file)
        sheet_dfs = {"CSV": df}
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel files.")

    hr_df = pd.read_excel(hr_file) if hr_file.name.endswith(('.xls', '.xlsx')) else pd.read_csv(hr_file)
    hr_email_col = find_email_column(hr_df)
    hr_df['Email_Username'] = hr_df[hr_email_col].str.split('@').str[0]

    results = {}
    combined_df = []

    for sheet_name, access_df in sheet_dfs.items():
        try:
            access_email_col = find_email_column(access_df)
            access_df['Email_Username'] = access_df[access_email_col].str.split('@').str[0]

            merged_df = access_df.merge(hr_df[['Email_Username', 'Employment Status']], on='Email_Username', how='left')
            merged_df['AccessStatus'] = merged_df['Employment Status'].apply(
                lambda x: 'Valid Staff' if pd.notnull(x) and x.lower() == 'active'
                else 'Former Staff' if pd.notnull(x) and x.lower() == 'inactive'
                else 'Unknown Staff'
            )
            merged_df['Platform'] = sheet_name

            results[sheet_name] = merged_df
            combined_df.append(merged_df)
        except Exception:
            results[sheet_name] = pd.DataFrame()

    return results, pd.concat(combined_df, ignore_index=True) if combined_df else pd.DataFrame()

# =====================================================
# Access Level Analysis Logic
# =====================================================
import re

# --- Your existing definitions ---
ACCESS_COLUMN_MAP = {
    "FBBM": "Role",
    "Google": "Access level",
    "Meta": "Role",
    "TikTok": "Member Role"
}

ACCESS_RULES = {
    "admin": {"permissions": ["all"], "level": 3},
    "ADMIN": {"permissions": ["all"], "level": 3},
    "DEVELOPER": {"permissions": ["deploy", "code"], "level": 2},
    "FINANCE_EDITOR": {"permissions": ["budget", "reports"], "level": 2},
    "FINANCE_ANALYST": {"permissions": ["budget", "reports"], "level": 2},
    "standard": {"permissions": ["read-only"], "level": 1},
    "EMPLOYEE": {"permissions": ["read-only"], "level": 1},
    "Read Only": {"permissions": ["read-only"], "level": 1},
    "Email Only": {"permissions": ["limited"], "level": 0}
}

# --- Normalize keys for case-insensitive lookups ---
NORMALIZED_ACCESS_RULES = {k.lower(): v for k, v in ACCESS_RULES.items()}

ROLE_PATTERNS = {
    "admin": r"admin",
    "developer": r"developer",
    "finance_editor": r"finance[_\s-]?editor",
    "finance_analyst": r"finance[_\s-]?analyst",
    "finance": r"finance",
    "employee": r"employee",
    "standard": r"standard",
    "read only": r"read\s*only",
    "email only": r"email\s*only"
}

def _map_token_to_canonicals(token):
    t = token.strip().lower()
    matched = set()
    # Exact match to any normalized rule key
    if t in NORMALIZED_ACCESS_RULES:
        matched.add(t)
    # Regex-based pattern matching
    for canonical, pattern in ROLE_PATTERNS.items():
        if re.search(pattern, t, re.IGNORECASE):
            matched.add(canonical)
    # fallback: try partial matches (splitting by non-alphanumerics)
    if not matched:
        parts = re.split(r'[^a-z0-9]+', t)
        for p in parts:
            if p in NORMALIZED_ACCESS_RULES:
                matched.add(p)
    # if still nothing, mark as standard or unknown
    if not matched:
        matched.add("standard")
    return matched

def analyze_access_levels(df, sheet_name):
    access_col = ACCESS_COLUMN_MAP.get(sheet_name)
    if not access_col or access_col not in df.columns:
        return df

    def parse_roles(cell):
        tokens = [tok.strip() for tok in re.split(r'[,;/]', str(cell)) if tok.strip()]
        canonical_roles = set()
        for tok in tokens:
            canonical_roles.update(_map_token_to_canonicals(tok))
        return sorted(canonical_roles)

    df['NormalizedRoles'] = df[access_col].astype(str).apply(parse_roles)

    def compute_effective_level(role_list):
        levels = [NORMALIZED_ACCESS_RULES[r]['level'] for r in role_list if r in NORMALIZED_ACCESS_RULES]
        return max(levels) if levels else 0

    def compute_effective_permissions(role_list):
        perms = set()
        for r in role_list:
            if r in NORMALIZED_ACCESS_RULES:
                perms.update(NORMALIZED_ACCESS_RULES[r]['permissions'])
        return sorted(perms)

    df['EffectiveLevel'] = df['NormalizedRoles'].apply(compute_effective_level)
    df['EffectivePermissions'] = df['NormalizedRoles'].apply(compute_effective_permissions)

    return df


# =====================================================
# Tabs
# =====================================================
def home_tab_ui(access_file_state, hr_file_state):
    with gr.Blocks() as home:
        gr.Markdown("""
        #  Business Application Access Management
        Welcome! Upload the required files once here.  
        These files will be used automatically across all tabs.
        """)

        with gr.Row():
            access_input = gr.File(label="Upload Access Log File (Excel/CSV)")
            hr_input = gr.File(label="Upload HR Master File (Excel/CSV)")

        def save_files(access_file, hr_file):
            return access_file, hr_file

        access_input.change(save_files, inputs=[access_input, hr_input], outputs=[access_file_state, hr_file_state])
        hr_input.change(save_files, inputs=[access_input, hr_input], outputs=[access_file_state, hr_file_state])

        gr.Markdown("✅ Files uploaded here will be shared across tabs.")
    return home

def staff_verification_ui(access_file_state, hr_file_state):
    with gr.Blocks() as verify_tab:
        gr.Markdown("## Business Application Access Management - Staff Verification")

        submit_btn = gr.Button("Run Verification")
        sheet_filter = gr.Dropdown(label="Filter by Sheet", choices=[], value=None)
        mismatch_filter = gr.Checkbox(label="Show only mismatches", value=False)
        column_selector = gr.CheckboxGroup(label="Select columns to display", choices=[], interactive=True)
        output_df = gr.DataFrame(label="Verified Access Records", interactive=False)
        download_dropdown = gr.Dropdown(label="Download Verified Sheet", choices=[], value=None)
        download_file = gr.File(label="Download CSV")

        full_df_state = gr.State(pd.DataFrame())
        results_state = gr.State({})

        def process_files(access_file, hr_file):
            if access_file is None or hr_file is None:
                raise ValueError("Please upload both files first.")
            results, combined_df = verify_access_by_sheet(access_file, hr_file)
            sheet_names = list(results.keys())
            available_columns = list(combined_df.columns)
            return (
                gr.update(choices=sheet_names, value=None),
                combined_df,
                gr.update(choices=available_columns, value=['AccessStatus', 'Employment Status']),
                results,
                gr.update(choices=sheet_names, value=None)
            )

        submit_btn.click(
            process_files,
            inputs=[access_file_state, hr_file_state],
            outputs=[sheet_filter, full_df_state, column_selector, results_state, download_dropdown]
        )

        def filter_dataframe(df, sheet_name, mismatches_only, selected_columns):
            if df.empty:
                return df
            filtered = df.copy()
            if sheet_name:
                filtered = filtered[filtered['Platform'] == sheet_name]
            if mismatches_only:
                filtered = filtered[filtered['AccessStatus'] != 'Valid Staff']
            display_columns = [col for col in selected_columns if col in filtered.columns]
            return filtered[display_columns]

        def download_by_sheet(sheet_name, results):
            if not results or sheet_name not in results:
                raise ValueError("No data found for this sheet.")
            df = results[sheet_name]
            tmp_path = os.path.join(tempfile.mkdtemp(), f"{sheet_name}_verified.csv")
            df.to_csv(tmp_path, index=False)
            return tmp_path

        sheet_filter.change(filter_dataframe, inputs=[full_df_state, sheet_filter, mismatch_filter, column_selector], outputs=output_df)
        mismatch_filter.change(filter_dataframe, inputs=[full_df_state, sheet_filter, mismatch_filter, column_selector], outputs=output_df)
        column_selector.change(filter_dataframe, inputs=[full_df_state, sheet_filter, mismatch_filter, column_selector], outputs=output_df)
        download_dropdown.change(download_by_sheet, inputs=[download_dropdown, results_state], outputs=[download_file])

    return verify_tab

def access_level_analysis_ui(access_file_state):
    with gr.Blocks() as access_tab:
        gr.Markdown("## Business Application Access Management - Access Level")

        analyze_btn = gr.Button("Analyze Access Levels")
        sheet_selector = gr.Dropdown(label="Select Sheet", choices=[], value=None)
        column_selector = gr.CheckboxGroup(label="Select Columns to Display", choices=[], interactive=True)
        output_df = gr.DataFrame(label="Access Level Details", interactive=False)
        download_dropdown = gr.Dropdown(label="Download Analyzed Sheet", choices=[], value=None)
        download_file = gr.File(label="Download CSV")

        full_df_state = gr.State(pd.DataFrame())
        results_state = gr.State({})

        def process_access(access_file):
            if access_file is None:
                raise ValueError("Please upload the Access Log file first.")
            ext = os.path.splitext(access_file.name)[-1].lower()
            if ext in ['.xls', '.xlsx']:
                xls = pd.ExcelFile(access_file)
                results = {}
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    results[sheet] = analyze_access_levels(df, sheet)
                combined = pd.concat([df.assign(Sheet=sheet) for sheet, df in results.items()])
                sheet_names = list(results.keys())
                available_columns = list(combined.columns)
                return (
                    gr.update(choices=sheet_names, value=None),
                    combined,
                    gr.update(choices=available_columns, value=['Sheet', 'EffectiveLevel', 'EffectivePermissions']),
                    results,
                    gr.update(choices=sheet_names, value=None)
                )
            else:
                raise ValueError("Upload an Excel file containing multiple sheets.")

        def filter_sheet(df, sheet_name, selected_columns):
            if df.empty:
                return df
            filtered = df.copy()
            if sheet_name and 'Sheet' in df.columns:
                filtered = filtered[filtered['Sheet'] == sheet_name]
            display_columns = [c for c in selected_columns if c in filtered.columns]
            return filtered[display_columns]

        def download_by_sheet(sheet_name, results):
            if not results or sheet_name not in results:
                raise ValueError("No data found for this sheet.")
            df = results[sheet_name]
            tmp_path = os.path.join(tempfile.mkdtemp(), f"{sheet_name}_analysis.csv")
            df.to_csv(tmp_path, index=False)
            return tmp_path

        analyze_btn.click(process_access, inputs=[access_file_state], outputs=[sheet_selector, full_df_state, column_selector, results_state, download_dropdown])
        sheet_selector.change(filter_sheet, inputs=[full_df_state, sheet_selector, column_selector], outputs=output_df)
        column_selector.change(filter_sheet, inputs=[full_df_state, sheet_selector, column_selector], outputs=output_df)
        download_dropdown.change(download_by_sheet, inputs=[download_dropdown, results_state], outputs=[download_file])

    return access_tab

# =====================================================
# Main Gradio Interface
# =====================================================
with gr.Blocks() as demo:
    access_file_state = gr.State()
    hr_file_state = gr.State()

    with gr.Tab("🏠 Home"):
        home_tab_ui(access_file_state, hr_file_state)
    with gr.Tab("👥 Staff Verification"):
        staff_verification_ui(access_file_state, hr_file_state)
    with gr.Tab("🔐 Access Level "):
        access_level_analysis_ui(access_file_state)

# =====================================================
# FastAPI Integration
# =====================================================
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

@app.get("/{path_name:path}")
async def catch_all(path_name: str):
    return PlainTextResponse("Not Found", status_code=404)
