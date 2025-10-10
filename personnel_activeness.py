#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import gradio as gr
import os
import tempfile
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

def find_email_column(df):
    for col in df.columns:
        if 'email' in col.lower():
            return col
    raise ValueError("No email column found in the uploaded file.")

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

            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, f"{sheet_name}_verified.csv")
            merged_df.to_csv(file_path, index=False)

            message = "✅ All good — no unknown or former staff found." if merged_df['AccessStatus'].eq('Valid Staff').all() else "⚠️ Some access records are not matched to active staff."

            results[sheet_name] = (merged_df, message, file_path)
            combined_df.append(merged_df)
        except Exception as e:
            results[sheet_name] = (pd.DataFrame(), f"❌ Error processing sheet '{sheet_name}': {str(e)}", None)

    return results, pd.concat(combined_df, ignore_index=True) if combined_df else pd.DataFrame()

with gr.Blocks() as demo:
    gr.Markdown("## 🛡️ Staff Access Verifier")

    with gr.Row():
        access_file = gr.File(label="Access log file")
        hr_file = gr.File(label="HR Master file")

    submit_btn = gr.Button("Submit")
    sheet_filter = gr.Dropdown(label="Filter by Sheet", choices=[], value=None, interactive=True)
    mismatch_filter = gr.Checkbox(label="Show only mismatches", value=False)
    column_selector = gr.CheckboxGroup(label='Select columns to display', choices=[], interactive=True)
    output_df = gr.DataFrame(label="Verified Access Records", interactive=False)
    dropdown = gr.Dropdown(label="Download verified sheet", choices=[])
    download_file = gr.File(label="Download CSV")
    file_state = gr.State({})
    full_df_state = gr.State(pd.DataFrame())

    def process_files(access_file, hr_file):
        results, combined_df = verify_access_by_sheet(access_file, hr_file)
        sheet_names = list(results.keys())
        file_map = {k: v[2] for k, v in results.items() if v[2] is not None}
        available_columns = list(combined_df.columns)
        return(
            gr.update(choices=sheet_names,value=None),
            gr.update(choices=sheet_names,value=None),
            file_map,
            combined_df,
            gr.update(choices=available_columns, value=['AccessStatus','Employment Status'])
        )
        
    submit_btn.click(
        process_files,
        inputs=[access_file, hr_file],
        outputs=[sheet_filter, dropdown, file_state, full_df_state, column_selector]
    )

    def filter_dataframe(df, sheet_name, mismatches_only,selected_columns):
        if df.empty:
            return df
        filtered = df.copy()
        if sheet_name:
            filtered = filtered[filtered['Platform'] == sheet_name]
        if mismatches_only:
            filtered = filtered[filtered['AccessStatus'] != 'Valid Staff']

        # ✅ Choose which columns to display
        display_columns = [col for col in selected_columns if col in filtered.columns]
        return filtered[display_columns]


    sheet_filter.change(filter_dataframe, inputs=[full_df_state, sheet_filter, mismatch_filter,column_selector], outputs=output_df)
    mismatch_filter.change(filter_dataframe, inputs=[full_df_state, sheet_filter, mismatch_filter,column_selector], outputs=output_df)
    column_selector.change(filter_dataframe, inputs=[full_df_state, sheet_filter, mismatch_filter,column_selector], outputs=output_df)

    def download_selected(sheet_name, file_map):
        return file_map.get(sheet_name, None)

    dropdown.change(download_selected, inputs=[dropdown, file_state], outputs=download_file)

app = FastAPI()
app.mount("/", WSGIMiddleware(demo.app))
