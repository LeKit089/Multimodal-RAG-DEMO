import tabula
import pandas as pd
import os 
from pdf2image import convert_from_path
from PIL import Image

def convert_pdf_to_table_data(input_pdf_path,output_folder):
    try:
        tables=tabula.read_pdf(input_pdf_path,pages='all',multiple_tables=True,area=None,guess=True,silent=True,stream=True)
        structured_data=[]
        for idx,table in enumerate(tables):
            output_json_path=os.path.join(output_folder,f"table_{idx}.json")
            table.to_json(output_json_path,orient='records')
            structured_data.append(table.to_dict(orient='records'))


        return structured_data
    except Exception as e :
        print(f"Error processing{input_pdf_path}:{e}")
        return None
"""    
abc=convert_pdf_to_table_data("src/chatbot_web_demo/filtered_9012388.pdf","input")
print(abc)

"""