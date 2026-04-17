# %%
# !pip install streamlit
# !pip install base64

import streamlit as st
import base64
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from bokeh.plotting import figure
import pydeck as pdk
import graphviz
from datetime import datetime

# 1. 必須放在最前面！
st.set_page_config(
    page_title="Streamlitアプリケーション",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help':'https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config',
        'Report a bug':'https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config',
        'About':'Streamlitでアプリを作成しよう'
    }
)

# # 2. 定義讀取圖片的函數
# def get_base64_image(image_path):
#     # 建議加上 try-except，防止圖片路徑錯誤導致程式崩潰
#     try:
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode()
#     except FileNotFoundError:
#         return ""

# image_path = './static/images/polar_bear.jpg'
# encoded_image = get_base64_image(image_path)

# # 3. 設定 CSS（加入 blend-mode 讓背景半透明效果生效）
# if encoded_image:
#     css = f'''
#     <style>
#         .stApp {{
#             background-image: url("data:image/jpg;base64,{encoded_image}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#             background-color: rgba(255, 255, 255, 0.6); /* 這裡控制白色遮罩濃度 */
#             background-blend-mode: overlay;            /* 必須加這行，顏色才會跟圖片疊加 */
#         }}
#         .stApp > header {{
#             background-color: transparent;
#         }}
#     </style>
#     '''
#     st.markdown(css, unsafe_allow_html=True)



# %%
from functions.multi_pages import multi_pages
multi_pages()

# %%
# 4. 顯示內容

st.set_page_config(layout="wide")

image_path = './static/images/polar_bear.jpg'
with open(image_path, 'rb') as f:
    image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
data = {
    'ProductID': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'ProductName': ['Product A','Product B','Product C','Product D',
                    'Product E','Product F','Product G','Product H',
                    'Product I','Product J','Product K','Product L',
                    'Product M','Product N','Product O'],
    'Price': [100,200,150,300,250,180,220,270,190,310,120,280,350,240,200],
    'InStock': [True, False, True, True, False, True, False, True, False, True, True, True, False, True, False],
    'Category': ['Electronic','Clothing','Electronic','Books','Kitchen',
                 'Electronics','Clothing','Books','Kitchen',
                 'Electronics','Electronics','Books','Clothing','Kitchen',                 
                 'Electronics'],
    'ReleaseDate': [datetime(2023,1,1), datetime(2022,12,1),datetime(2023,2,15),datetime(2022,11,20),
                    datetime(2023,3,10), datetime(2023,4,5),datetime(2022,9,12),datetime(2022,8,25),
                    datetime(2023,6,8), datetime(2023,7,15),datetime(2022,7,1),datetime(2022,6,15),
                    datetime(2023,8,20), datetime(2023,9,5),datetime(2023,10,10)],
    'Features': [['Feature 1', 'Feature 2'],['Feature 3'],
                 ['Feature 1', 'Feature 4'],['Feature 5'],
                 ['Feature 2'],['Feature 1', 'Feature 3'],
                 ['Feature 2'],['Feature 4'],
                 ['Feature 1'],['Feature 2'],['Feature 1', 'Feature 2'],['Feature 3'],
                 ['Feature 4'],['Feature 5'],
                 ['Feature 2']],
    'Link': ['https://www.producta.com','https://www.productb.com',
             'https://www.productc.com','https://www.productd.com',
             'https://www.producte.com','https://www.productf.com',
             'https://www.productg.com','https://www.producth.com',
             'https://www.producti.com','https://www.productj.com',
             'https://www.productk.com','https://www.productl.com',
             'https://www.productm.com','https://www.productn.com',
             'https://www.producto.com'],
    'Image': [f'data:image/png;base64,{encoded_image}']*15,
    'SalesTrends': [[1000, 2000, 1500, 3000, 2500],[5000, 3000, 8500, 4000, 500],
                    [3000, 6000, 500, 500, 4500],[4000, 5000, 4500, 6000, 5500],
                    [5000, 6000, 5500, 7000, 6500],[2000, 3000, 2500, 4000, 3500],
                    [7000, 8000, 7500, 6000, 5500],[6000, 5000, 4500, 3000, 3500],
                    [8000, 7000, 8500, 9000, 9500],[9000, 8000, 7500, 7000, 6500],
                    [1200, 3200, 4200, 5200, 6200],[1500, 2500, 3500, 4500, 5500],
                    [1800, 2800, 3800, 4800, 5800],[2100, 3100, 4100, 5100, 26100],
                    [2400, 3400, 4400, 5400, 6400]],
    'SalesAmount': [10000, 20000, 15000, 30000, 25000,
                    18000, 22000, 27000, 19000, 31000,
                    15000, 25000, 35000, 45000, 55000]
        
}
df = pd.DataFrame(data)
column_config = {
    "ProductID": st.column_config.NumberColumn("製品ID"),
    "ProductName": st.column_config.TextColumn("製品名"),
    "Price": st.column_config.NumberColumn("販売価額", format="%.2f"),
    "InStock": st.column_config.CheckboxColumn("在庫有無"),
    "Cagegory": st.column_config.SelectboxColumn("カテゴリー", options=['Electronics', 'Clothing', 'Books', 'Kitchen']),
    "ReleaseDate": st. column_config.DatetimeColumn("発売日"),
    "Features": st. column_config.ListColumn("分類"),
    "Link": st.column_config.LinkColumn("製品URL"),
    "Image": st.column_config.ImageColumn("製品画像"),
    "SalesTrends": st.column_config.LineChartColumn("売上推移"),
    "SalesAmount": st.column_config.ProgressColumn("売上金額", format="%.0f 円", min_value=0, max_value=55000)
}
st.title("st.dataeditor")
st.data_editor(df, column_config=column_config, height=570, hide_index=True)

# %%
