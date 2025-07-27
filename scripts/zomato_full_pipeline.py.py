import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler
import umap
from mpl_toolkits.mplot3d import Axes3D
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.cluster import KMeans
import optuna
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import warnings
import matplotlib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
import time
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# 데이터 불러오기
food_df = pd.read_csv("food.csv")
menu_df = pd.read_csv("menu.csv")
orders_df = pd.read_csv("orders.csv")
restaurant_df = pd.read_csv("restaurant.csv")
users_df = pd.read_csv("users.csv")

##users 전처리
users_df.drop(columns=['Unnamed: 0', 'name', 'email', 'password'], inplace=True)

# Convert 'user_id' to category
users_df['user_id'] = users_df['user_id'].astype('category')
# 주문 + 레스토랑 정보 병합
merged_res= orders_df.merge(restaurant_df, left_on="r_id", right_on="id", how="left")

print(merged_res.shape)
print(merged_res.columns)


### 불필요한 컬럼 삭제
merged_res.drop(columns=['Unnamed: 0_x','Unnamed: 0_y'], inplace=True)
merged_res.drop(columns=['lic_no','link','menu'], inplace=True)
print(merged_res.shape)
print(merged_res.columns)


### 결측치 처리
print(merged_res.isnull().sum())
cleaned_df = merged_res.dropna()

#식별자 데이터타입 변경
cleaned_df['user_id'] = cleaned_df['user_id'].astype(int).astype('category')
cleaned_df['r_id'] = cleaned_df['r_id'].astype(int).astype('category')
cleaned_df['id'] = cleaned_df['id'].astype(int).astype('category')

#currency 줄바꿈 제거, USD 제거
cleaned_df['currency'] = cleaned_df['currency'].str.strip()
cleaned_df = cleaned_df[cleaned_df['currency'] != 'USD']

cleaned_df['cost'] = cleaned_df['cost'].replace('[^0-9]', '', regex=True).astype(float).astype(int)

region_map = {
    'Delhi': 'North India', 'Chandigarh': 'North India', 'Jaipur': 'North India',
    'Lucknow': 'North India', 'Ludhiana': 'North India',

    'Mumbai': 'West India', 'Pune': 'West India', 'Ahmedabad': 'West India', 'Surat': 'West India',

    'Bangalore': 'South India', 'Hyderabad': 'South India', 'Chennai': 'South India', 'Kochi': 'South India',

    'Kolkata': 'East India', 'Bhubaneswar': 'East India', 'Patna': 'East India',

    'Bhopal': 'Central India', 'Indore': 'Central India', 'Raipur': 'Central India',

    'Guwahati': 'North-East India', 'Shillong': 'North-East India', 'Imphal': 'North-East India'
}

# restaurant의 city 컬럼에서 실제 도시명만 추출 (예: 'GOTA,Ahmedabad' → 'Ahmedabad')
cleaned_df['parsed_city'] = cleaned_df['city'].apply(lambda x: x.split(',')[-1].strip())

# parsed_city 기준으로 region 맵핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(region_map)

mapped_ratio = cleaned_df['region'].notna().mean()
unmapped_sample = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

cleaned_df['parsed_city'] = cleaned_df['city'].apply(lambda x: x.split(',')[-1].strip())

replace_dict = {
    'Noida-1': 'Noida',
    'Vizag': 'Visakhapatnam',
    'Gurugram': 'Gurgaon',
    'Trivandrum': 'Thiruvananthapuram',
    'Bangalore': 'Bengaluru'
}
cleaned_df['parsed_city'] = cleaned_df['parsed_city'].replace(replace_dict)

cleaned_df['parsed_city'] = cleaned_df['parsed_city'].str.strip().str.title()
# 도시명 표준화를 위한 replace dictionary (오타, 약어 등 보정)
# 도시명 표준화를 위한 replace dictionary (오타, 약어 등 보정)
replace_dict = {
    'Noida-1': 'Noida',
    'Vizag': 'Visakhapatnam',
    'Gurugram': 'Gurgaon',
    'Trivandrum': 'Thiruvananthapuram',
    'Bangalore': 'Bengaluru'
}

# 1. 쉼표가 포함된 경우 → 가장 마지막 값 (대도시) 기준
cleaned_df['parsed_city'] = cleaned_df['city'].apply(lambda x: x.split(',')[-1].strip())

# 2. replace_dict 기반 정제
cleaned_df['parsed_city'] = cleaned_df['parsed_city'].replace(replace_dict)

# 3. 대소문자 및 공백 정리
cleaned_df['parsed_city'] = cleaned_df['parsed_city'].str.strip().str.title()

# 정제된 도시명 개수와 상위 10개 출력
num_cleaned_cities = cleaned_df['parsed_city'].nunique()
cleaned_city_sample = cleaned_df['parsed_city'].value_counts().head(10)

# 확장된 region_map (표준화된 도시 기준)
extended_region_map = {
    # North India
    'Delhi': 'North India', 'Chandigarh': 'North India', 'Jaipur': 'North India',
    'Lucknow': 'North India', 'Ludhiana': 'North India', 'Noida': 'North India',
    'Gurgaon': 'North India', 'Dehradun': 'North India',

    # West India
    'Mumbai': 'West India', 'Pune': 'West India', 'Ahmedabad': 'West India',
    'Surat': 'West India', 'Nagpur': 'West India', 'Bikaner': 'West India',

    # South India
    'Bengaluru': 'South India', 'Hyderabad': 'South India', 'Chennai': 'South India',
    'Kochi': 'South India', 'Coimbatore': 'South India', 'Vijayawada': 'South India',
    'Mysore': 'South India', 'Visakhapatnam': 'South India', 'Thiruvananthapuram': 'South India',

    # East India
    'Kolkata': 'East India', 'Bhubaneswar': 'East India', 'Patna': 'East India',

    # Central India
    'Bhopal': 'Central India', 'Indore': 'Central India', 'Raipur': 'Central India',

    # North-East India
    'Guwahati': 'North-East India', 'Shillong': 'North-East India', 'Imphal': 'North-East India'
}

# 표준화된 도시명 기준 region 컬럼 생성
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑 성공률과 매핑 안 된 도시 수 확인
region_mapping_rate = cleaned_df['region'].notna().mean()
unmapped_cities = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 추가 매핑 (누락된 주요 도시들)
additional_region_map = {
    # North India
    'Varanasi': 'North India', 'Faridabad': 'North India', 'Kanpur': 'North India',
    'Allahabad': 'North India', 'Agra': 'North India',

    # West India
    'Vadodara': 'West India', 'Udaipur': 'West India', 'Gondal': 'West India', 'North-Goa': 'West India',

    # South India
    'Madurai': 'South India'
}

# 기존 매핑 딕셔너리에 병합
extended_region_map.update(additional_region_map)

# region 컬럼 다시 매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 새로운 매핑 성공률 확인
region_mapping_rate_updated = cleaned_df['region'].notna().mean()
unmapped_cities_updated = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)


# 추가 매핑 (남은 주요 도시들)
additional_region_map_2 = {
    # North India
    'Amritsar': 'North India', 'Gorakhpur': 'North India', 'Bareilly': 'North India',
    'Meerut': 'North India',

    # West India
    'Aurangabad': 'West India', 'Jodhpur': 'West India', 'Central-Goa': 'West India',

    # East India
    'Ranchi': 'East India', 'Adityapur': 'East India',

    # South India
    'Pondicherry': 'South India'
}

# 기존 매핑 딕셔너리에 병합
extended_region_map.update(additional_region_map_2)

# region 컬럼 다시 매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 업데이트된 매핑 성공률 확인
final_region_mapping_rate = cleaned_df['region'].notna().mean()
remaining_unmapped = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 마지막 추가 매핑 (최상위 미매핑 도시들)
final_additional_map = {
    # Central India
    'Gwalior': 'Central India', 'Jabalpur': 'Central India',

    # West India
    'Kolhapur': 'West India',

    # North India
    'Rohtak': 'North India', 'Sultanpur': 'North India',

    # South India
    'Trichy': 'South India', 'Guntur': 'South India',
    'Tirupati': 'South India', 'Tirupur': 'South India',

    # East India
    'Siliguri': 'East India'
}

# 매핑 딕셔너리 업데이트
extended_region_map.update(final_additional_map)

# region 컬럼 다시 매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 확인
final_mapping_ratio = cleaned_df['region'].notna().mean()
still_unmapped = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 최종 매핑 확장 (미매핑 Top 10 도시)
final_region_map_addition = {
    # South India
    'Salem': 'South India', 'Kozhikode': 'South India', 'Thrissur': 'South India',
    'Erode': 'South India',

    # North India
    'Jhansi': 'North India', 'Hisar': 'North India', 'Aligarh': 'North India', 'Panipat': 'North India',

    # East India
    'Cuttack': 'East India',

    # West India
    'Ajmer': 'West India'
}

# 확장된 매핑 추가
extended_region_map.update(final_region_map_addition)

# region 컬럼 재매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 재확인
final_mapping_rate_post_expansion = cleaned_df['region'].notna().mean()
still_missing_after_expansion = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 추가 매핑 (Top 10 미매핑 도시)
final_region_map_addition_2 = {
    # South India
    'Hubli': 'South India', 'Nellore': 'South India', 'Vellore': 'South India', 'Mangaluru': 'South India',
    'Belgaum': 'South India',

    # Central India
    'Bhilai': 'Central India',

    # West India
    'Anand': 'West India', 'Solapur': 'West India',

    # North India
    'Yamuna-Nagar': 'North India', 'Mansa': 'North India'
}

# 업데이트
extended_region_map.update(final_region_map_addition_2)

# 재매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 확인
final_mapping_rate_updated = cleaned_df['region'].notna().mean()
new_top_unmapped = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

bulk_region_mapping = {
    # North India
    'Bathinda': 'North India', 'Haldwani': 'North India', 'Karnal': 'North India',
    'Alwar': 'North India', 'Bahadurgarh': 'North India', 'Ambala': 'North India',
    'Mathura': 'North India', 'Phagwara': 'North India', 'Sirsa': 'North India',
    'Haridwar': 'North India', 'Kurukshetra': 'North India', 'Saharanpur': 'North India',
    'Sonipat': 'North India', 'Firozpur': 'North India', 'Sikar': 'North India',
    'Hoshiarpur': 'North India', 'Faizabad': 'North India', 'Pathankot': 'North India',

    # South India
    'Anantapur': 'South India', 'Thanjavur': 'South India', 'Kakinada': 'South India',
    'Manipal': 'South India', 'Rajahmundry': 'South India', 'Kollam': 'South India',
    'Tirunelveli': 'South India', 'Kurnool': 'South India', 'Thoothukudi': 'South India',
    'Davanagere': 'South India', 'Nagercoil': 'South India', 'Tumakuru': 'South India',

    # East India
    'Bokaro': 'East India', 'Durgapur': 'East India', 'Bhagalpur': 'East India',
    'Asansol': 'East India', 'Muzaffarpur': 'East India', 'Berhampur': 'East India',
    'Gaya': 'East India',

    # West India
    'Morbi': 'West India', 'Amravati': 'West India', 'Sangli': 'West India',
    'Jamnagar': 'West India', 'Bhavnagar': 'West India', 'Gandhidham': 'West India',
    'Latur': 'West India',

    # Central India
    'Bilaspur': 'Central India', 'Sagar': 'Central India', 'Ujjain': 'Central India',
    'Rewa': 'Central India',

    # North-East India
    'Agartala': 'North-East India', 'Jorhat': 'North-East India'
}

# 매핑 통합
extended_region_map.update(bulk_region_mapping)

# region 컬럼 업데이트
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 확인
final_bulk_mapping_rate = cleaned_df['region'].notna().mean()
final_unmapped_sample = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

bulk_region_mapping_2 = {
    # West India
    'Mehsana': 'West India', 'Akola': 'West India',

    # South India
    'Kottayam': 'South India', 'Shivamogga': 'South India',

    # North India
    'Rishikesh': 'North India', 'Roorkee': 'North India', 'Rewari': 'North India',
    'Shahjahanpur': 'North India',

    # Central India
    'Satna': 'Central India',

    # East India
    'Puri': 'East India'
}

# 매핑 테이블 업데이트
extended_region_map.update(bulk_region_mapping_2)

# region 컬럼 재매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 확인
final_mapping_rate_v2 = cleaned_df['region'].notna().mean()
remaining_unmapped_v2 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 50개 도시 region 매핑
bulk_region_mapping_3 = {
    # West India
    'Daman': 'West India', 'Jalgaon': 'West India', 'Nanded': 'West India',
    'Halol': 'West India', 'Chandrapur': 'West India', 'Ahmednagar': 'West India',
    'Lonavala': 'West India', 'Satara': 'West India', 'Valsad': 'West India',
    'Ratnagiri': 'West India',

    # South India
    'Karur': 'South India', 'Kadapa': 'South India', 'Kumbakonam': 'South India',
    'Dindigul': 'South India', 'Ongole': 'South India', 'Palakkad': 'South India',
    'Khammam': 'South India', 'Ballari': 'South India', 'Kannur': 'South India',
    'Alappuzha': 'South India', 'Kalady': 'South India', 'Nizamabad': 'South India',
    'Eluru': 'South India', 'Namakkal': 'South India', 'Kalaburagi': 'South India',
    'Hassan': 'South India',

    # North India
    'Rudrapur': 'North India', 'Muzaffarnagar': 'North India', 'Kota': 'North India',
    'Bhiwadi': 'North India', 'Bharatpur': 'North India', 'Jammu': 'North India',
    'Sri-Ganganagar': 'North India', 'Bhilwara': 'North India', 'Solan': 'North India',
    'Bhiwani': 'North India',

    # East India
    'Rourkela': 'East India', 'Balasore': 'East India', 'Purnea': 'East India',
    'Sambalpur': 'East India', 'Bardhaman': 'East India', 'Deoghar': 'East India',
    'Darbhanga': 'East India', 'Kanchrapara': 'East India',

    # Central India
    'Ratlam': 'Central India', 'Gondia': 'Central India',

    # North-East India
    'Dibrugarh': 'North-East India'
}

# 매핑 추가 적용
extended_region_map.update(bulk_region_mapping_3)

# region 컬럼 재매핑
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 및 미매핑 상위 10개 확인
mapping_rate_50_added = cleaned_df['region'].notna().mean()
next_unmapped_after_50 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 매핑 사전 업데이트 (다음 50개 도시)
bulk_region_mapping_4 = {
    # West India
    'Nadiad': 'West India', 'Wardha': 'West India', 'Bharuch': 'West India',
    'Junagadh': 'West India', 'Karad': 'West India', 'Navsari': 'West India',
    'Dhule': 'West India',

    # South India
    'Karimnagar': 'South India', 'Bhimavaram': 'South India', 'Thiruvallur': 'South India',
    'Virudhunagar': 'South India', 'Dharmapuri': 'South India', 'Karaikkudi': 'South India',
    'Tiruvannamalai': 'South India', 'Pudukkottai': 'South India', 'Changanassery': 'South India',
    'Chitradurga': 'South India', 'Cuddalore': 'South India', 'Sivakasi': 'South India',
    'Madanapalle': 'South India', 'Vizianagaram': 'South India', 'Mandya': 'South India',
    'Mancherial': 'South India', 'Ambur': 'South India', 'Proddatur': 'South India',

    # Central India
    'Dhar': 'Central India', 'Chhindwara': 'Central India', 'Ambikapur': 'Central India',

    # North India
    'Khanna': 'North India', 'Kashipur': 'North India', 'Moga': 'North India',
    'Moradabad': 'North India', 'Sangrur': 'North India', 'Orai': 'North India',
    'Barnala': 'North India', 'Baddi': 'North India', 'Firozabad': 'North India',
    'Jaunpur': 'North India', 'Kapurthala': 'North India', 'Rae-Bareli': 'North India',
    'Etawah': 'North India', 'Pali': 'North India', 'Hanumangarh': 'North India',

    # East India
    'Habra': 'East India', 'Kharagpur': 'East India',

    # North-East India
    'Bagdogra': 'North-East India', 'Tezpur': 'North-East India',
    'Lakhimpur': 'North-East India', 'Sivasagar': 'North-East India', 'Nagaon': 'North-East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_4)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 잔여 미매핑 확인
mapping_rate_v3 = cleaned_df['region'].notna().mean()
remaining_unmapped_v3 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)


# 다음 50개 도시 매핑 사전
bulk_region_mapping_5 = {
    # South India
    'Krishnagiri': 'South India', 'Rajapalayam': 'South India', 'Mahbubnagar': 'South India',
    'Chittoor': 'South India', 'Ooty': 'South India', 'Bijapur': 'South India',
    'Nandyal': 'South India', 'Srikakulam': 'South India', 'Machilipatnam': 'South India',
    'Kothamanagalam': 'South India', 'Nalgonda': 'South India', 'Chikmagalur': 'South India',

    # North India
    'Batala': 'North India', 'Bharabanki': 'North India', 'Jind': 'North India',
    'Budaun': 'North India', 'Ropar': 'North India', 'Rampur': 'North India',
    'Abohar': 'North India', 'Farrukhabad': 'North India', 'Hapur': 'North India',
    'Bulandshahr': 'North India', 'Sitapur': 'North India', 'Beawar': 'North India',
    'Muktsar': 'North India', 'Kaithal': 'North India', 'Kotdwar': 'North India',
    'Mirzapur': 'North India', 'Faridkot': 'North India', 'Pilibhit': 'North India',

    # West India
    'Yavatmal': 'West India', 'Beed': 'West India', 'Veraval': 'West India',
    'Bhuj': 'West India', 'Rajkot': 'West India', 'Ankleshwar': 'West India',

    # Central India
    'Katni': 'Central India', 'Vidisha': 'Central India', 'Korba': 'Central India',
    'Dewas': 'Central India', 'Chhatarpur': 'Central India', 'Hoshangabad': 'Central India',

    # East India
    'Jamshedpur': 'East India', 'Begusarai': 'East India', 'Berhampore': 'East India',
    'Malda': 'East India', 'Biharsharif': 'East India', 'Jalpaiguri': 'East India',

    # North-East India
    'Silchar': 'North-East India', 'Bagdogra': 'North-East India', 'Duliajan': 'North-East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_5)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 확인 및 남은 미매핑 도시 출력
final_mapping_rate_94 = cleaned_df['region'].notna().mean()
remaining_unmapped_v4 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_6 = {
    # South India
    'Chidambaram': 'South India', 'Kovilpatti': 'South India', 'Raichur': 'South India',

    # North India
    'Banda': 'North India', 'Gonda': 'North India', 'Basti': 'North India',

    # East India
    'Arrah': 'East India', 'Aurangabad_Bihar': 'East India', 'Purulia': 'East India',

    # North-East India
    'Tinsukia': 'North-East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_6)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_95 = cleaned_df['region'].notna().mean()
still_unmapped_v5 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_7 = {
    # North India
    'Fatehpur': 'North India', 'Chittorgarh': 'North India', 'Chhapra': 'North India',

    # West India
    'Himmatnagar': 'West India', 'Porbandar': 'West India', 'Baramati': 'West India',

    # East India
    'Dumka': 'East India', 'Motihari': 'East India', 'Medinipur': 'East India',

    # South India
    'Siddipet': 'South India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_7)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_96 = cleaned_df['region'].notna().mean()
still_unmapped_v6 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_8 = {
    # South India
    'Karunagappaly': 'South India', 'Ramanathapuram': 'South India', 'Tadepalligudem': 'South India',

    # North India
    'Azamgarh': 'North India', 'Hardoi': 'North India', 'Modinagar': 'North India',

    # West India
    'Parbhani': 'West India',

    # East India
    'Katihar': 'East India',

    # North-East India
    'Dimapur': 'North-East India', 'Bongaigaon': 'North-East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_8)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_97 = cleaned_df['region'].notna().mean()
still_unmapped_v7 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_9 = {
    # North India
    'Patiala': 'North India', 'Jhunjhunu': 'North India', 'Bijnor': 'North India',

    # South India
    'Gudivada': 'South India',

    # Central India
    'Jagdalpur': 'Central India',

    # West India
    'Bhusawal': 'West India', 'Gangapur-City': 'West India', 'Bardoli': 'West India',

    # East India
    'Siwan': 'East India', 'Samastipur': 'East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_9)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_98 = cleaned_df['region'].notna().mean()
still_unmapped_v8 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_10 = {
    # North India
    'Palampur': 'North India', 'Fatehabad': 'North India', 'Jagraon': 'North India', 'Bahraich': 'North India',

    # Central India
    'Khandwa': 'Central India', 'Neemuch': 'Central India', 'Guna': 'Central India',

    # North-East India
    'Alipurduar': 'North-East India',

    # South India
    'Ramagundam': 'South India', 'Bidar': 'South India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_10)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_97 = cleaned_df['region'].notna().mean()
still_unmapped_v9 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_11 = {
    # South India
    'Palani': 'South India', 'Kolar': 'South India',

    # Central India
    'Raigarh': 'Central India', 'Burhanpur': 'Central India', 'Shivpuri': 'Central India',

    # West India
    'Kopargaon': 'West India', 'Dahod': 'West India', 'Kishangarh': 'West India',

    # North India
    'Fatehgarh-Sahib': 'North India',

    # East India
    'Cooch-Behar': 'East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_11)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_975 = cleaned_df['region'].notna().mean()
still_unmapped_v10 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_12 = {
    # West India
    'Ichalkaranji': 'West India', 'Pusad': 'West India',

    # North India
    'Dharamshala': 'North India', 'Gurdaspur': 'North India',

    # Central India
    'Mandsaur': 'Central India', 'Betul': 'Central India', 'Rajnandgaon': 'Central India',

    # South India
    'Adoni': 'South India', 'Chalakkudy': 'South India', 'Tenkasi': 'South India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_12)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_98 = cleaned_df['region'].notna().mean()
still_unmapped_v11 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_13 = {
    # North India
    'Narnaul': 'North India', 'Maunath-Bhanjan': 'North India', 'Nangal': 'North India', 'Suratgarh': 'North India',

    # South India
    'Hospet': 'South India', 'Theni': 'South India', 'Tanuku': 'South India',
    'Bapatlachirala': 'South India', 'Thiruvarur': 'South India', 'Chikkaballapur': 'South India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_13)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_984 = cleaned_df['region'].notna().mean()
still_unmapped_v12 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 다음 미매핑 도시 매핑 사전
bulk_region_mapping_14 = {
    # North India
    'Mughalsarai': 'North India', 'Nainital': 'North India',

    # South India
    'Perinthalmanna': 'South India', 'Kunnamkullam': 'South India', 'Mayiladuthurai': 'South India',

    # East India
    'Bettiah': 'East India', 'Munger': 'East India', 'Saharsa': 'East India',

    # Central India
    'Morena': 'Central India',

    # West India
    'Bhandara': 'West India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_14)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_98 = cleaned_df['region'].notna().mean()
still_unmapped_v13 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 상위 50개 미매핑 도시 매핑 사전
bulk_region_mapping_15 = {
    # South India
    'Kumarakom': 'South India', 'Jagtial': 'South India', 'Doddaballapura': 'South India',
    'Cherthala': 'South India', 'Chengannur': 'South India', 'Pala': 'South India',
    'Narasaraopet': 'South India', 'Suryapet': 'South India', 'Madikeri': 'South India',
    'Warangal': 'South India', 'Thalassery': 'South India', 'Tuni': 'South India',
    'Guntakal': 'South India', 'Tadpatri': 'South India', 'Adilabad': 'South India',
    'Irinjalakuda': 'South India', 'Thodupuzha': 'South India',

    # North India
    'Bela-Pratapgarh': 'North India', 'Jhalawar': 'North India', 'Rajsamand': 'North India',
    'Malout': 'North India', 'Lalitpur': 'North India', 'Kannauj': 'North India',
    'Barmer': 'North India', 'Balrampur': 'North India',

    # East India
    'Balurghat': 'East India', 'Raiganj': 'East India', 'Giridih': 'East India',
    'Gopalganj': 'East India', 'Madhubani': 'East India', 'Chakdaha': 'East India',
    'Kendujhar': 'East India', 'Kishanganj': 'East India', 'Nabadwip': 'East India',

    # West India
    'Silvassa': 'West India', 'Surendranagar-Dudhrej': 'West India', 'Malegaon': 'West India',
    'Chalisgaon': 'West India', 'Uran-Islampur': 'West India', 'Udgir': 'West India',
    'Amreli': 'West India', 'Osmanabad': 'West India', 'Barshi': 'West India',
    'Vyara': 'West India', 'Godhra': 'West India', 'Washim': 'West India',

    # Central India
    'Narsinghpur': 'Central India', 'Waidhan': 'Central India',

    # North-East India
    'Aizawl': 'North-East India', 'Golaghat': 'North-East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_15)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 남은 미매핑 확인
final_mapping_rate_batch = cleaned_df['region'].notna().mean()
still_unmapped_batch = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 상위 50개 도시 지역 매핑 사전
bulk_region_mapping_16 = {
    # South India
    'Kottarakkara': 'South India', 'Kanyakumari': 'South India', 'Neyveli': 'South India',
    'Kodaikanal': 'South India', 'Arakkonam': 'South India', 'Gadag-Betigeri': 'South India',
    'Kamareddy': 'South India', 'Miryalaguda': 'South India', 'Kavali': 'South India',
    'Varkala': 'South India', 'Bhatkal': 'South India', 'Kadiri': 'South India',
    'Kottakkal': 'South India', 'Sirsi': 'South India', 'Muvattupuzha': 'South India',
    'Moodbidri': 'South India', 'Aruppukottai': 'South India',

    # North India
    'Bhadohi': 'North India', 'Gauriganj': 'North India', 'Baran': 'North India',
    'Shimla': 'North India', 'Shikohabad': 'North India', 'Dausa': 'North India',
    'Churu': 'North India', 'Hindaun': 'North India', 'Hansi': 'North India',
    'Tohana': 'North India',

    # East India
    'Dehri': 'East India', 'Buxar': 'East India', 'Bolpur': 'East India',
    'Raniganj': 'East India', 'Raghunathpur': 'East India', 'Jhargram': 'East India',
    'Balangir': 'East India', 'Jahanabad': 'East India', 'Ranaghat-Wb': 'East India',
    'Haldia': 'East India',

    # Central India
    'Itarsi': 'Central India', 'Bhind': 'Central India', 'Balaghat': 'Central India',
    'Sehore': 'Central India', 'Nagda': 'Central India',

    # West India
    'Khamgaon': 'West India', 'Chikhli': 'West India', 'Palanpur': 'West India',
    'Diu': 'West India', 'Chiplun': 'West India', 'Dhoraji': 'West India',
    'Buldana': 'West India', 'Shrirampur': 'West India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_16)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 잔여 미매핑 확인
final_mapping_rate_99 = cleaned_df['region'].notna().mean()
still_unmapped_50batch = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 마지막 미매핑 도시 일괄 매핑 사전
final_bulk_mapping = {
    # East India
    'Arambagh': 'East India', 'Krishnanagar': 'East India', 'Bhawanipatna': 'East India',

    # South India
    'Bhadrachalam': 'South India', 'Manjeri': 'South India', 'Rayachoty': 'South India',
    'Malappuram': 'South India', 'Nirmal': 'South India',

    # South/West 경계 (Bagalkot, Hinganghat → West India로 분류)
    'Bagalkot': 'West India', 'Hinganghat': 'West India'
}

# 매핑 적용
extended_region_map.update(final_bulk_mapping)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 및 미매핑 확인
final_mapping_rate_complete = cleaned_df['region'].notna().mean()
remaining_unmapped_final = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts()


# 미매핑 도시 50개 매핑
bulk_region_mapping_17 = {
    # South India
    'Nagapattinam': 'South India', 'Sindhanur': 'South India', 'Gokak': 'South India',
    'Kadayanallur': 'South India', 'Karwar': 'South India', 'Dharwad': 'South India',
    'Tiptur': 'South India', 'Gadwal': 'South India', 'Kayamkulam': 'South India',
    'Ramanagara': 'South India', 'Puttur': 'South India', 'Markapur': 'South India',
    'Koppal': 'South India', 'Kothagudem': 'South India', 'Nipani': 'South India',
    'Bantwal': 'South India', 'Bodhan-Rural': 'South India', 'Srivilliputhur': 'South India',
    'Kundapura': 'South India', 'Kasaragod': 'South India', 'Thiruvalla': 'South India',
    'Bodinayakanur': 'South India', 'Kumta': 'South India',

    # North India
    'Naraingarh': 'North India', 'Bundi': 'North India', 'Fazilka': 'North India',
    'Mandi-Dabwali': 'North India', 'Chandausi': 'North India', 'Jalaun': 'North India',
    'Sawai-Madhopur': 'North India', 'Mussoorie': 'North India', 'Budhwal': 'North India',
    'Tarn-Taran-Sahib': 'North India',

    # East India
    'Sasaram': 'East India', 'Basirhat': 'East India', 'Uluberia': 'East India',
    'Bongaon': 'East India', 'Murshidabad': 'East India',

    # West India
    'Sangamner': 'West India', 'Vapi': 'West India', 'Nandurbar': 'West India',
    'Visnagar': 'West India', 'Chopda': 'West India', 'Lonavla': 'West India',
    'Bilimora': 'West India',

    # Central India
    'Singrauli': 'Central India', 'Barwani': 'Central India',

    # North-East India
    'Gangtok': 'North-East India', 'Darjeeling': 'North-East India'
}

# 매핑 적용
extended_region_map.update(bulk_region_mapping_17)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 매핑률 및 잔여 미매핑 확인
final_mapping_rate_update = cleaned_df['region'].notna().mean()
still_unmapped_final50 = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts().head(10)

# 마지막 남은 10개 도시 지역 매핑
final_10_mapping = {
    # West India
    'South-Goa': 'West India', 'Dahanu': 'West India', 'Boisar': 'West India',
    'Mount-Abu': 'West India',

    # East India
    'Dhanbad': 'East India',

    # North India
    'Mandi-Gobindgarh': 'North India',

    # South India
    'Ranibennur': 'South India', 'Hampi': 'South India',

    # North-East India
    'Itanagar': 'North-East India', 'Kohima': 'North-East India'
}

# 매핑 적용
extended_region_map.update(final_10_mapping)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 확인 및 결측 확인
final_mapping_rate_100 = cleaned_df['region'].notna().mean()
final_unmapped_check = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts()

# 마지막 3개 도시 매핑
final_final_mapping = {
    'Manali': 'North India',
    'Naharlagun': 'North-East India',
    'Rangpo': 'North-East India'
}

# 매핑 적용
extended_region_map.update(final_final_mapping)
cleaned_df['region'] = cleaned_df['parsed_city'].map(extended_region_map)

# 최종 매핑률 및 남은 미매핑 확인
final_mapping_rate_absolute = cleaned_df['region'].notna().mean()
final_check_empty = cleaned_df[cleaned_df['region'].isna()]['parsed_city'].value_counts()

### 전처리 이어서
cleaned_df.drop(columns=['id'], inplace=True)
cleaned_df.drop(columns=['currency'], inplace=True)
cleaned_df.drop(columns=['address'], inplace=True)
cleaned_df.drop(columns=['parsed_city'], inplace=True)
cleaned_df['order_date'] = pd.to_datetime(cleaned_df['order_date'])
# '--'을 'Unrated'로 대체
cleaned_df['rating'] = cleaned_df['rating'].replace('--', 'Unrated')
# sales_amount 음수값 제거
cleaned_df = cleaned_df[cleaned_df['sales_amount'] >= 0]
rating_total = len(cleaned_df)
missing_rating_count = (cleaned_df['rating'] == 'Unrated').sum()
missing_rating_ratio = missing_rating_count / rating_total

print(f"결측치 (--) 개수: {missing_rating_count}")
print(f"비율: {missing_rating_ratio:.2%}")
# cuisine 전처리
#1. cuisine 값을 콤마 기준으로 리스트로 변환
cleaned_df['cuisine_list'] = cleaned_df['cuisine'].str.split(',')

# 2. explode()로 리스트를 세로로 늘리기
exploded_df = cleaned_df.explode('cuisine_list')

# 소문자화 + 앞뒤 공백 제거
exploded_df['cuisine_list'] = exploded_df['cuisine_list'].str.lower().str.strip()

## 1차 대분류 - 인도식 분류(north,south,indian)
#인도식 분류(north,south,indian)
north_india_cuisines = [
    "north indian", "mughlai", "awadhi", "lucknowi", "kashmiri", "punjabi"
]

south_india_cuisines = [
    "south indian", "andhra", "hyderabadi", "chettinad", "kerala",
    "mangalorean", "rayalaseema", "telangana"
]

india_cuisines = [
    "gujarati", "konkan", "maharashtrian", "malwani", "parsi", "rajasthani", "goan",
    "bengali", "bihari", "oriya",
    "north eastern", "khasi", "naga", "assamese",
    "sindhi"
]

def categorize_region(cuisine_list):
    if cuisine_list in north_india_cuisines:
        return "north indian"
    elif cuisine_list in south_india_cuisines:
        return "south indian"
    elif cuisine_list in india_cuisines:
        return "indian"
    else:
        return cuisine_list  # 나머지는 그대로 두기

# cuisine_list 앞뒤 공백 제거 + 소문자 상태여야 함!
exploded_df['region_categorized'] = exploded_df['cuisine_list'].apply(categorize_region)

# 이상치 리스트 (전처리된 기준)
invalid_cuisines = [
    "8:15 to 11:30 pm",
    "attractive combos available",
    "biryani - shivaji military hotel",
    "bowl company",
    "code valid on bill over rs.99",
    "default",
    "discount offer from garden cafe express kankurgachi",
    "free delivery ! limited stocks!",
    "max 2 combos per order!",
    "popular brand store",
    "special discount from (hotel swagath)",
    "svanidhi street food vendor",
    "use code jumbo30 to avail",
    "use code xpress121 to avail."
]

exploded_df = exploded_df[~exploded_df['region_categorized'].isin(invalid_cuisines)]

## 2차 대분류 - 글로벌 분류(아시아, 서양, 중동 분류)
#아시아, 서양, 중동 분류
#아시아, 서양, 중동 분류
asian_cuisines = [
    "asian", "bangladeshi", "bhutanese", "burmese", "chinese", "indian",
    "indonesian", "japanese", "korean", "malaysian", "mongolian", "nepalese",
    "singaporean", "sri lankan", "thai", "tibetan", "vietnamese"
]

western_cuisines = [
    "american", "australian", "british", "continental", "european", "french", "german",
    "greek", "italian", "italian-american", "portuguese", "south american", "spanish", "mediterranean"
]

middle_eastern_cuisines = [
    "arabian", "lebanese", "turkish", "afghani", "middle eastern"
]


def overwrite_region_category(cuisine, current_value):
    if current_value in ["north indian", "south indian", "indian"]:
        return current_value  # 인도 요리 그대로 유지
    elif cuisine in asian_cuisines:
        return "Asian"
    elif cuisine in western_cuisines:
        return "Western"
    elif cuisine in middle_eastern_cuisines:
        return "Middle Eastern"
    else:
        return current_value  # 나머지는 그대로 유지

exploded_df['region_categorized'] = exploded_df.apply(
    lambda row: overwrite_region_category(row['cuisine_list'], row['region_categorized']),
    axis=1
)
## 3차 대분류 - 나머지 (dessert, fast food, others)
preclassified = ["indian", "north indian", "south indian", "Asian", "Western", "Middle Eastern"]

india_foods = ["biryani", "chaat", "tandoor", "thalis", "sweets"]
dessert_foods = ["bakery", "desserts", "ice cream", "ice cream cakes", "paan", "waffle", "snacks"]
fast_foods = ["fast food", "street food"]
western_foods = ["pastas", "pizzas", "salads", "steakhouse"]
asia_foods = ["oriental", "pan-asian", "sushi"]
middle_eastern_foods = ["kebabs"]
others_foods = ["barbecue", "combo", "grill", "healthy food", "home food", "jain", "keto", "meat"]

def smart_category_overwrite(cuisine, current_category):
    if current_category.lower() in preclassified:
        return current_category  # 기존 분류 유지
    elif cuisine in india_foods:
        return "indian"
    elif cuisine in dessert_foods:
        return "dessert"
    elif cuisine in fast_foods:
        return "fast food"
    elif cuisine in western_foods:
        return "Western"
    elif cuisine in asia_foods:
        return "Asian"
    elif cuisine in middle_eastern_foods:
        return "Middle Eastern"
    elif cuisine in others_foods:
        return "others"
    else:
        return current_category  # 분류 기준 없으면 그대로 두기

#적용
exploded_df['region_categorized'] = exploded_df.apply(
    lambda row: smart_category_overwrite(row['cuisine_list'], row['region_categorized']),
    axis=1
)

## 4차 대분류 - 나머지의 나머지(beverages, others)

custom_category_map = {
    "juices": "beverages",
    "burgers": "fast food",
    "seafood": "others",
    "mexican": "Western",
    "cafe": "dessert",
    "coastal": "others",
    "haleem": "Middle Eastern",
    "african": "others",
    "tex-mex": "Western",
    "persian": "Middle Eastern",
    "tribal": "others",
    "bakery products": "dessert",
    "navratri special": "indian",
    "beverage": "beverages"
}

def update_region_category(current_value):
    # current_value가 매핑 대상이면 변경, 아니면 그대로
    return custom_category_map.get(current_value, current_value)

exploded_df['region_categorized'] = exploded_df['region_categorized'].apply(update_region_category)

exploded_df = exploded_df[exploded_df['region_categorized'] != 'grocery products']

exploded_df['region_categorized'] = exploded_df['region_categorized'].replace({
    "Western": "western",
    "Asian": "asian",
    "Middle Eastern": "middle eastern"
})


## 5차 분류 - healthy, seafood, meat/grill, combo삭제
# 덮어쓸 분류
healthy_set = {"healthy food", "keto", "jain", "home food"}
meatgrill_set = {"barbecue", "grill", "meat"}
seafood_set = {"seafood", "coastal"}

# 삭제 대상
delete_set = {"african", "tribal", "combo"}


def refine_other_category(row):
    cuisine = row['cuisine_list']
    category = row['region_categorized']

    if category == "others":
        if cuisine in healthy_set:
            return "healthy"
        elif cuisine in meatgrill_set:
            return "meat/grill"
        elif cuisine in seafood_set:
            return "seafood"
        elif cuisine in delete_set:
            return "delete"

    return category  # 기존 분류 그대로 유지

exploded_df['region_categorized'] = exploded_df.apply(refine_other_category, axis=1)
exploded_df = exploded_df[exploded_df['region_categorized'] != "delete"]

cols = ['sales_qty', 'sales_amount', 'cost']

for col in cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(exploded_df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 복사본 생성
log_iqr_df = exploded_df.copy()

# 로그 변환할 컬럼들
cols_to_log = ['sales_qty', 'sales_amount', 'cost']

# 로그 변환 (0이 있는 경우 대비해 log1p 사용)
for col in cols_to_log:
    log_iqr_df[f'log_{col}'] = np.log1p(log_iqr_df[col])

# IQR 이상치 제거 함수
def remove_outliers_iqr(df, columns, k=1.5):
    df_cleaned = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

# 로그변환한 컬럼들에 대해 IQR 제거
log_cols = [f'log_{col}' for col in cols_to_log]
log_iqr_cleaned = remove_outliers_iqr(log_iqr_df, log_cols)

# 결과 확인
print(f"원본 행 수: {len(exploded_df)}")
print(f"제거된 이상치 수: {len(exploded_df) - len(log_iqr_cleaned)}")
print(f"제거 후 행 수: {len(log_iqr_cleaned)}")

log_cols = ['log_sales_qty', 'log_sales_amount', 'log_cost']

for col in log_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(log_iqr_cleaned[col], bins=50, kde=True)
    plt.title(f'Log-Transformed Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

log_iqr_cleaned.drop(columns=['cuisine','log_sales_amount','log_cost'], inplace=True)
log_iqr_cleaned.drop(columns=['log_sales_qty'], inplace=True)

# B안 파생변수 생성

# 1. r_id별로 제공하는 cuisine 리스트 생성
cuisine_cat_r = log_iqr_cleaned.groupby('r_id', observed=False)['region_categorized'] \
    .apply(lambda x: list(set(x))).reset_index()

# 2. One-hot encoding
mlb = MultiLabelBinarizer()
cuisine_encoded_cat_r = pd.DataFrame(
    mlb.fit_transform(cuisine_cat_r['region_categorized']),
    columns=mlb.classes_,
    index=cuisine_cat_r['r_id']
).reset_index()

# 사용할 전체 cuisine 컬럼
cuisine_cols = [
    'asian', 'beverages', 'dessert', 'fast food', 'healthy', 'indian',
    'meat/grill', 'middle eastern', 'north indian', 'seafood',
    'south indian', 'western'
]

# 레스토랑이 제공하는 요리 개수
cuisine_encoded_cat_r['cuisine_variety'] = cuisine_encoded_cat_r[cuisine_cols].sum(axis=1)

indian_cuisines = ['indian', 'north indian', 'south indian']

# 해당 레스토랑의 전통 인도 요리 집중도
cuisine_encoded_cat_r['indian_focus'] = (
    cuisine_encoded_cat_r[indian_cuisines].sum(axis=1) /
    cuisine_encoded_cat_r['cuisine_variety'].replace(0, np.nan)
)

global_cuisines = ['asian', 'western', 'middle eastern']

cuisine_encoded_cat_r['global_ratio'] = (
    cuisine_encoded_cat_r[global_cuisines].sum(axis=1) /
    cuisine_encoded_cat_r['cuisine_variety'].replace(0, np.nan)
)

cuisine_encoded_cat_r['single_cuisine'] = (cuisine_encoded_cat_r['cuisine_variety'] == 1).astype(int)

sweet_cuisines = ['dessert', 'beverages']
cuisine_encoded_cat_r['sweet_ratio'] = (
    cuisine_encoded_cat_r[sweet_cuisines].sum(axis=1) /
    cuisine_encoded_cat_r['cuisine_variety'].replace(0, np.nan)
)

snack_cuisines = ['fast food']
cuisine_encoded_cat_r['snack_focus'] = (
    cuisine_encoded_cat_r[snack_cuisines].sum(axis=1) /
    cuisine_encoded_cat_r['cuisine_variety'].replace(0, np.nan)
)

# 해당 레스토랑이 제공하는 모든 요리 리스트로 정리
cuisine_encoded_cat_r['main_cuisines'] = cuisine_encoded_cat_r[cuisine_cols].apply(
    lambda row: [col for col in cuisine_cols if row[col] == 1], axis=1
)

r_cuisine = cuisine_encoded_cat_r

r_orders = log_iqr_cleaned.copy()
r_orders.drop(columns=['cuisine_list', 'region_categorized'], inplace=True, errors='ignore')
r_orders = r_orders.drop_duplicates(subset=['r_id', 'user_id', 'order_date']).reset_index(drop=True)
# 파생변수 생성
r_orders['order_month'] = r_orders['order_date'].dt.month
r_orders['order_day'] = r_orders['order_date'].dt.dayofweek
r_orders['is_weekend'] = r_orders['order_day'].apply(lambda x: 1 if x in [5,6] else 0)
r_orders['is_unrated'] = r_orders['rating'].apply(lambda x: 1 if x == 'Unrated' else 0)

r_orders['cost'] = pd.to_numeric(r_orders['cost'], errors='coerce')
r_orders['cost_level'] = pd.qcut(r_orders['cost'], q=3, labels=['low', 'mid', 'high'])

r_orders['avg_price_per_item'] = (r_orders['sales_amount'] / r_orders['sales_qty']).round(2)
r_orders['cost_diff'] = (r_orders['avg_price_per_item'] - r_orders['cost']).round(2)

# 소비 성향 관련
def classify_spending(diff):
    if diff < -100:
        return 'under_spender'
    elif -100 <= diff <= 100:
        return 'expected_level'
    elif 100 < diff <= 500:
        return 'premium_spender'
    else:
        return 'vip_spender'
r_orders['spending_level'] = r_orders['cost_diff'].apply(classify_spending)

# 평점 구간화
def classify_rating(r):
    if r == 'Unrated':
        return 'unrated'
    try:
        r = float(r)
        if r < 3.5:
            return 'low'
        elif r < 4.2:
            return 'mid'
        else:
            return 'high'
    except:
        return 'unknown'
r_orders['rating_level'] = r_orders['rating'].apply(classify_rating)

# 리뷰 수 기반 레스토랑 규모
def scale_categorize(val):
    if val == 'Too Few Ratings':
        return 'tiny'
    elif val in ['20+ ratings', '50+ ratings']:
        return 'small'
    elif val in ['100+ ratings', '500+ ratings']:
        return 'medium'
    elif val in ['1K+ ratings', '5K+ ratings']:
        return 'large'
    elif val == '10K+ ratings':
        return 'huge'
    else:
        return 'unknown'
r_orders['restaurant_review_scale'] = r_orders['rating_count'].apply(scale_categorize)

## user_id 기준 region_categorized 집계
# 각 사용자별 고유 cuisine을 수집
region_by_user = log_iqr_cleaned.groupby('user_id')['region_categorized'].apply(lambda x: list(set(x))).reset_index()

# MultiLabelBinarizer를 적용
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
cuisine_cat_encoded = pd.DataFrame(mlb.fit_transform(region_by_user['region_categorized']),
                               columns=mlb.classes_,
                               index=region_by_user['user_id'])

# user_id를 인덱스(=컬럼)로 복구
cuisine_cat_encoded.reset_index(inplace=True)

# region_by_user + cuisine_cat_encoded 병합
region_by_user = region_by_user.merge(cuisine_cat_encoded, on='user_id', how='left')

# prefers_ 컬럼 리스트
prefers_cols = [col for col in region_by_user.columns if col.startswith('prefers_')]

#### C안 파생변수 생성
##1차 추가
top_regions = region_by_user[prefers_cols].sum().sort_values(ascending=False).head(5).index.tolist()
region_by_user['popular_region_count'] = region_by_user[top_regions].sum(axis=1)
region_by_user['preference_count'] = region_by_user['region_categorized'].apply(lambda x: len(x) if isinstance(x, list) else 0)
region_by_user['has_multiple_preferences'] = region_by_user['preference_count'].apply(lambda x: 1 if x >= 2 else 0)
region_by_user['is_unknown_preference'] = region_by_user['region_categorized'].apply(lambda x: 1 if x == ['Unknown'] else 0)

def classify_preference_type(row):
    if row['is_unknown_preference'] == 1:
        return 'Unknown'
    elif row['preference_count'] == 1:
        return 'Single_Preference'
    elif row['preference_count'] <= 3:
        return 'Moderate_Diversity'
    else:
        return 'High_Diversity'
region_by_user['preference_type'] = region_by_user.apply(classify_preference_type, axis=1)

regional_groups = {
    'north_indian_group': ['prefers_north_indian', 'prefers_punjabi', 'prefers_kashmiri'],
    'south_indian_group': ['prefers_south_indian', 'prefers_kerala', 'prefers_tamil_nadu'],
    'chinese_group': ['prefers_chinese', 'prefers_sichuan', 'prefers_cantonese'],
    'international_group': ['prefers_italian', 'prefers_mexican', 'prefers_thai', 'prefers_continental']
}
for group_name, cols in regional_groups.items():
    valid_cols = [col for col in cols if col in region_by_user.columns]
    region_by_user[f'prefers_{group_name}'] = region_by_user[valid_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

##2차 추가
region_by_user['preference_density'] = (
    region_by_user['popular_region_count'] / region_by_user['preference_count']
).replace([np.inf, np.nan], 0).round(2)
region_by_user['rare_preference_count'] = (
    region_by_user['preference_count'] - region_by_user['popular_region_count']
).clip(lower=0)
region_by_user['is_extreme_preference'] = region_by_user['preference_count'].apply(lambda x: 1 if x == 1 or x >= 5 else 0)

##3차 추가
region_by_user['is_single_region_user'] = region_by_user['preference_count'].apply(lambda x: 1 if x == 1 else 0)
region_by_user['is_only_popular_region_user'] = (
    (region_by_user['popular_region_count'] == region_by_user['preference_count']) &
    (region_by_user['preference_count'] > 0)
).astype(int)
region_by_user['is_only_rare_region_user'] = (
    (region_by_user['popular_region_count'] == 0) &
    (region_by_user['preference_count'] > 0)
).astype(int)

# r_orders + region_by_user 병합(user_id 기준)
r_orders = pd.merge(
    r_orders,
    region_by_user,
    on='user_id',
    how='left'
)

r_orders_final = r_orders.merge(r_cuisine, on='r_id', how='left')

# 고객 기반 컬럼 이름 바꾸기
r_orders_final.rename(columns=lambda x: x.replace('_x', '_by_user') if x.endswith('_x') else x, inplace=True)

# 레스토랑 기반 컬럼 이름 바꾸기
r_orders_final.rename(columns=lambda x: x.replace('_y', '_by_restaurant') if x.endswith('_y') else x, inplace=True)

# 가성비
r_orders_final['numeric_rating'] = pd.to_numeric(r_orders_final['rating'], errors='coerce')
r_orders_final['cost_per_rating'] = (r_orders_final['cost'] / r_orders_final['numeric_rating']).round(2)

#가성비 여부
def label_cost_per_rating(x):
    if pd.isna(x):
        return 'Unrated'  # No rating, cannot calculate
    elif x <= 50:
        return 'Good Value' #가성비 좋음: 낮은 가격 대비 높은 평점
    elif x <= 65:
        return 'Average' #평균
    else:
        return 'Poor Value' #가성비 나쁨: 높은 가격 대비 낮은 평점

r_orders_final['cost_per_rating_level'] = r_orders_final['cost_per_rating'].apply(label_cost_per_rating)


def make_final_merged_with_ltv_churn(orders_df, users_df, r_orders_final_df):
    # 날짜 변환
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

    # 2회 이상 주문 유저 필터링
    order_counts = orders_df['user_id'].value_counts()
    users_2plus = order_counts[order_counts >= 2].index
    orders = orders_df[orders_df['user_id'].isin(users_2plus)].copy()

    # 기준 날짜 계산
    max_order_date = orders_df['order_date'].max()
    reference_date = max_order_date - pd.Timedelta(days=100)

    # 판매금액 > 0만 추출
    orders_positive = orders[orders['sales_amount'] > 0].copy()

    # 기본 집계
    main_agg = orders.groupby('user_id').agg(
        total_order_count=('order_date', 'count'),
        first_order_date=('order_date', 'min'),
        last_order_date=('order_date', 'max'),
        avg_quantity_per_order=('sales_qty', 'mean'),
        total_quantity=('sales_qty', 'sum'),
        active_days=('order_date', lambda x: (x.max() - x.min()).days),
    )

    # 판매 금액 집계
    sales_agg = orders_positive.groupby('user_id').agg(
        total_sales_amount=('sales_amount', 'sum'),
        avg_order_amount=('sales_amount', 'mean')
    )

    ltv_df = main_agg.join(sales_agg, how='left').reset_index()
    #  여기에 추가
    orders['order_day'] = orders['order_date'].dt.day_name()
    most_active_weekday = (
        orders.groupby(['user_id', 'order_day'])
        .size()
        .reset_index(name='count')
        .sort_values(['user_id', 'count'], ascending=[True, False])
        .drop_duplicates('user_id')[['user_id', 'order_day']]
        .rename(columns={'order_day': 'most_active_weekday'})
    )
    most_visited_restaurant = (
        orders.groupby(['user_id', 'r_id'])
        .size()
        .reset_index(name='count')
        .sort_values(['user_id', 'count'], ascending=[True, False])
        .drop_duplicates('user_id')[['user_id', 'r_id']]
        .rename(columns={'r_id': 'most_visited_restaurant'})
    )
    ltv_df = ltv_df.merge(most_visited_restaurant, on='user_id', how='left')
    ltv_df = ltv_df.merge(most_active_weekday, on='user_id', how='left')
    # 파생 변수 생성

    ltv_df["days_since_last_order"] = (reference_date - ltv_df["last_order_date"]).dt.days
    ltv_df["recency_ratio"] = ltv_df["days_since_last_order"] / (ltv_df["active_days"] + 1)
    ltv_df["revisit_count"] = ltv_df["total_order_count"] - 1
    ltv_df["order_frequency"] = ltv_df["total_order_count"] / (ltv_df["active_days"] + 1)
    ltv_df["monetary_per_day"] = ltv_df["total_sales_amount"] / (ltv_df["active_days"] + 1)
    ltv_df["avg_days_between_orders"] = ltv_df["active_days"] / (ltv_df["revisit_count"].clip(lower=1))
    ltv_df["duration"] = (reference_date - ltv_df["last_order_date"]).dt.days
    ltv_df["churn"] = (ltv_df["duration"] > 100).astype(int)

    # 유저 정보 병합
    df_ltv = users_df.merge(ltv_df, on="user_id", how="inner")

    # 최종 병합: ltv_df 중심으로 r_orders_final_df 붙이기
    final_merged_df = pd.merge( r_orders_final_df,df_ltv, on="user_id", how="left")

    return final_merged_df

delta_df = make_final_merged_with_ltv_churn(orders_df, users_df, r_orders_final)
# order_date를 datetime으로 변환
delta_df['order_date'] = pd.to_datetime(delta_df['order_date'], errors='coerce')

# 파생변수 생성: user_id 기준 집계
ltv_features = delta_df.groupby('user_id').agg(
    total_sales_amount=('sales_amount', lambda x: x[x > 0].sum()),
    total_order_count=('order_date', 'count'),
    avg_order_amount=('sales_amount', lambda x: x[x > 0].mean()),
    first_order_date=('order_date', 'min'),
    last_order_date=('order_date', 'max'),
    avg_quantity_per_order=('sales_qty', 'mean'),
    total_quantity=('sales_qty', 'sum'),
    active_days=('order_date', lambda x: (x.max() - x.min()).days),
    revisit_count=('order_date', lambda x: x.nunique() - 1)
).reset_index()

# 마지막 주문일 기준으로 days_since_last_order 계산
latest_date = ltv_features['last_order_date'].max()
ltv_features['days_since_last_order'] = (latest_date - ltv_features['last_order_date']).dt.days

# 추가 파생변수 생성
ltv_features['order_frequency'] = ltv_features['total_order_count'] / (ltv_features['active_days'] + 1)
ltv_features['monetary_per_day'] = ltv_features['total_sales_amount'] / (ltv_features['active_days'] + 1)
ltv_features['recency_ratio'] = ltv_features['days_since_last_order'] / (ltv_features['active_days'] + 1)
ltv_features['revisit_rate'] = ltv_features['revisit_count'] / (ltv_features['total_order_count'] + 1)
ltv_features['avg_days_between_orders'] = ltv_features['active_days'] / (ltv_features['total_order_count'] + 1)

# 1. user_id별로 Monthly Income 유일하게 추출
income_per_user = delta_df[['user_id', 'Monthly Income']].drop_duplicates(subset='user_id')

# 2. ltv_features에 병합 (left join)
ltv_features = ltv_features.merge(income_per_user, on='user_id', how='left')
# user_id 기준으로 성별 유일값 추출
gender_per_user = delta_df[['user_id', 'Gender']].drop_duplicates(subset='user_id')
# user_id 기준으로 직업 유일값 추출
occupation_per_user = delta_df[['user_id', 'Occupation']].drop_duplicates(subset='user_id')

# 병합
ltv_features = ltv_features.merge(occupation_per_user, on='user_id', how='left')

# ltv_features에 병합
ltv_features = ltv_features.merge(gender_per_user, on='user_id', how='left')

user_info = delta_df[['user_id', 'Marital Status', 'Educational Qualifications', 'most_active_weekday']].drop_duplicates(subset='user_id')

# ltv_features에 병합
ltv_features = ltv_features.merge(user_info, on='user_id', how='left')

# 병합할 컬럼들만 user_id 기준으로 고유하게 추출
most_restaurant = delta_df[['user_id', 'most_visited_restaurant']].drop_duplicates(subset='user_id')

# ltv_features에 병합
ltv_features = ltv_features.merge(most_restaurant, on='user_id', how='left')

# 병합할 컬럼들만 user_id 기준으로 고유하게 추출
family = delta_df[['user_id','Family size']].drop_duplicates(subset='user_id')

# ltv_features에 병합
ltv_features = ltv_features.merge(family, on='user_id', how='left')

# user_id 기준으로 성별 유일값 추출
age = delta_df[['user_id', 'Age']].drop_duplicates(subset='user_id')

# ltv_features에 병합
final_df = ltv_features.merge(age, on='user_id', how='left')

# 리스트 또는 set, dict 같은 비해시형 객체를 포함한 컬럼 탐지
non_hashable_cols = []

for col in delta_df.columns:
    if delta_df[col].apply(lambda x: isinstance(x, (list, dict, set))).any():
        non_hashable_cols.append(col)

print("🔍 리스트/딕셔너리/셋 타입을 포함한 컬럼들:")
print(non_hashable_cols)

# 번호와 함께 컬럼명 나열
for i, col in enumerate(final_df.columns):
    print(f"{i}. {col}")
# 전체 범주형(object) 변수 목록 확인
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype
categorical_columns = [
    col for col in delta_df.columns
    if (
        delta_df[col].dtype == 'object' or
        is_categorical_dtype(delta_df[col]) or
        is_datetime64_any_dtype(delta_df[col])
    )
    and not delta_df[col].apply(lambda x: isinstance(x, (list, dict, set))).any()
]
# 유니크 값 개수도 함께 확인
summary_df = pd.DataFrame({
    'Column': categorical_columns,
    'Unique_Values': [delta_df[col].nunique() for col in categorical_columns]
}).sort_values('Unique_Values', ascending=False)
print(summary_df)
list_type_cols = [
    col for col in categorical_columns
    if delta_df[col].apply(lambda x: isinstance(x, list)).any()
]

# 각 리스트형 컬럼에서 유니크 요소 개수 확인
for col in list_type_cols:
    flat_values = delta_df[col].dropna().explode()
    print(f"{col}: {flat_values.nunique()} unique values in list items")
final_df.to_csv("final_df_alpha.csv")
df=final_df.copy()
df_alpha = df
df_orgin = df.copy()
#####클러스터#####
numerical_features_cl = [
    'total_sales_amount','total_order_count','avg_order_amount',
    'avg_quantity_per_order','total_quantity','active_days',
    'order_frequency','monetary_per_day','days_since_last_order',
    'avg_days_between_orders','recency_ratio'
]

# 3. VIF 계산 함수
def calculate_vif_cl(df, features):
    df = df[features].dropna()
    X = df.copy()
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

# 4. 전처리
df = df.dropna(subset=numerical_features_cl)
scaler = RobustScaler()
df[numerical_features_cl] = scaler.fit_transform(df[numerical_features_cl])
X_scaled = df[numerical_features_cl].values

# 5. Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. KMeans 클러스터링
optimal_k = 3  # 적절히 수정 가능
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['user_segment'] = kmeans.fit_predict(X_scaled)

# 7. UMAP 2D 시각화
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=100,      # 기본 15 → 줄이면 국소 구조 강조
    min_dist=0.05,       # 기본 0.1 → 줄이면 군집 더 조밀하게 표현
    random_state=42
)
umap_result = umap_model.fit_transform(X_scaled)
df['umap1'], df['umap2'] = umap_result[:, 0], umap_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='umap1', y='umap2', hue='user_segment', palette='Set2', s=10)
plt.title('KMeans Clustering Result (UMAP Projection)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# 8. UMAP 3D 시각화
umap_3d = umap.UMAP(n_components=3, random_state=42)
umap_3d_result = umap_3d.fit_transform(X_scaled)
df['umap1'], df['umap2'], df['umap3'] = umap_3d_result[:, 0], umap_3d_result[:, 1], umap_3d_result[:, 2]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['umap1'], df['umap2'], df['umap3'],
                     c=df['user_segment'], cmap='Set2', s=10, alpha=0.6)
ax.set_title('KMeans Clustering Result (UMAP 3D Projection)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
plt.legend(*scatter.legend_elements(), title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 9. 클러스터별 수치형 해석
interpret_cols = numerical_features_cl.copy()
global_avg = df[interpret_cols].mean()
summary = df.groupby('user_segment')[interpret_cols].mean()
summary['N'] = df['user_segment'].value_counts().sort_index()

for cid, row in summary.iterrows():
    print(f"\n[클러스터 {cid}] (표본 수: {int(row['N'])})")
    for col in interpret_cols:
        diff = row[col] - global_avg[col]
        direction = "높음 ↑" if diff > 0 else "낮음 ↓" if diff < 0 else "동일 ="
        print(f"  - {col}: {row[col]:.2f} ({direction})")

# 10. VIP 군집 산정
vip_score_cols = [
    'total_sales_amount',
    'revisit_count',
    'order_frequency',
    'total_order_count',
    'avg_days_between_orders'
]
vip_score_raws = df.groupby("user_segment")[vip_score_cols].mean()
vip_score_norm = (vip_score_raws - vip_score_raws.min()) / (vip_score_raws.max() - vip_score_raws.min())
vip_score_norm['avg_days_between_orders'] = 1 - vip_score_norm['avg_days_between_orders']
vip_score_norm['vip_score'] = (
    0.35 * vip_score_norm['total_sales_amount'] +
    0.25 * vip_score_norm['revisit_count'] +
    0.15 * vip_score_norm['order_frequency'] +
    0.15 * vip_score_norm['total_order_count'] +
    0.10 * vip_score_norm['avg_days_between_orders']
)
vip_segment_id = vip_score_norm['vip_score'].idxmax()
print(f"\n⭐ 최종 VIP 세그먼트는: {vip_segment_id}번 군집")

# 클러스터링 평가 지표 계산
sil_score = silhouette_score(X_scaled, df['user_segment'])
db_score = davies_bouldin_score(X_scaled, df['user_segment'])

print(f"\n✅ Silhouette Score: {sil_score:.4f}")
print(f"✅ Davies-Bouldin Score: {db_score:.4f}")

# 1. 데이터 불러오기 및 결측 제거

df_alpha = df_alpha.dropna(subset=numerical_features_cl)

# 2. 정규화 (스케일링)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_alpha[numerical_features_cl]), columns=numerical_features_cl)

# 3. VIF 계산 함수
def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data.sort_values(by="VIF", ascending=False)

# 4. VIF 계산 실행
vif_result = calculate_vif(df_scaled, numerical_features_cl)
print(vif_result)

#입력 피처 정의
numerical_features = [
    'total_sales_amount', 'avg_order_amount',
    'avg_quantity_per_order', 'total_quantity','active_days',
     'order_frequency','monetary_per_day',
    'avg_days_between_orders'
 ] #수치형
log_transform_cols = [
    'total_sales_amount', 'avg_order_amount',
    'avg_quantity_per_order', 'total_quantity',
    'avg_days_between_orders'
]

# 범주형 피처
categorical_features = [
    'Gender', 'Marital Status', 'Occupation',
    'Monthly Income', 'Educational Qualifications',
    'Family size','Age'
]

# 최종 인코딩된 범주형 피처 리스트
encoded_categorical_features_cl = []

target_col = 'churn' #이탈율 예측으로 인한 이진분류

df_base = df_orgin.copy()

# 2. target 결측 제거
df_base = df_base.dropna(subset=numerical_features + [target_col])
# 3. 범주형 → OneHot 인코딩
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(df_base[categorical_features])
encoded_categorical_features = list(ohe.get_feature_names_out(categorical_features))

# 4. OHE 결과 결합
df_ohe = pd.DataFrame(X_ohe, columns=encoded_categorical_features, index=df_base.index)
df_base = pd.concat([df_base.drop(columns=categorical_features), df_ohe], axis=1)

mlp_features = numerical_features
xgb_features = mlp_features + encoded_categorical_features

missing = [col for col in xgb_features if col not in df_base.columns]
if missing:
    print(f"[경고] 누락된 피처 자동 생성: {missing}")
    for col in missing:
        df_base[col] = 0

df_base['last_order_date'] = pd.to_datetime(df_base['last_order_date'])
df_base=df_base.sort_values("last_order_date").reset_index(drop=True)
cutoff = pd.Timestamp('2019-12-01')
train_users = df_base[df_base['last_order_date'] < cutoff]['user_id']
test_users  = df_base[df_base['last_order_date'] >= cutoff]['user_id']

df_train = df_base[df_base['user_id'].isin(train_users)]
df_test  = df_base[df_base['user_id'].isin(test_users)]
#mlp용
df_mlp_train = df_train.copy()
df_mlp_test = df_test.copy()

# 타깃도 추출
y = df_base[target_col].values

y_train = df_train[target_col].values
y_test = df_test[target_col].values

test_idx = df_test.index

#MLP용 데이터 증강을 위한 CTGAN사용

df_ctgan = df_base[mlp_features + [target_col]].copy()
df_ctgan_log = df_ctgan.copy()

for col in log_transform_cols:
    df_mlp_train[col] = np.log1p(df_mlp_train[col])
    df_mlp_test[col] = np.log1p(df_mlp_test[col])
    df_ctgan_log[col] = np.log1p(df_ctgan_log[col])
df_ctgan_log = df_ctgan_log.dropna()

# 메타 생성 및 GAN 초기화
print("[1/5] Detecting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_ctgan_log)
for col in df_ctgan_log.columns:
    if col != target_col:
        metadata.update_column(column_name=col, sdtype="numerical")
metadata.update_column(column_name=target_col, sdtype="categorical")
print("[2/5] Initializing CTGAN synthesizer...")
ctgan = CTGANSynthesizer(metadata, epochs=50)
#샘플링
print("[3/5] Fitting CTGAN...")
start = time.time()
ctgan.fit(df_ctgan_log)
print(f"CTGAN 학습 완료 (소요 시간: {time.time() - start:.2f}초)")
print("    → CTGAN 학습 완료.")
print("[4/5] Generating synthetic samples...")
df_synth_0 = ctgan.sample(num_rows=int(len(df_ctgan_log) * 3.0))
print(f"    → Synthetic rows: {len(df_synth_0)}")
#증강 결합
df_gan_augmented = pd.concat([df_ctgan_log, df_synth_0], ignore_index=True).dropna()
print(f"[5/5] 증강 완료. 최종 샘플 수: {len(df_gan_augmented)}")
# 클래스 비율 확인
print("\n[✓] 클래스 분포 (y):")
print(pd.Series(df_gan_augmented[target_col]).value_counts(normalize=True).rename("비율"))
print(df_ctgan_log['churn'].value_counts(normalize=True))
print(df_synth_0['churn'].value_counts(normalize=True))
#  X, y 분리
scaler_mlp = StandardScaler()
mlp_features = [col for col in mlp_features if col in df_mlp_train.columns]


X_mlp_train = scaler_mlp.fit_transform(df_mlp_train[mlp_features])
X_mlp_test = scaler_mlp.transform(df_mlp_test[mlp_features])
X_mlp_gan = scaler_mlp.fit_transform(df_gan_augmented[mlp_features])
y_mlp_gan = df_gan_augmented[target_col].values

scaler_xgb = StandardScaler()
X_xgb_train = scaler_xgb.fit_transform(df_train[xgb_features])
X_xgb_test = scaler_xgb.transform(df_test[xgb_features])

X_mlp_train_tensor = torch.tensor(X_mlp_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_mlp_test_tensor = torch.tensor(X_mlp_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

X_mlp = np.vstack([X_mlp_train, X_mlp_test])
input_dim = X_mlp.shape[1]
##-------------------------------------------디버그 확인용------------------------------------------------------

vif_result = calculate_vif(df_base, numerical_features) #다중공정성 검사
print(vif_result)

print("input_dim:", input_dim)
print(torch.isnan(X_mlp_train_tensor).any())  # True면 문제
print(torch.isnan(y_train_tensor).any())     # y도 확인

test_df = df_test.copy()
X_test_df = df_test[numerical_features + encoded_categorical_features]
print("Test label distribution:", np.unique(y_test, return_counts=True))
#초기 xgboost
xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_base.fit(X_xgb_train, y_train)

##---------------------------------------------튜닝 전 평가--------------------------------------------------------
#XGB
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("y_test 분포:\n", pd.Series(y_test).value_counts())
    print("y_pred 분포:\n", pd.Series(y_pred).value_counts())

    # KeyError 방지용
    f1_class1 = report["1"]["f1-score"] if "1" in report else 0.0

    return {
        "Model": name,
        "F1 (class 1)": round(f1_class1, 4),
        "Macro F1": round(report["macro avg"]["f1-score"], 4),
        "Weighted F1": round(report["weighted avg"]["f1-score"], 4)
    }

baseline_results = []
for name, model in {
    "XGBoost (base)": xgb_base
}.items():
    baseline_results.append(evaluate_model(name, model, X_xgb_test, y_test))

baseline_df = pd.DataFrame(baseline_results)
print(" Optuna 실행 전 기본 모델 성능 비교")
print(baseline_df)
#-------------------------------------------------------------------------------------------------------------------
##----------------------------------------------튜닝전 MLP 정의 ------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# 학습 준비
model = MLP(input_dim=X_mlp.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 학습
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_mlp_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 평가
model.eval()
with torch.no_grad():
    y_pred = model(X_mlp_test_tensor)
    y_pred_cls = (y_pred > 0.5).int()

print(classification_report(y_test_tensor.numpy(), y_pred_cls.numpy()))
##-----------------------------------------이제 튜닝을 새로 트레인닝 설정-----------------------------------------------
# ------------------------XGBoost 최적 파라미터 설정을 위한 트레이닝 재분류 ------------------------------------------
X_mlp_train_val, X_mlp_val, y_mlp_train_val, y_mlp_val = train_test_split(
    X_mlp_gan, y_mlp_gan, stratify=y_mlp_gan, test_size=0.2, random_state=42
)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_xgb_train, y_train, stratify=y_train, test_size=0.2, random_state=42
)

neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
scale_ratio = neg / pos

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10)
    }

    model = XGBClassifier(**params,scale_pos_weight=scale_ratio)

    model.fit(
        X_train_val, y_train_val,
    )

    preds = model.predict(X_test_val)
    return f1_score(y_test_val, preds)

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)

print("Best XGBoost params:", xgb_study.best_params)

##------------------------- Optuna 튜닝 대상 MLP objective 함수--------------------------------------
#  튜닝용은 train 데이터에서만 분리

X_train_val_tensor = torch.tensor(X_mlp_train_val, dtype=torch.float32)
y_train_val_tensor = torch.tensor(y_mlp_train_val, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_mlp_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_mlp_val, dtype=torch.float32).unsqueeze(1)

#  pos_weight 계산
neg = np.sum(y_train_val == 0)
pos = np.sum(y_train_val == 1)
pos_weight = torch.tensor([neg / pos])


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def objective(trial):
    hidden1 = trial.suggest_int("hidden1", 32, 128)
    hidden2 = trial.suggest_int("hidden2", 16, 64)
    hidden3 = trial.suggest_int("hidden3", 8, 24)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("epochs", 20, 120)  #
    threshold = trial.suggest_float("threshold", 0.3, 0.7)  #
#----------------------------------------------MLP 튜닝-----------------------------------------------
    class TunedMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, hidden3),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden3, 1),
            )
        def forward(self, x):
            return self.net(x)

    model = TunedMLP(X_mlp.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_val_tensor)
        loss = criterion(y_pred, y_train_val_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        probs = torch.sigmoid(y_val_pred)
        y_val_pred_cls = (probs > threshold).int()
        return f1_score(y_val_tensor.numpy(), y_val_pred_cls.numpy(), average='macro')

# Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)

# best_params를 적용하여 최종 MLP 모델 재학습
best_params = study.best_params
best_threshold = study.best_params['threshold']
best_epochs = study.best_params['epochs']
#------------------------------------------최종 MLP 실행-------------------------------------------------
class FinalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, best_params['hidden1']),
            nn.ReLU(),
            nn.Dropout(best_params['dropout']),
            nn.Linear(best_params['hidden1'], best_params['hidden2']),
            nn.ReLU(),
            nn.Dropout(best_params['dropout']),
            nn.Linear(best_params['hidden2'], best_params['hidden3']),
            nn.ReLU(),
            nn.Dropout(best_params['dropout']),
            nn.Linear(best_params['hidden3'], 1),
        )
    def forward(self, x):
        return self.model(x)

# 데이터 텐서 다시 설정
X_train_tensor = torch.tensor(X_mlp_gan, dtype=torch.float32)
y_train_tensor = torch.tensor(y_mlp_gan, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_mlp_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 모델 학습
final_model = FinalMLP(X_mlp.shape[1])
optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
criterion = nn.BCEWithLogitsLoss()


for epoch in range(best_params['epochs']):
    final_model.train()
    optimizer.zero_grad()
    y_pred_train = final_model(X_train_tensor)
    loss = criterion(y_pred_train, y_train_tensor)
    loss.backward()
    optimizer.step()

# 최종 평가
final_model.eval()
with torch.no_grad():
    logits = final_model(X_test_tensor)
    probs = torch.sigmoid(logits)
    y_pred_cls = (probs > best_params['threshold']).int()

#------------------------------앙상블을 위한 MLP래핑, 파이토치 호환성 때문---------------------

class MLPWrapper(BaseEstimator):
    def __init__(self, model, epochs=50, lr=0.001, threshold=0.5, pos_weight=1.0):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.pos_weight = pos_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy().flatten()
        return probs

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy().flatten()
        #  threshold 반영
        return (probs > self.threshold).astype(int)

best_xgb = XGBClassifier(
    **xgb_study.best_params,
    eval_metric='logloss'
)
best_xgb.fit(X_xgb_train, y_train)

xgb_pred = best_xgb.predict(X_xgb_test)

# XGBoost는 그대로 사용
xgb_model = best_xgb

# MLPWrapper는 X_mlp에 맞는 버전으로 따로 구성
mlp_input = X_mlp_train.copy()  # 수치형만
#-----------------------------------래핑적용------------------------------------------
class MLPAdaptor(BaseEstimator):
    def __init__(self, model_wrapper, feature_idx):
        self.model_wrapper = model_wrapper
        self.feature_idx = feature_idx
        self.classes_ = np.array([0, 1])  # 이진 분류 기준

    def fit(self, X, y):
        self.model_wrapper.fit(X[:, self.feature_idx], y)
        return self

    def predict(self, X):
        return self.model_wrapper.predict(X[:, self.feature_idx])

    def predict_proba(self, X):
        probs = self.model_wrapper.predict_proba(X[:, self.feature_idx])
        return np.vstack([1 - probs, probs]).T
mlp_wrapper = MLPWrapper(
    final_model,
    epochs=best_params['epochs'],
    lr=best_params['lr'],
    threshold=best_params['threshold']
     )
mlp_feature_idx = [i for i, col in enumerate(xgb_features) if col in mlp_features]

mlp_adaptor = MLPAdaptor(mlp_wrapper, feature_idx=mlp_feature_idx)
##----------------------스태킹-----------------------------------
# 스태킹 정의
stacking_clf =  StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('mlp', mlp_adaptor)
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    passthrough=False,
    cv=5,
    n_jobs=-1
)

stacking_clf.fit(X_xgb_train, y_train)
#---------------------------------------------------------------------------------------
# 학습 및 평가
stack_pred = stacking_clf.predict(X_xgb_test)
mlp_probs = torch.sigmoid(final_model(X_mlp_test_tensor)).detach().numpy().flatten()
mlp_preds = (mlp_probs > best_params['threshold']).astype(int)
xgb_preds = best_xgb.predict(X_xgb_test)



print("MLP Report:")
print(classification_report(y_test, mlp_preds))
print("\nXGBoost Report:")
print(classification_report(y_test, xgb_preds))
print("\nStacking Report:")
print(classification_report(y_test, stack_pred))
##-------------------------비교---------------------------------------------
# 스태킹 모델 확률 예측
stacking_probs = stacking_clf.predict_proba(X_xgb_test)[:, 1]


# XGBoost 예측
xgb_preds = best_xgb.predict(X_xgb_test)

# Stacking 예측
stack_preds = stacking_clf.predict(X_xgb_test)

# 평가 함수
def evaluate_model(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

# 결과 수집
results = [
    evaluate_model(y_test, mlp_preds, "MLP"),
    evaluate_model(y_test, xgb_preds, "XGBoost"),
    evaluate_model(y_test, stack_preds, "Stacking"),
]

# 텍스트 테이블 출력
print("\n 성능 비교 (MLP vs XGBoost vs Stacking)\n")
print(tabulate(results, headers="keys", floatfmt=".4f"))


# 예측
train_pred = stacking_clf.predict(X_xgb_train)
test_pred  = stacking_clf.predict(X_xgb_test)

##과적합 확인을 위한 평가 디버그
print("TRAIN")
print(classification_report(y_train, train_pred))
print("TEST")
print(classification_report(y_test, test_pred))

# 위험 등급 함수 (이탈(1)을 기준으로 봤을 때 이탈로 갈 확률)
def assign_risk_level(prob):
    if prob > 0.7:         #수치 조정 필요
        return "상"
    elif prob > 0.4:
        return "중"
    else:
        return "하"

# 위험 등급 데이터프레임 생성
risk_df = pd.DataFrame({
    'predicted_prob': stacking_probs,  # 예측된 확률
    'true_label': y_test,
})

# 위험 등급 지정
risk_df["risk_level"] = risk_df["predicted_prob"].apply(assign_risk_level)

assert len(risk_df) == len(test_df)
df_matched = df_orgin.loc[test_df.index].reset_index(drop=True)
# test_df에는 user_segment 등 원래 feature들이 있음 → 병합
risk_df = pd.concat([test_df.reset_index(drop=True), risk_df.reset_index(drop=True)], axis=1)
merged_df = df_orgin.loc[test_df.index].reset_index(drop=True)
merged_df = pd.concat([
    merged_df,
    risk_df[['predicted_prob', 'true_label', 'risk_level']].reset_index(drop=True)
], axis=1)
assert len(merged_df) == len(risk_df) == len(test_df)
print(merged_df[['user_id', 'predicted_prob', 'risk_level']].head())
print("risk_df:", len(risk_df))
print("test_df:", len(test_df))
print("df_orgin:", len(df_orgin))
print("test_df.index (유니크 여부):", len(test_df.index), "→ 유니크:", test_df.index.is_unique)
print(df_orgin.loc[test_df.index[:5]][['user_id']])
print(risk_df.head()[['user_id']])
merged_scaled_df = pd.concat([
    df.reset_index(drop=True),
    risk_df[['predicted_prob', 'true_label', 'risk_level']].reset_index(drop=True)
], axis=1)


# 저장용 임시.
merged_df.to_csv("C:/Users/malthael/Downloads/고니/최종_risk_포함_고객_데이터.csv", index=False)
# vip_segment별 이탈 위험도 분포 교차표
cross_tab = pd.crosstab(risk_df['user_segment'], risk_df['risk_level'], normalize='index')


# 시각화
plt.figure(figsize=(8, 5))
sns.heatmap(cross_tab, annot=True, cmap="Reds")
plt.title("user 세그먼트 × 이탈 위험도 비율 (Stacking 기준)")
plt.xlabel("이탈 위험도")
plt.ylabel("user 세그먼트")
plt.tight_layout()
plt.show()

# 주요 수치형 피처만 선택
summary_cols = [
    'total_sales_amount', 'avg_order_amount',
    'avg_quantity_per_order', 'total_quantity', 'revisit_count',
    'order_frequency',
    'avg_days_between_orders']

# Risk Level별 평균값 계산
summary_df = risk_df.groupby("risk_level")[summary_cols].mean().T

# 정규화 (0~1)
summary_norm = (summary_df - summary_df.min()) / (summary_df.max() - summary_df.min())
summary_norm = summary_norm.reindex(columns=["하", "중", "상"])  # 순서 맞춤

# 시각화
plt.figure(figsize=(12, 6))
summary_norm.plot(kind="bar", figsize=(12, 6), colormap="Reds", edgecolor="black")
plt.title("Risk Level별 주요 행동 특성 (정규화)")
plt.ylabel("정규화된 평균값 (0~1)")
plt.xlabel("Feature")
plt.xticks(rotation=30)
plt.legend(title="Risk Level")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=risk_df, x="risk_level", hue="true_label", order=["하", "중", "상"], palette="Reds")
plt.title("Risk Level별 실제 이탈/잔존 고객 분포")
plt.xlabel("Risk Level (예측 위험 등급)")
plt.ylabel("고객 수")
plt.legend(title="실제 라벨", labels=["잔존 (0)", "이탈 (1)"])
plt.tight_layout()
plt.show()


# 확률 해석을 이탈 기준으로 변환


plt.figure(figsize=(8, 5))
sns.histplot(risk_df["predicted_prob"], bins=100, kde=True, color="red", alpha=0.6)

# 기준선 추가 (이탈 확률 기준)
plt.axvline(0.4, color='grey', linestyle='--', label='Low → Medium Threshold')
plt.axvline(0.7, color='black', linestyle='--', label='Medium → High Threshold')

plt.title("Stacking 모델의 예측된 이탈 확률 분포")
plt.xlabel("이탈 확률 ")
plt.ylabel("고객 수")
plt.legend()
plt.tight_layout()
plt.show()

# Risk 비율 계산
risk_ratio = pd.crosstab(risk_df["risk_level"], risk_df["true_label"], normalize="index") * 100
risk_ratio = risk_ratio.rename(columns={0: "잔존", 1: "이탈"}).reindex(["하", "중", "상"])

# 시각화
ax = risk_ratio.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Reds", edgecolor="black")

# 비율 텍스트 추가
for i, row in enumerate(risk_ratio.values):
    for j, val in enumerate(row):
        if val > 0:
            ax.text(i, row[:j+1].sum() - val / 2, f"{val:.1f}%", ha="center", va="center", fontsize=10, color="black")

plt.title("Risk Level별 이탈률 구성 (%)")
plt.ylabel("비율 (%)")
plt.xlabel("Risk Level")
plt.legend(title="실제 라벨")
plt.tight_layout()
plt.show()

print(classification_report(y_test, stack_pred, target_names=["잔존", "이탈"]))
cm = confusion_matrix(y_test, stack_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["잔존", "이탈"])
disp.plot(cmap='Reds')
plt.title("Confusion Matrix (Stacking Model)")
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, stacking_probs)
ap = average_precision_score(y_test, stacking_probs)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label=f"Stacking (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Stacking)")
plt.legend()
plt.tight_layout()
plt.show()

# 범주형 컬럼 목록 (분석 대상)
categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'region', 'preference_type']
risk_df = pd.concat([
    df_orgin.iloc[test_idx][categorical_cols].reset_index(drop=True),
    risk_df.reset_index(drop=True)
], axis=1)
for col in categorical_cols:
    plt.figure(figsize=(8, 4))

    # 그룹별 이탈률 계산
    churn_rate = risk_df.groupby(col)['true_label'].mean().sort_values(ascending=False)

    sns.barplot(x=churn_rate.index, y=churn_rate.values, palette="Reds")
    plt.title(f"{col}별 이탈률")
    plt.ylabel("이탈률 (mean of true_label)")
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

vip_df = risk_df[merged_df['user_segment'] == vip_segment_id]
plt.figure(figsize=(6, 4))
sns.countplot(data=vip_df, x="risk_level", palette="Reds", order=["하", "중", "상"])
plt.title("VIP 클러스터의 Risk Level 분포")
plt.xlabel("Risk Level")
plt.ylabel("고객 수")
plt.tight_layout()
plt.show()

categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'region', 'preference_type']

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    churn_rate = vip_df.groupby(col)['true_label'].mean().sort_values(ascending=False)

    sns.barplot(x=churn_rate.index, y=churn_rate.values, palette="Reds")
    plt.title(f"[VIP 클러스터] {col}별 이탈률")
    plt.ylabel("이탈률 (mean of true_label)")
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(vip_df["predicted_prob"], bins=100, kde=True, color="red", alpha=0.6)
plt.axvline(0.4, color='grey', linestyle='--', label='Low → Medium Threshold')
plt.axvline(0.7, color='black', linestyle='--', label='Medium → High Threshold')
plt.title("VIP 클러스터의 이탈 확률 분포")
plt.xlabel("이탈 확률")
plt.ylabel("고객 수")
plt.legend()
plt.tight_layout()
plt.show()

# MLP Permutation Importance
wrapped_model = MLPWrapper(final_model)
result = permutation_importance(
    wrapped_model, X_mlp_test, y_test, n_repeats=10, random_state=42, scoring='f1'
)
importance_df = pd.DataFrame({
    'feature': mlp_features,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values(by='importance_mean', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(data=importance_df, x='importance_mean', y='feature', palette='Reds')
plt.title("Permutation Feature Importance (MLP F1 기반)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# XGBoost Permutation Importance
xgb_importance = permutation_importance(
    best_xgb, X_xgb_test, y_test, n_repeats=10, random_state=42, scoring='f1'
)
xgb_imp_df = pd.DataFrame({
    'feature': xgb_features,
    'importance_mean': xgb_importance.importances_mean,
    'importance_std': xgb_importance.importances_std
}).sort_values(by='importance_mean', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(data=xgb_imp_df, x='importance_mean', y='feature', palette='Reds')
plt.title("Permutation Feature Importance (XGBoost, F1 기반)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# SHAP (XGBoost)
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_xgb_test)
X_test_xgb_df = pd.DataFrame(X_xgb_test, columns=xgb_features)
shap.summary_plot(shap_values, X_test_xgb_df, plot_type='bar')

# CSV 저장(테스트 df으로만 합쳐있음)
risk_df.to_csv("C:/Users/malthael/Downloads/고니/stacking_risk_prediction_with_vip.csv", index=False)
print(" 저장 완료: stacking_risk_prediction_with_vip.csv")




