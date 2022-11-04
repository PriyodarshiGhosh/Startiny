import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def prediction(num1, num2, num3,num4,num5,num6, num7, num8, num9, num10, num11, num12, num13, num14, num15, num16, num17, num18, num19, num20, num21,num22,num23,num24,num25,num26):
    d = pd.read_csv("/Users/priyodarshighosh/demo/startup data.csv")
    d['status'] = np.where(d['status'] == 'closed', 0,1)
    d = d.drop(["Unnamed: 0", "Unnamed: 6", "labels","closed_at", "id"], axis = 1)
    d['age_first_milestone_year'] = d['age_first_milestone_year'].fillna(method="ffill")
    d['age_last_milestone_year'] = d['age_last_milestone_year'].fillna(method="ffill")
    Y = d['status']
    X = d[['milestones', 'is_top500', 'has_roundB', 'funding_rounds', 'age_last_milestone_year',
               'avg_participants', 'has_roundA', 'has_roundC', 'has_roundD', 'age_first_milestone_year', 
                'is_enterprise', 'age_last_funding_year', 'is_advertising', 
               'funding_total_usd', 'is_software', 'is_mobile', 'is_consulting', 'is_biotech', 'is_web',
               'is_gamesvideo', 'is_othercategory', 'is_TX', 'has_VC', 'is_ecommerce', 'has_angel', 
               'age_first_funding_year']]
    scale= StandardScaler()
    scaled_data = scale.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, Y, test_size=0.20, random_state=7)
    model = RandomForestClassifier(random_state = 7)
    model.fit(X_train, y_train)
    data = {
  'milestones': num1, 
  'is_top500': num2, 
  'has_roundB': num3, 
  'funding_rounds': num4, 
  'age_last_milestone_year': num5,
  'avg_participants': num6,
  'has_roundA': num7, 
  'has_roundC': num8, 
  'has_roundD': num9, 
  'age_first_milestone_year': num10, 
  'is_enterprise': num11, 
  'age_last_funding_year': num12, 
  'is_advertising': num13, 
  'funding_total_usd': num14, 
  'is_software': num15, 
  'is_mobile': num16, 
  'is_consulting': num17, 
  'is_biotech': num18,  
  'is_web': num19,
  'is_gamesvideo': num20, 
  'is_othercategory': num21, 
  'is_TX': num22, 
  'has_VC': num23, 
  'is_ecommerce': num24, 
  'has_angel': num25, 
  'age_first_funding_year': num26
}
    df = pd.DataFrame([list(data.values())])
    scale= StandardScaler()
    scaled_data = scale.fit_transform(df)
    return model.predict(df)