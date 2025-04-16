## EXNO-3-DS
## NAME : AVINASH T
## REG NO : 212223230026
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```c
#AVINASH T
#212223230026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/cd16be02-db99-4c85-863f-1dfdee486919)
```c
#odinal encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/d855533a-c67b-488b-98f7-3433c72ecfbe)
```c
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/a148d715-27a8-4429-ae9e-2335dc5678f0)
```c
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/efc8c594-4854-461e-bb0a-97d5a5253b8b)
```c
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/b6239160-bd4a-4e8f-8fd6-e2bd22bf6e42)
```c
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/354d62cb-7ea7-4050-9c26-6ce35501f1be)
```c
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/0880c33d-a7ee-4789-86a0-1366d24182c2)
```c
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/f17222db-78a2-48f7-8ce0-8207407d5287)
```c
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/bf6a6f0f-9367-4d13-aaa3-750101498ea6)
```c
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/f1cd7a47-1e9f-495a-a06d-f74c522d7571)
```c
from scipy import stats
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/a21d6a77-ad5c-41b1-acc2-a93647418505)
```c
df.skew()
```
![image](https://github.com/user-attachments/assets/ef5f546b-87b4-4a15-91b8-64c537d31a6b)
```c
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/07c39101-785a-48f5-9c61-1f2c22cb1e5a)
```c
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/dd25d7f1-123c-4b20-b6f3-8fa929f3a600)
```c
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e787c083-75c3-4433-9a35-09159073aa0e)
```c
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7e6c2c76-931f-4964-a08a-9e0d7d579548)

```c
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d2964424-9d8b-453a-b695-140c8c067250)

```c
df.skew()
```
![image](https://github.com/user-attachments/assets/ca4bb340-b013-4a9a-8855-b5e45da07f9b)

```c
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/ec25ee6d-b874-46ef-83cd-50086ea08014)

```c
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/f4e64e55-ad96-42f6-b6a6-69bcc0c8f3de)

```c
import statsmodels.api as sm
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f2d229f5-4506-4ea4-9ac0-ecab8d444107)

```c
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/18f7f4df-4aef-4ee8-bfec-c2ba7c0a7b87)

```c
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/83a98ed8-6428-4b6a-872a-9e0ae80b0a6a)
```c
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/373f54c1-1692-4106-9ca7-d41eeb5502e2)
# RESULT:
       thus the experiment is successfully implemented

       
