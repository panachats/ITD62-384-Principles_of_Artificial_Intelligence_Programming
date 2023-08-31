import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
sample = ['I love the apple', 'the application is too bad', 'The life is so hard']
vec = CountVectorizer()
X = vec.fit_transform(sample)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(df,'\n#########################')

# โอกาศที่คำเหล่านี้จะปรากฏในอนาคต
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(df2)