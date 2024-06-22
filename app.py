from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

import re
def remove_regExp(text1):
  text1=re.sub('[^a-z A-Z 0-9,]', '', text1)
  return text1

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def remove_stopwords(text):
  new_text=[]
  for word in text.split():
    if word in stopwords.words('english'):
      new_text.append('')
    else:
      new_text.append(word)
  x=new_text[:]
  new_text.clear()
  return " ".join(x)

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem_words(text2):
  return " ".join([ps.stem(word) for word in text2.split()])

vector = pickle.load(open("Vector.pkl", 'rb'))
model = pickle.load(open("finalModel.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predeiction():
    if request.method=='POST':
        news=str(request.form['news'])
        # news=news.lower()
        # news=remove_regExp(news)
        # news=remove_stopwords(news)
        # news=stem_words(news)
        # predict=model.predict(vector.transform(['news']))[0]
        # print(predict)

        testing_news={"text":[news]}
        new_def_test=pd.DataFrame(testing_news)
        new_def_test["text"].str.lower()
        new_def_test["text"]=new_def_test["text"].apply(remove_regExp)
        new_def_test["text"]=new_def_test["text"].apply(remove_stopwords)
        new_def_test["text"]=new_def_test["text"].apply(stem_words)
        new_x_test=new_def_test["text"]
        new_xv_test=vector.transform(new_x_test)
        predict=model.predict(new_xv_test)[0]
        print(predict)

    # return print("Prediction : {}".format(output_lable(pred_final[0])))

        return render_template("index.html", prediction_text="News is {}".format(predict))
if __name__ == '__main__':
    app.run(debug=True)