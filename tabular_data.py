import pandas as pd
from ast import literal_eval
from pandas.api.types import is_numeric_dtype

class clean_tabular_data():
    def __init__(self):
        self.df=pd.read_csv('./tabular_data/listing.csv')

    def remove_rows_with_missing_ratings(self):
        df2=self.df.dropna(subset=["Cleanliness_rating",'Accuracy_rating', 'Communication_rating'	,'Location_rating',	'Check-in_rating' ,	'Value_rating' ])
        print(df2)
        print(type(df2))
    
    # def literal_return(val):
    #         try:
    #             return literal_eval(val)
    #         except (ValueError, SyntaxError) as e:
    #             return val 

    # def combine_description_strings(df):
    #     df['Description']=df['Description'].apply(lambda x: literal_return(x))
    #     df.Description=df.Description.str.join(",")
    #     return df

    def combine_description_strings(self):
        df = self.df.dropna(subset=['Description'])
        df["Description"] = df["Description"].apply(lambda x: x.replace("'About this space', ", '').replace("'', ", '').replace('[', '').replace(']', '').replace('\\n', '.      ').replace("''", '').split(" "))
        df["Description"] = df["Description"].apply(lambda x: " ".join(x))
        return df

    def set_default_feature_values(self, df):
        df[["guests", "beds", "bathrooms","bedrooms"]]=df[["guests", "beds", "bathrooms","bedrooms"]].fillna(value=1)
        return df

    def load_airbnb(self,df,**kwargs):
        for col in df.columns:
            if is_numeric_dtype(df[col])== False:
                df=df.drop(col, axis=1)
        print(df.head())
        for col in df.columns:
            print(col)
        label=df[kwargs['label']]
        features=df[kwargs['features']]
        print('label\n',label)
        print('features\n',features)
        return label,features
        




        # features=df.drop(df[kwargs],axis=1)
        # label={}
        
        # print(features,label)
        # return features,label

        
        # numeric=df._get_numeric_data()
        #
        # features=df.drop(df[label], axis=1)
        
        # print(numeric)
        # print(features,label)
        # return features,label
    
if __name__ =='__main__':
    tabular_data=clean_tabular_data()
    tabular_data.remove_rows_with_missing_ratings()
    df2=tabular_data.combine_description_strings()
    print(df2['Description'])
    df3=tabular_data.set_default_feature_values(df2)
    print(df3)
    df3.to_csv('clean_data.csv')
    
    kwargs={'label':'Price_Night', 'features':['beds', 'bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','Unnamed: 19']}
    tabular_data.load_airbnb(df3,**kwargs)
