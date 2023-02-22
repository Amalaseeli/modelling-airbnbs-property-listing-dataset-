import pandas as pd
from ast import literal_eval


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


if __name__ =='__main__':
    tabular_data=clean_tabular_data()
    tabular_data.remove_rows_with_missing_ratings()
    df2=tabular_data.combine_description_strings()
    print(df2['Description'])
    df3=tabular_data.set_default_feature_values(df2)
    print(df3)
    df3.to_csv('clean_data.csv')