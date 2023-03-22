import pandas as pd

class clean_tabular_data():
    def __init__(self):
        self.df=pd.read_csv('./tabular_data/listing.csv')

    def remove_rows_with_missing_ratings(self):
        self.df=self.df.dropna(subset=["Cleanliness_rating",'Accuracy_rating', 'Communication_rating'	,'Location_rating',	'Check-in_rating' ,	'Value_rating' ])
       
    def combine_description_strings(self):
        self.df = self.df.dropna(subset=['Description'])
        self.df["Description"] = self.df["Description"].apply(lambda x: x.replace("'About this space', ", '').replace("'', ", '').replace('[', '').replace(']', '').replace('\\n', '.      ').replace("''", '').split(" "))
        self.df["Description"] = self.df["Description"].apply(lambda x: " ".join(x))
        
    def set_default_feature_values(self):
        self.df[["guests", "beds", "bathrooms","bedrooms"]]=self.df[["guests", "beds", "bathrooms","bedrooms"]].fillna(value=1)

    def load_airbnb(self,df):
        df = df.drop(columns=["Unnamed: 19"])
        df = df.drop(columns=["Unnamed: 0"])

        features = df.drop(columns=['Price_Night', 'ID', 'Category', 'Title', 'Description', 'Amenities', 'Location', 'url'])
        labels = df["Price_Night"]

        features_labels  = (features, labels)
        return features_labels    
    
if __name__ =='__main__':
    tabular_data=clean_tabular_data()
    tabular_data.remove_rows_with_missing_ratings()
    tabular_data.combine_description_strings()
    tabular_data.set_default_feature_values()
    tabular_data.df.to_csv('clean_data.csv')
    df = pd.read_csv('clean_data.csv')
    feature_labels = tabular_data.load_airbnb(df)
    feature, labels = feature_labels
    print(feature)
    print(labels)