import pandas as pd

events=pd.read_csv("data_pipeline/processed/cleaned_events.csv")

purchase_events = events[events["event"]=="transaction"]

purchase_events["interaction"]=1

interactions_df=purchase_events[["visitorid","itemid","interaction"]]

interactions_df.to_csv("data_pipeline/processed/interactions.csv",index=False)

print("Prepared interaction data:",interactions_df.shape)