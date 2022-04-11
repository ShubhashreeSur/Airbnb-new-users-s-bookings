
import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import math
from datetime import datetime


st.title(" “Airbnb new users’ bookings”- Where will a new guest book their first travel experience. ")


global X1
input_data=st.file_uploader(label='Enter the file for user entry details (in csv format)',type=['csv'])

if input_data is not None:
    st.write(input_data)
    try:
        X1=pd.read_csv(input_data)
    except Exception as e:
        print()



global X2

session_data=st.file_uploader(label='Enter the file for user session data (in csv format)',type=['csv'])


if session_data is not None:
    st.write(session_data)
    try:
        X2=pd.read_csv(session_data)
    except Exception as e:
        print()




segment=st.selectbox('Show destination country predictions',['yes','no'],index=1)



if segment == 'yes':


    try:
        
        start=time.time()
    
    
        with open('one_hot_encoder','rb') as f:
            ohe=pickle.load(f)
        
        with open('standard_scaler','rb') as f:
            sc=pickle.load(f)
        
        with open('label_encoder','rb') as f:
            le=pickle.load(f) 
        
        trained_models=[]

        with open('ce_meta_model', 'rb') as files:
            meta_model=pickle.load(files)

    
    
        for i in range(1,6):              
            with open('ce_model_'+str(i),'rb') as files:
                model=pickle.load(files)
                trained_models.append(model)  
        
        trained_models.append(model) 
        
        for i,data in X1.iterrows():
            if data['gender'] == '-unknown-':
                a=np.random.uniform()
                if a <= 0.54:
                    data['gender']='FEMALE'
                else:
                    data['gender']='MALE'
            
        for i,data in X1.iterrows():
            if data['age']<18:
                X1.at[i,'age']=18
            if data['age']>95:
                X1.at[i,'age']=95
            if math.isnan(data['age']):
                X1.at[i,'age']=np.random.randint(28,43)


        X1['first_affiliate_tracked'].fillna('untracked',inplace=True)
    
      
        X1.date_account_created=X1.date_account_created.apply(lambda x: datetime.strptime(str(x)[:10], "%d-%m-%Y"))
    
    
    
        X1['month']=X1.date_account_created.apply(lambda x: x.month)
        X1['weekday']=X1.date_account_created.apply(lambda x: x.weekday())
    
        for i,data in X1.iterrows(): 
            if str(int(data['timestamp_first_active']))[5]!=0:
                X1.at[i,'timestamp_first_active']=str(str(int(data['timestamp_first_active']))[:5])+'1'
            else:
                X1.at[i,'timestamp_first_active']=str(str(int(data['timestamp_first_active']))[:6])
            
            
    
        X1['date_first_active']=X1.timestamp_first_active.apply(lambda x: datetime.strptime(str(x)[:6],'%Y%m'))
                                                           
    
        for i,data in X1.iterrows():
            X1.at[i,'waiting_days']=(data['date_account_created']-data['date_first_active']).components.days
        
        
        X1['1st_active_month']=X1.date_first_active.apply(lambda x: x.month)
        X1['1st_active_weekday']=X1.date_first_active.apply(lambda x: x.weekday())
    
        X1.drop('date_first_booking',axis=1,inplace=True)
    
    
    
    
    
    
        X2.action.fillna('NULL',inplace=True)
        X2.action_type.fillna('NULL',inplace=True)
        X2.action_detail.fillna('NULL',inplace=True)
        X2.device_type.fillna('NULL',inplace=True)

      
        X2.secs_elapsed.fillna(0.0,inplace=True)
    
    
        for i,data in X2.iterrows():
            X2.at[i,'action_row']=data['action']+" "+data['action_type']+" "+data['action_detail']
        
    
    
        action_row2=X2[['user_id','action_row']].groupby(['user_id']).max().reset_index()

        device_type2=X2[['user_id','device_type']].groupby(['user_id']).max().reset_index()
    
        count_action=X2[['user_id','action']].groupby(['user_id']).count().reset_index().rename(columns={'action':'count_action'})

        unique_action=X2[['user_id','action']].groupby(['user_id']).nunique().reset_index().rename(columns={'action':'unique_action'})
        unique_action_type=X2[['user_id','action_type']].groupby(['user_id']).nunique().reset_index().rename(columns={'action_type':'unique_action_type'})
        unique_action_detail=X2[['user_id','action_detail']].groupby(['user_id']).nunique().reset_index().rename(columns={'action_detail':'unique_action_detail'})
        unique_device=X2[['user_id','device_type']].groupby(['user_id']).nunique().reset_index().rename(columns={'device_type':'unique_device'})

        secs_elapsed_sum=X2[['user_id','secs_elapsed']].groupby(['user_id']).sum().reset_index().rename(columns={'secs_elapsed':'secs_elapsed_sum'})
        secs_elapsed_mean=X2[['user_id','secs_elapsed']].groupby(['user_id']).mean().reset_index().rename(columns={'secs_elapsed':'secs_elapsed_mean'})
        secs_elapsed_max=X2[['user_id','secs_elapsed']].groupby(['user_id']).max().reset_index().rename(columns={'secs_elapsed':'secs_elapsed_max'})
        secs_elapsed_std=X2[['user_id','secs_elapsed']].groupby(['user_id']).std().reset_index().rename(columns={'secs_elapsed':'secs_elapsed_std'})
        secs_elapsed_median=X2[['user_id','secs_elapsed']].groupby(['user_id']).median().reset_index().rename(columns={'secs_elapsed':'secs_elapsed_median'})
    
    
    
        X2_fea=pd.merge(count_action,unique_action,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,unique_action_type,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,unique_action_detail,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,unique_device,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,secs_elapsed_sum,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,secs_elapsed_mean,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,secs_elapsed_max,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,secs_elapsed_std,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,secs_elapsed_median,how='inner',on='user_id')


        X2_fea=pd.merge(action_row2,X2_fea,how='inner',on='user_id')
        X2_fea=pd.merge(X2_fea,device_type2,how='inner',on='user_id')
    
    
        X2_fea.secs_elapsed_std.fillna(0,inplace=True)
    
    
        X2_fea.rename(columns={'user_id':'id'},inplace=True)
    
        X=pd.merge(X1,X2_fea,how='left',on='id')

        
        categorical_cols=['gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked',\
                 'signup_app','first_device_type','first_browser','action_row','device_type']
    
    
        to_drop=['id','date_account_created','timestamp_first_active','date_first_active']
    
    
        cat_X=X[categorical_cols]
    
        X.drop(categorical_cols,axis=1,inplace=True)
        X.drop(to_drop,axis=1,inplace=True)
    
    
        encoded_X=ohe.transform(cat_X)

    
        X_df=np.hstack((np.array(X),encoded_X))
    
        X_df=sc.transform(X_df)
    
    
    
    
        for i in range(len(trained_models)):
        
            predictions_test=trained_models[i].predict_proba(pd.DataFrame(X_df))
        
            if i==0:
                predictions_test_array=predictions_test
            
            else:
                predictions_test_array=np.hstack((predictions_test_array,predictions_test))
    
    

        predict_test_y=meta_model.predict_proba(predictions_test_array)
    
    
    
        for i in predict_test_y:
            country=np.argsort(i)[-1:-6:-1]
        
        country=le.inverse_transform(country)

        end=time.time()
    
    
        st.write("The top 5 countries that the user would like to visit are:")
        for i in country:
            st.write(i)

    
        st.write("\n\ntime required:",round(end-start,3),"secs")

    except Exception as e:
        st.write()





    

    

