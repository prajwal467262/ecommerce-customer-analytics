import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import warnings; warnings.filterwarnings('ignore')

st.set_page_config(page_title="E-Commerce Customer Analytics", page_icon="🛒", layout="wide")

COLORS={'Champions':'#c8f04a','Loyal Customers':'#7c6af7','Promising':'#4af0a0','Needs Attention':'#f0c060','At Risk':'#ff8c42','Lost':'#ff4a4a'}

@st.cache_data
def gen_data():
    np.random.seed(42); custs=1200
    cids=[f'C{i:05d}' for i in range(custs)]
    segs=np.random.choice(['Premium','Regular','Occasional','Churned'],custs,p=[0.15,0.35,0.30,0.20])
    cities=np.random.choice(['Bengaluru','Mumbai','Delhi','Hyderabad','Chennai'],custs,p=[0.30,0.20,0.18,0.17,0.15])
    rows=[]
    for i,cid in enumerate(cids):
        s=segs[i]
        n={'Premium':np.random.randint(15,50),'Regular':np.random.randint(6,20),
           'Occasional':np.random.randint(2,8),'Churned':np.random.randint(1,4)}[s]
        mu={'Premium':2500,'Regular':900,'Occasional':600,'Churned':400}[s]
        for _ in range(n):
            da=np.random.randint(1 if s!='Churned' else 90,365)
            rows.append({'cid':cid,'city':cities[i],
                'date':pd.Timestamp('2025-01-01')-pd.Timedelta(days=da),
                'amount':max(50,np.random.normal(mu,mu*0.3)),
                'cat':np.random.choice(['Electronics','Fashion','Home','Beauty','Sports']),
                'seg':s})
    return pd.DataFrame(rows)

@st.cache_data
def rfm(df):
    snap=pd.Timestamp('2025-01-01')
    r=df.groupby('cid').agg(recency=('date',lambda x:(snap-x.max()).days),
        frequency=('date','count'),monetary=('amount','sum'),
        city=('city','first'),seg=('seg','first')).reset_index()
    for col,asc,lbl in [('recency',False,[1,2,3,4,5]),('frequency',True,[1,2,3,4,5]),('monetary',True,[1,2,3,4,5])]:
        r[col[0].upper()]=pd.qcut(r[col],5,labels=lbl,duplicates='drop').astype(int)
    r['rfm']=r['R']+r['F']+r['M']
    def seg(row):
        if row['rfm']>=13: return 'Champions'
        elif row['R']>=4 and row['F']>=3: return 'Loyal Customers'
        elif row['R']>=4: return 'Promising'
        elif row['R']<=2 and row['rfm']>=9: return 'At Risk'
        elif row['R']<=2: return 'Lost'
        return 'Needs Attention'
    r['segment']=r.apply(seg,axis=1)
    r['clv']=(r['monetary']*(r['frequency']/r['recency'].clip(1))*12).clip(0)
    return r

df=gen_data(); rfm_df=rfm(df)

st.title("🛒 E-Commerce Customer Analytics")
st.caption("RFM Segmentation · CLV Prediction · Retention Strategy | Prajwal Markal Puttaswamy")
st.divider()

tab1,tab2,tab3,tab4=st.tabs(["🎯 RFM Segments","💰 CLV Predictor","📊 Cohort Retention","📋 Strategy"])

with tab1:
    ss=rfm_df.groupby('segment').agg(n=('cid','count'),rev=('monetary','sum'),clv=('clv','mean')).reset_index().sort_values('rev',ascending=False)
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Customers",f"{len(rfm_df):,}")
    k2.metric("Champions",f"{len(rfm_df[rfm_df['segment']=='Champions']):,}")
    k3.metric("At Risk",f"{len(rfm_df[rfm_df['segment']=='At Risk']):,}")
    k4.metric("Total Revenue",f"₹{rfm_df['monetary'].sum()/100000:.1f}L")
    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(ss,values='n',names='segment',color='segment',color_discrete_map=COLORS,title="Customer Segments")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.bar(ss,x='segment',y='rev',color='segment',color_discrete_map=COLORS,title="Revenue by Segment")
        fig.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0.1)')
        st.plotly_chart(fig,use_container_width=True)
    fig=px.scatter(rfm_df.sample(400),x='recency',y='frequency',size='monetary',color='segment',
        color_discrete_map=COLORS,title="RFM Space — size = spend")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig,use_container_width=True)

with tab2:
    st.subheader("Predict Customer Lifetime Value")
    c1,c2=st.columns(2)
    with c1:
        rec=st.slider("Days since last purchase",1,365,30)
        freq=st.slider("Orders (lifetime)",1,50,8)
        mon=st.number_input("Total spend (₹)",100,500000,12000,500)
    with c2:
        rs=st.slider("Recency Score (1-5)",1,5,4)
        fs=st.slider("Frequency Score (1-5)",1,5,3)
        ms=st.slider("Monetary Score (1-5)",1,5,3)
        if st.button("💰 Predict CLV",use_container_width=True):
            clv=max(0,mon*(freq/max(rec,1))*12)
            score=rs+fs+ms
            seg='Champions' if score>=13 else ('Loyal Customers' if rs>=4 and fs>=3 else ('Promising' if rs>=4 else ('At Risk' if rs<=2 and score>=9 else ('Lost' if rs<=2 else 'Needs Attention'))))
            st.metric("Predicted 12M CLV",f"₹{clv:,.0f}")
            st.metric("Segment",seg)
            st.metric("RFM Score",f"{score}/15")

with tab3:
    df2=df.copy()
    df2['om']=df2['date'].dt.to_period('M')
    df2['fm']=df2.groupby('cid')['date'].transform('min').dt.to_period('M')
    df2['ci']=(df2['om']-df2['fm']).apply(lambda x:x.n)
    c=df2.groupby(['fm','ci'])['cid'].nunique().reset_index()
    cp=c.pivot(index='fm',columns='ci',values='cid').fillna(0)
    cr=(cp.divide(cp[0],axis=0)*100).round(1)
    cr.index=cr.index.astype(str)
    cr=cr.iloc[:,:7]
    fig=px.imshow(cr.fillna(0),color_continuous_scale=['#111118','#7c6af7','#c8f04a'],
        title="Cohort Retention (%)",text_auto='.0f')
    fig.update_layout(height=400)
    st.plotly_chart(fig,use_container_width=True)

with tab4:
    st.subheader("Marketing Strategy Playbook")
    acts={'Champions':['🎁 Enroll in VIP loyalty programme','📣 Leverage as brand ambassadors','🛍️ Personalised upsell'],
          'At Risk':['📧 Win-back email + 15% discount','🔔 Push notification: new arrivals','🎯 Retargeting ads'],
          'Promising':['🏆 Onboard to loyalty programme','📦 Free shipping threshold nudge','⭐ Introduce subscription'],
          'Lost':['📉 Suppress from regular campaigns','🎰 Last-attempt deep discount offer','🗄️ Archive after 12 months']}
    for seg,actions in acts.items():
        with st.expander(f"{seg} — {len(rfm_df[rfm_df['segment']==seg]):,} customers"):
            for a in actions: st.markdown(f"• {a}")
    st.caption("Built by Prajwal Markal Puttaswamy | [Portfolio](https://prajwalmarkalputtaswamyportfolio.netlify.app/)")
