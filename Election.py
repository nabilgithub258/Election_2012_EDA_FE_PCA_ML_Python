#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############################################################################################################################
#### We are currently working on a dataset from the 2012 Election, motivated solely by our passion for statistics and ######
#### data analysis. We want to make it clear that our approach is unbiased, and we are not making any assumptions or #######
#### expressing support for any party involved. Our goal is to explore the data objectively, and we do not wish for any ####
#### assumptions or judgments to be drawn from this dataset. ###############################################################
############################################################################################################################


# In[593]:


#########################################################################################################
######################### ELECTION 2012 DATA SET  #######################################################
#########################################################################################################


# In[594]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[595]:


#### getting the data

df = pd.read_csv('2012-general-election-romney-vs-obama.csv')


# In[596]:


df.head()


# In[597]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[598]:


df[df.duplicated()]                         #### no duplicates


# In[599]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[600]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[601]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[602]:


df.isnull().any()


# In[603]:


df.info()


# In[604]:


df.drop(columns='Question Text',inplace=True)

#### its best to drop Question text


# In[605]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### off to the next ones


# In[606]:


df[df['Source URL'].isna()]


# In[607]:


df.dropna(subset=['Source URL'],inplace=True)


# In[608]:


df[df['Source URL'].isna()]                    #### took care of null values in source url col


# In[609]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### 3 to go


# In[610]:


df.head()


# In[611]:


df[df['Number of Observations'].isna()]               #### because this is numerical col instead of dropping we will just fill it up with the mean


# In[612]:


df['Number of Observations'].fillna(df['Number of Observations'].mean(),inplace=True)


# In[613]:


df[df['Number of Observations'].isna()]          #### took care of it


# In[614]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### 2 to go


# In[615]:


df[df.Undecided.isna()]


# In[616]:


df['Others'] = (100 - (df.Obama + df.Romney))


# In[617]:


df.head()              #### we created a new column to sum of two columns which are Undecided and other so now we dont need them


# In[618]:


df.drop(columns=['Undecided','Other'],inplace=True)


# In[620]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### now our data is very clean


# In[621]:


####################################################################
############## Part IV - Feature Engineering
####################################################################


# In[622]:


df.head(2)            #### for now we will drop the URL cols


# In[623]:


df.drop(columns=['Pollster URL','Source URL'],inplace=True)


# In[624]:


df.head(1)


# In[625]:


df = df.reindex(columns=['Start Date','End Date','Entry Date/Time (ET)','Number of Observations','Population','Mode','Obama','Romney','Others','Partisan','Affiliation','Question Iteration','Pollster'])


# In[626]:


df.head()                   #### this is much better to understand now


# In[627]:


df.rename(columns={'Start Date':'Start_date',
                   'End Date':'End_date',
                   'Entry Date/Time (ET)':'Entry_date',
                   'Number of Observations':'Observations',
                   'Question Iteration':'Questions'},inplace=True)


# In[628]:


df.head(1)


# In[629]:


######################################################################
############## Part V - EDA
######################################################################


# In[630]:


df['Others'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='black')

plt.title('Election Others Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')


# In[631]:


df['Obama'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='blue',color='black')

plt.title('Election Obama Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')


# In[632]:


df['Romney'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Election Romney Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')


# In[633]:


df.plot(x='End_date',y=['Obama','Romney','Others'],linestyle='',marker='o',figsize=(20,7),color={'Obama':'blue',
                                                                                                 'Romney':'red',
                                                                                                 'Others':'black'})
#### seems a very close call but Obama is slighyly winning here


# In[634]:


g = sns.jointplot(x=df.Observations,y=df['Obama'],data=df,hue='Population')

g.fig.set_size_inches(17,9)


# In[635]:


df.Population.unique()


# In[636]:


df[df.Observations > 5000]                 #### these two are putting out data off, in short outliers


# In[637]:


df = df[df.Observations < 5000]


# In[638]:


df.head()


# In[639]:


custom = {'Likely Voters':'green',
          'Registered Voters':'purple',
          'Adults':'black',
          'Likely Voters - Republican':'red'}

g = sns.jointplot(x=df.Observations,y=df['Others'],data=df,hue='Population',palette=custom)

g.fig.set_size_inches(17,9)

#### we can clearly see that majority of population is registered voters and likely voters in Others col


# In[640]:


df.head()


# In[641]:


df['Winner'] = df.apply(lambda row: 'Obama' if row['Obama'] > row['Romney'] else ('Romney' if row['Obama'] < row['Romney'] else 'Tie'), axis=1)

#### I honestly prefer lambda and apply, you can use the fucntion and call it later but I prefer lambda


# In[642]:


df.head()


# In[643]:


df.Winner.unique()


# In[644]:


df.Winner.value_counts()                   #### clearly Obama has the lead here


# In[645]:


custom = {'Likely Voters':'green',
          'Registered Voters':'purple',
          'Adults':'black',
          'Likely Voters - Republican':'red'}

sns.catplot(x='Winner',y='Observations',data=df,kind='strip',height=7,aspect=2,legend=True,hue='Population',jitter=True,palette=custom)

#### clearly registered voters are dominating in the winner circle


# In[646]:


custom ={'Obama':'blue',
         'Romney':'red',
         'Tie':'black'}

sns.catplot(x='Population',y='Observations',data=df,kind='box',height=7,aspect=2,legend=True,hue='Winner',palette=custom)

#### this is interesting, likely voters shows support for Obama while Registered voters mean is same for both majority parties but it leans more towards Romney


# In[647]:


df['Winner_Num'] = df.Winner.map({'Obama':0,
                                  'Romney':1,
                                  'Tie':2})

#### this will help us to further plot and make sense of it


# In[648]:


df.head()


# In[649]:


pl = sns.FacetGrid(df,hue='Winner',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Observations',fill=True)

pl.set(xlim=(0,df.Observations.max()))

pl.add_legend()

#### seems like Obama peaks at observations 1000 and starts to decline in larger observations compared to Romney


# In[650]:


pl = sns.FacetGrid(df,hue='Affiliation',aspect=4,height=4)

pl.map(sns.kdeplot,'Observations',fill=True)

pl.set(xlim=(0,df.Observations.max()))

pl.add_legend()

#### note in the observations around 1000, Rep were highly affiliated then if you move forward with Observations we see a steep decline in Rep affiliation while Dem affiliation remains stable


# In[651]:


sns.catplot(x='Affiliation',data=df,kind='count',hue='Mode',height=7,aspect=2,palette='Set2')

#### seems like we had the most Affiliation with Dem via Live phone and Automated phone calls


# In[652]:


df.Affiliation.value_counts()


# In[653]:


custom = {'Likely Voters':'green',
          'Registered Voters':'purple',
          'Adults':'black',
          'Likely Voters - Republican':'red'}


sns.catplot(x='Affiliation',data=df,kind='count',hue='Population',height=7,aspect=2,palette=custom)

plt.savefig('Election_affiliation_population_catplot.jpeg', dpi=300, bbox_inches='tight')

#### interesting


# In[654]:


avg = df[['Obama','Romney','Others']].mean()

avg = pd.DataFrame(avg)


# In[655]:


avg.head()


# In[656]:


std = pd.DataFrame(df[['Obama','Romney','Others']].std())


# In[657]:


std.head()


# In[658]:



avg.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',linestyle='dashed',linewidth=4,markersize=20)

plt.scatter(avg.index[0], avg.iloc[0], color='blue', s=150, zorder=5)

plt.scatter(avg.index[1], avg.iloc[1], color='red', s=150, zorder=5)

plt.scatter(avg.index[2], avg.iloc[2], color='black', s=150, zorder=5)

plt.title('Election Avg Graph')

plt.xlabel('Candidates')

plt.ylabel('Density percentage')

#### from the avg it seems the race is very tight


# In[659]:



new_df = pd.concat([avg,std],axis=1)

#### we will make a new dataframe to combine avg and std together


# In[660]:


new_df.columns = ['Average','STD']


# In[661]:


new_df                   #### this dataframe will come very handy


# In[662]:


new_df['STD'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',linestyle='dashed',linewidth=4,markersize=20)

plt.scatter(new_df['STD'].index[0], new_df['STD'].iloc[0], color='blue', s=150, zorder=5)

plt.scatter(new_df['STD'].index[1], new_df['STD'].iloc[1], color='red', s=150, zorder=5)

plt.scatter(new_df['STD'].index[2], new_df['STD'].iloc[2], color='black', s=150, zorder=5)

plt.title('Election STD Graph')

plt.xlabel('Candidates')

plt.ylabel('Density percentage')


# In[663]:


df['Difference'] = (df.Obama - df.Romney)/100             #### we are making this column to better understand the dip of both candidates on a plot


# In[664]:


df.groupby(['Start_date'],as_index=False).mean()[['Start_date','Difference']].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

#### we clearly see some dip on Romney side, note y axis 0 means a tie and + means Obama and - means Romney


# In[665]:


poll_df = df.groupby(['Start_date'],as_index=False).mean()


# In[666]:


poll_df.drop(columns=['Questions'],inplace=True)


# In[667]:


poll_df.head()


# In[668]:


poll_df.plot('Start_date','Difference',linestyle='-',marker='o',figsize=(15,4),color='black',markerfacecolor='red')

#### here we clearly see the dip and also see the date at which it occured


# In[669]:


#### we see a major dip between 10-2011 and 1-12, lets investigate why is that case because it shows major support for Romney in those polls

poll_df.plot('Start_date','Difference',linestyle='dashed',marker='o',figsize=(15,4),xlim=(95,128),color='black',markerfacecolor='red',markersize=7,linewidth=2)


# In[670]:


poll_df[poll_df['Difference'] < -0.10]


# In[671]:


poll_df.plot('Start_date','Difference',linestyle='-',marker='o',figsize=(15,4),xlim=(95,128),color='black',markerfacecolor='red')

plt.axvline(x=95+2,linewidth=4,color='red')

plt.axvline(x=128-6,linewidth=4,color='red')

#### it seems like those are the dates where there a debate hence we see a major support for Romney after that


# In[672]:


poll_df = df.groupby(['End_date'],as_index=False).mean()


# In[673]:


poll_df.head()


# In[674]:


poll_df.plot('End_date','Difference',linestyle='-',marker='o',figsize=(15,4),color='black',markerfacecolor='red')

#### at the end date we see a massive support for both parties, lets investigate


# In[675]:


poll_df[poll_df['Difference'] < -0.10]           #### Romney major support 


# In[676]:


poll_df[poll_df['Difference'] > 0.15]             #### Obama major support 


# In[677]:


poll_df.plot('End_date','Difference',linestyle='-',marker='o',figsize=(15,4),xlim=(0,50),color='black',markerfacecolor='blue')

plt.axvline(x=2,linewidth=4,color='red')


#### one thing to note with Obama is that the support is constant with this major peak while in Romney case the peaks are more but the support is not constant like here


# In[678]:


#### I don't think there much more left to explore from this data set so we will move forward with donor data set now


# In[679]:


#############################################################################
############## Part VI - EDA - Donor Data Set
#############################################################################


# In[680]:



donor_df = pd.read_csv('Election_Donor_Data.csv')

#### ignore the warning, its column 6 with mixed values


# In[681]:


donor_df.info()


# In[682]:


donor_df[donor_df.duplicated()]                     #### will drop them


# In[683]:


donor_df = donor_df.drop_duplicates()


# In[684]:


donor_df[donor_df.duplicated()]                   #### no duplicates left


# In[685]:


donor_df.info()


# In[686]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(donor_df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


# In[687]:


donor_df[donor_df.contbr_employer.isnull()]                              #### we will fill this up with unknown, dropping this many rows will not benefit us


# In[688]:


donor_df['contbr_employer'] = donor_df['contbr_employer'].fillna('Unknown')


# In[689]:


donor_df[donor_df.contbr_employer.isnull()]


# In[690]:


fig, ax = plt.subplots(figsize=(25,10))

sns.heatmap(donor_df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


# In[691]:


donor_df[donor_df.contbr_occupation.isnull()]                     #### lets make this one Unknown too


# In[692]:


donor_df['contbr_occupation'] = donor_df['contbr_occupation'].fillna('Unknown')


# In[693]:


donor_df[donor_df.contbr_occupation.isnull()]                #### done


# In[694]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(donor_df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


# In[695]:


donor_df.receipt_desc                     #### we will drop this one for obvious reasons


# In[696]:


donor_df.memo_cd.unique()                      #### this column as well


# In[697]:


donor_df.memo_text.head()                      #### we wouldn't be working with column either


# In[698]:


donor_df.drop(columns=['receipt_desc','memo_cd','memo_text'],inplace=True)


# In[699]:


donor_df.isnull().any()


# In[700]:


donor_df.info()


# In[701]:


donor_df.dropna(subset=['contbr_city','contbr_st','contbr_zip'],inplace=True)


# In[702]:


donor_df.isna().any()


# In[703]:


donor_df.info()                              #### now its clean data


# In[704]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(donor_df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

plt.savefig('Donor_missing_data_heatmap_4.jpeg', dpi=300, bbox_inches='tight')


# In[705]:


donor_df.head()


# In[706]:


donor_df.contb_receipt_amt.value_counts()           #### seems like 100 is the most popular amount donated here


# In[707]:


donor_df.contbr_st.value_counts().plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black',markersize=7)

#### seems like CA is the highest donor state


# In[708]:


donor_df.contbr_st.value_counts()


# In[709]:


#### getting the mean and standard deviation for the amounts donated

donor_df.contb_receipt_amt.mean()


# In[710]:


donor_df.contb_receipt_amt.std()


# In[711]:


#### we see a huge difference in the mean and standard deviation which should never be the case

amount_df = donor_df.contb_receipt_amt.copy()


# In[712]:


donor_df.describe()


# In[713]:


amount_df[amount_df < 0].value_counts()            #### this doesn't make sense so we will drop these for now


# In[714]:


amount_df = amount_df[amount_df > 1]


# In[715]:


amount_df[amount_df < 0].value_counts()              #### got rid of donors who donated less then 1


# In[716]:


amount_df.value_counts()                 #### much better


# In[717]:


#### now lets see the avg and std

amount_df.mean()


# In[718]:


amount_df.std()            #### the only viable reason for his has to be some strong outliers, we will sort that out further down


# In[719]:


amount_df[amount_df < 1]            #### amount df has nothing less then 1 


# In[720]:


top_10 = amount_df.sort_values().value_counts().head(10)


# In[721]:


top_10.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='red',markersize=15,linestyle='dashed',linewidth=2)


#### in the top 10 we see the most is around 100 then it starts dipping but the donation increases but then it goes back to 1-100 range


# In[722]:


top_10               #### 100 is the most common followed by 50


# In[723]:


donor_df.drop(columns=['cmte_id','cand_id'],inplace=True)

donor_df.head()

#### for now we are dropping these two columns


# In[724]:


donor_df.cand_nm.unique()         #### we can do some feature engineering on this one


# In[725]:


#### making a new column for party 

party_map = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}


# In[726]:


donor_df['Party'] = donor_df.cand_nm.map(party_map)


# In[727]:


donor_df.Party.value_counts()                       #### seems like we have more Dem party


# In[728]:


donor_df.Party.value_counts().plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='blue',color='red',markersize=15,linestyle='dashed',linewidth=2)

plt.scatter(donor_df.Party.value_counts().index[0], donor_df.Party.value_counts().iloc[0], color='blue', s=150, zorder=5)

plt.scatter(donor_df.Party.value_counts().index[1], donor_df.Party.value_counts().iloc[1], color='red', s=150, zorder=5)

#### honestly the difference is quite big here as you can literally see


# In[729]:


donor_df.contb_receipt_amt.min()


# In[730]:


donor_df = donor_df[donor_df.contb_receipt_amt > 1]


# In[731]:


donor_df.contb_receipt_amt.min()


# In[732]:


donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()


# In[733]:


donor_df.groupby('cand_nm')['contb_receipt_amt'].count().plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='blue',color='red',markersize=15,linestyle='dashed',linewidth=2)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[0], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[0], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[1], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[1], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[2], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[2], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[3], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[3], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[4], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[4], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[5], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[5], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[6], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[6], color='blue', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[7], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[7], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[8], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[8], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[9], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[9], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[10], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[10], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[11], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[11], color='red', s=150, zorder=5)

plt.scatter(donor_df.groupby('cand_nm')['contb_receipt_amt'].count().index[12], donor_df.groupby('cand_nm')['contb_receipt_amt'].count().iloc[12], color='red', s=150, zorder=5)

#### we see the donations for Obama is the highest and it gets all distributed between other Republicans 


# In[734]:


cand_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()


# In[735]:


cand_amount.plot(kind='bar',figsize=(20,7),color='purple')

#### the republicans donations gets distributed while Obama is the only candidate from Democrats, hence the huge margine


# In[736]:


donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar',figsize=(20,7),color='purple')


#### this is interesting because if you sum all the donors then Republicans win, but because Democrat have only one candidate while Republicans have many, it acts to their disadvantage


# In[737]:


donor_df[['contbr_st','contb_receipt_amt','Party']]


# In[738]:


heat = donor_df.groupby(['contbr_st','Party'])['contb_receipt_amt'].sum().unstack()

heat.head()


# In[739]:


heat_filled = heat.fillna(0)                     #### filled Nan with 0 values

heat_filled.head()


# In[740]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(heat_filled,linewidths=0.1,ax=ax,cmap='viridis')


#### this is very important info, from this we can derive which state supports which party the most
#### it seems like Democrats have California as their strongest state while Republicans have texas as their strongest state


# In[741]:


heat_filled.sort_values(by='Republican',ascending=False).head(10)           #### top 10 Republican state donations


# In[742]:


Rep_heat = heat_filled.sort_values(by='Republican',ascending=False).head(10)


# In[743]:


fig, ax = plt.subplots(figsize=(25,15)) 

sns.heatmap(Rep_heat,linewidths=0.1,ax=ax,annot=True)


#### this is republicans donation top 10 state and the scale is in dollars


# In[744]:


heat_filled.sort_values(by='Democrat',ascending=False).head(10)            #### top 10 Democrats state donation wise


# In[745]:


Dem_heat = heat_filled.sort_values(by='Democrat',ascending=False).head(10)


# In[746]:


fig, ax = plt.subplots(figsize=(25,15)) 

sns.heatmap(Dem_heat,linewidths=0.1,ax=ax,cmap='viridis',annot=True)

#### this is democrats donation top 10 state and the scale is in dollars


# In[747]:


occupation_df = donor_df.copy()


# In[748]:


occupation_df = donor_df.groupby(['contbr_occupation','Party'])['contb_receipt_amt'].sum().unstack()

occupation_df

#### now we will do the similar treatment with regards to occupation
#### because it has more then 45k rows we will have to make some adjustment for the data to make any sense here


# In[749]:


occupation_df = occupation_df[occupation_df.sum(1) > 1000000]


# In[750]:


occupation_df.shape               #### now we have 31 rows, it had too many occupation so we have adjusted by telling it to give us only donors more then 1 million combined


# In[751]:


occupation_df.index                         #### we need to comibine CEO and drop information requested


# In[752]:


occupation_df.drop(index=['INFORMATION REQUESTED','INFORMATION REQUESTED PER BEST EFFORTS'],inplace=True)

#### ignore the warning, we should have made a copy but it did go through


# In[753]:


occupation_df.index


# In[754]:


occupation_df.index = occupation_df.index.to_series().replace('C.E.O.', 'CEO')


# In[755]:


occupation_df_combined = occupation_df.groupby(level=0).sum()


# In[756]:


occupation_df_combined.index           #### now we have a clean index


# In[757]:


occupation_df_combined.shape


# In[758]:


occupation_df_combined.head()


# In[759]:


fig, ax = plt.subplots(figsize=(25,15)) 

sns.heatmap(occupation_df_combined,linewidths=0.1,ax=ax,cmap='viridis',annot=True)

#### this is the most holistic approach to see who is donating to whom


# In[760]:


occupation_df_combined.plot(kind='barh',figsize=(20,12),color={'Democrat':'blue',
                                                               'Republican':'red'})

#### I think this is enough for EDA, now lets move to model phase


# In[761]:


donor_df['date'] = pd.to_datetime(donor_df['contb_receipt_dt'])


# In[762]:


donor_df['month'] = donor_df.date.apply(lambda x:x.month)

donor_df.month.head()


# In[763]:


donor_df['month_name'] = donor_df.month.map({1:'Jan',
                         2:'Feb',
                         3:'Mar',
                         4:'Apr',
                         5:'May',
                         6:'Jun',
                         7:'Jul',
                         8:'Aug',
                         9:'Sep',
                         10:'Oct',
                         11:'Nov',
                         12:'Dec'})


# In[764]:


donor_df['day_of_week'] = donor_df.date.apply(lambda x:x.dayofweek)


# In[765]:


donor_df['Day'] = donor_df.day_of_week.map({0:'Mon',
                                     1:'Tue',
                                     2:'Wed',
                                     3:'Thr',
                                     4:'Fri',
                                     5:'Sat',
                                     6:'Sun'})


# In[766]:


donor_df['year'] = donor_df.date.apply(lambda x:x.year)


# In[767]:


donor_df.head()


# In[768]:


donor_df.year.value_counts()


# In[769]:


#### we will make the best out of date

sns.catplot(x='Day',data=donor_df,kind='count',hue='Party',height=7,aspect=2,palette={'Republican':'red',
                                                                                      'Democrat':'blue'})


#### seems like the support for democrat was higher no matter the day of the week except Sunday which was evenly split, interesting


# In[770]:


pl = sns.FacetGrid(donor_df,hue='Party',aspect=4,height=4,palette={'Republican':'red',
                                                                   'Democrat':'blue'})

pl.map(sns.kdeplot,'month',fill=True)

pl.set(xlim=(0,donor_df.month.max()))

pl.add_legend()


#### interesting we have a big spike in donation towards democrats on month 3 and 4


# In[771]:


sns.catplot(x='year',data=donor_df,kind='count',hue='Party',height=7,aspect=2,palette={'Republican':'red',
                                                                                       'Democrat':'blue'})

#### definately we see a massive peak for support for democrats in 2012 compared to 2011


# In[772]:


pl = sns.FacetGrid(donor_df,hue='Party',aspect=4,height=4,palette={'Republican':'red',
                                                                   'Democrat':'blue'})

pl.map(sns.kdeplot,'day_of_week',fill=True)

pl.set(xlim=(0,donor_df.day_of_week.max()))

pl.add_legend()


#### interesting


# In[773]:


donor_df.month_name.unique()


# In[774]:


donor_df.head()


# In[775]:


donor_df.groupby('month_name')['Party'].count().plot(legend=True,figsize=(20,7),marker='o',markersize=14,markerfacecolor='black',linestyle='dashed',linewidth=4,color='red')

#### month july is the most donated month of the year


# In[776]:


heat = donor_df.groupby(['month_name','Day','Party'])['contb_receipt_amt'].sum().unstack().unstack()

heat

#### this gives us the most holistic view of which days and month related to which party


# In[777]:


fig, ax = plt.subplots(figsize=(25,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### seems like for democrats their best day was on Monday of April while for Republicans it was Friday September


# In[778]:


sns.lmplot(x='month',y='contb_receipt_amt',data=donor_df.groupby('month').count().reset_index(),height=7,aspect=2,line_kws={'color':'black'},scatter_kws={'color':'green'})


#### quite interesting


# In[779]:


donor_df.groupby('date').count()['Party'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=10,linestyle='dashed',color='red')

#### definately we see the major spike at the end of election which doesn't suprise us at all


# In[780]:


donor_df[donor_df.Party == 'Republican'].groupby('date').count()['year'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)


#### this is Republican graph as we see their major spike is at around 01-2012


# In[781]:


donor_df[donor_df.Party == 'Democrat'].groupby('date').count()['year'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='blue',color='black',markersize=10)

#### we see democrats biggest spike in April of 2012


# In[782]:


heat_2 = donor_df.groupby(by=['month_name','Day','year','Party'])['contb_receipt_amt'].sum().unstack().unstack().fillna(0)

heat_2


# In[783]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5)


#### from this we can draw a lot of insights


# In[784]:


heat_2 = donor_df.groupby(by=['month_name','year','Party'])['contb_receipt_amt'].sum().unstack().unstack().fillna(0)

heat_2


# In[785]:


fig, ax = plt.subplots(figsize=(30,15))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

plt.savefig('Donor_month_year_party_heatmap.jpeg', dpi=300, bbox_inches='tight')

#### here we only seeing the month year and party heatmap


# In[786]:


sns.clustermap(heat_2,cmap='viridis')

#### cluster map for better understanding


# In[500]:


################################################################################
######## Part VII - Model - Classification
################################################################################


# In[501]:


donor_df.head()


# In[504]:


X = donor_df.drop(columns=['cand_nm','contbr_nm','contb_receipt_dt','form_tp','Party','file_num','contbr_zip','date','month_name','Day'])


# In[505]:


X.head()


# In[506]:


X.contbr_occupation.nunique()


# In[507]:


y = donor_df.Party


# In[508]:


y.head()


# In[195]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[510]:


preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['contbr_city','contbr_st','contbr_employer','contbr_occupation']),
                                               ('num', StandardScaler(),['contb_receipt_amt','month','day_of_week','year'])
                                              ]
                                )


# In[196]:


from sklearn.pipeline import Pipeline


# In[512]:


from sklearn.linear_model import LogisticRegression


# In[519]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])


# In[197]:


from sklearn.model_selection import train_test_split


# In[521]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[522]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train,y_train)')


# In[523]:


y_predict = model.predict(X_test)


# In[198]:


from sklearn import metrics


# In[525]:


metrics.accuracy_score(y_test,y_predict)                        #### amazing result wasnt expecting this


# In[526]:


print(metrics.classification_report(y_test,y_predict))          #### honestly suprised with this metrics, was expecting much lower then this but I will take this without complains


# In[527]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Democrat','Republican']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(20,12))

disp.plot(ax=ax)

plt.savefig('Election_2012_confusion_matrix.jpeg', dpi=300, bbox_inches='tight')


# In[528]:


y.unique()


# In[529]:


from sklearn.ensemble import RandomForestClassifier                #### using random forest to see if we can get better result


# In[530]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1))
])


# In[531]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train,y_train)')


# In[532]:


y_predict = model.predict(X_test)


# In[533]:


print(metrics.classification_report(y_test,y_predict))              #### much better then previous model


# In[534]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Democrat','Republican']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(20,12))

disp.plot(ax=ax)

plt.savefig('Election_2012_confusion_matrix_random_forest.jpeg', dpi=300, bbox_inches='tight')


# In[535]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[536]:


import xgboost as xgb


# In[537]:


clf_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9]
}


# In[538]:


from sklearn.model_selection import RandomizedSearchCV            #### this one is giving us error due to y not being 0 and 1


# In[539]:


donor_df['Parties'] = donor_df.Party.map({'Republican':0,
                                          'Democrat':1})


# In[540]:


y = donor_df['Parties']

y.head()


# In[541]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[544]:


get_ipython().run_cell_magic('time', '', "\nrandom_search_xgb = RandomizedSearchCV(clf_xgb, param_grid_xgb, n_iter=50, cv=5, scoring='accuracy', random_state=42)\nrandom_search_xgb.fit(X_train, y_train)")


# In[545]:


best_model = random_search_xgb.best_estimator_


# In[546]:


y_predict = best_model.predict(X_test)


# In[547]:


print(metrics.classification_report(y_test,y_predict))              #### honestly not the best model


# In[548]:


############################################################################################################################
#### Due to the extensive size of the dataset and the limited computing power available, running model fits has been #######
#### taking more than four hours. Although there is potential to further improve the results with advanced methods, ########
#### our current computational resources restrict us from doing so. The best accuracy achieved so far is 0.90 using a ###### 
#### Random Forest classifier. Consequently, we have decided to halt further classification efforts at this point. #########
############################################################################################################################


# In[463]:


###################################################################################
##### Part VIII - Model - Regression
###################################################################################


# In[787]:


donor_df.head()                          #### our target is amount in dollars donated
                                         #### note we know this dataset is not suitable for regression model especially on amount donated but we will give it a shot


# In[788]:


mean_df = donor_df.contb_receipt_amt.mean()
std_df = donor_df.contb_receipt_amt.std()

print(mean_df,std_df)                             #### this is just horrible, look at std which means high value outliers 


# In[789]:


from scipy.stats import norm


# In[790]:


#### Comprehensive time

x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(20, 7))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')

#### areas under the curve
plt.fill_between(x, y, where=(x >= mean_df - std_df) & (x <= mean_df + std_df), color='black', alpha=0.2, label='68%')
plt.fill_between(x, y, where=(x >= mean_df - 2*std_df) & (x <= mean_df + 2*std_df), color='black', alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mean_df - 3*std_df) & (x <= mean_df + 3*std_df), color='black', alpha=0.2, label='99.7%')

#### mean and standard deviations
plt.axvline(mean_df, color='black', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 3*std_df, color='yellow', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 3*std_df, color='yellow', linestyle='dashed', linewidth=1)

plt.text(mean_df, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, plt.gca().get_ylim()[1]*0.05, f'z=1    {mean_df + std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, plt.gca().get_ylim()[1]*0.05, f'z=-1   {mean_df - std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=2  {mean_df + 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-2 {mean_df - 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=3  {mean_df + 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-3 {mean_df - 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')


#### annotate the plot
plt.text(mean_df, max(y), 'Mean', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, max(y), '-1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, max(y), '+1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, max(y), '-2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, max(y), '+2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, max(y), '-3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, max(y), '+3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')

#### labels
plt.title('Age distribution inside the Titanic Dataset')
plt.xlabel('Age')
plt.ylabel('Probability Density')

plt.legend()


#### the majority is mean which is 304 but z-score level 1 is 4121, so we can do one interesting thing, exclude anybody above that


# In[791]:


df = donor_df[donor_df.contb_receipt_amt < 4121].copy()


# In[792]:


df.count()


# In[793]:


df[df.contb_receipt_amt<0]              #### nobody below 0


# In[794]:


df[df.contb_receipt_amt<1]              #### nobody less then 1


# In[795]:


df.contb_receipt_amt.mean()


# In[796]:


df.contb_receipt_amt.std()                  #### still not ideal but we will take it 


# In[797]:


df.head()


# In[798]:


X = df.drop(columns=['cand_nm','contbr_nm','contbr_zip','contb_receipt_amt','contb_receipt_dt','form_tp','file_num','month_name','Day'])


# In[799]:


X.head()


# In[800]:


X['day'] = X.date.apply(lambda x:x.day)

X.day.head()


# In[801]:


X.drop(columns=['date','day_of_week'],inplace=True)

X.head()


# In[802]:


X.isnull().any()


# In[803]:


y = df['contb_receipt_amt']

y.head()


# In[804]:


y.isnull().any()


# In[805]:


X.head()


# In[806]:


from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder


# In[807]:


preprocessor = ColumnTransformer(transformers=[
                                               ('cat', TargetEncoder(), ['contbr_city','contbr_st','contbr_employer','contbr_occupation','Party']),
                                               ('num', StandardScaler(),['month','day','year'])

                                              ]
                                )


# In[808]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[809]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression(n_jobs=-1))
                       ])


# In[810]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[811]:


y_predict = model.predict(X_test)


# In[812]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### just not a good model honestly


# In[813]:


metrics.r2_score(y_test,y_predict)


# In[814]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### not a good fit obviously


# In[815]:


from sklearn.linear_model import Ridge


# In[816]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', Ridge(alpha=1.0))
                       ])


# In[817]:


X.head()


# In[818]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train,y_train)')


# In[819]:


y_predict = model.predict(X_test)


# In[820]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### its making a pattern which is always bad


# In[821]:


metrics.r2_score(y_test,y_predict)                      #### no improvement


# In[822]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))


# In[823]:


from sklearn.decomposition import PCA


# In[824]:



preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]), ['month','day','year']),
        ('cat', TargetEncoder(), ['contbr_city','contbr_st','contbr_employer','contbr_occupation','Party']),

    ])

#### lets see with PCA


# In[825]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression(n_jobs=-1))
                       ])


# In[826]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[827]:


y_predict = model.predict(X_test)


# In[828]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[829]:


metrics.r2_score(y_test,y_predict)


# In[830]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### PCA didn't help either


# In[831]:


from sklearn.ensemble import RandomForestRegressor


# In[832]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42,max_features='auto',n_estimators=100,n_jobs=-1))
])


# In[833]:



param_grid = {
    'regressor__n_estimators': [50,100],
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_split': [2],
    'regressor__min_samples_leaf': [1]
}


# In[834]:


from sklearn.model_selection import GridSearchCV


# In[835]:


grid_model = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=2)


# In[836]:


get_ipython().run_cell_magic('time', '', '\ngrid_model.fit(X_train, y_train)')


# In[837]:


best_model = grid_model.best_estimator_


# In[838]:


y_predict = best_model.predict(X_test)


# In[839]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[840]:


metrics.r2_score(y_test,y_predict)


# In[841]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### not helpful obviously


# In[843]:


from sklearn.model_selection import RandomizedSearchCV


# In[844]:


param_grid = {
    'regressor__n_estimators': [100, 200, 500],
    'regressor__max_features': ['auto', 'sqrt'],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__bootstrap': [True, False]
}


# In[845]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42,max_features='auto',n_estimators=100,n_jobs=-1))
])


# In[846]:


random_search = RandomizedSearchCV(model, param_grid, cv=3, random_state=42, verbose=2)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[535]:


best_model = random_search.best_estimator_


# In[536]:


y_predict = best_model.predict(X_test)


# In[537]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[538]:


metrics.r2_score(y_test,y_predict)


# In[539]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### this is the best we got so far


# In[541]:


preprocessor = ColumnTransformer(transformers=[
                                               ('cat', TargetEncoder(), ['contbr_city','contbr_st','contbr_employer','contbr_occupation','Party']),
                                               ('num', StandardScaler(),['month','day','year'])

                                              ]
                                )


# In[542]:


param_grid = {
    'regressor__n_estimators': [100, 200, 500],
    'regressor__max_features': ['auto', 'sqrt'],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__bootstrap': [True, False]
}


# In[543]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42,max_features='auto',n_estimators=100,n_jobs=-1))
])


# In[544]:


random_search = RandomizedSearchCV(model, param_grid, cv=3, random_state=42, verbose=2)


# In[545]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[546]:


best_model = random_search.best_estimator_


# In[547]:


y_predict = best_model.predict(X_test)


# In[548]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### just not a good model honestly


# In[549]:


metrics.r2_score(y_test,y_predict)                                      


# In[550]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              


# In[553]:


param_grid = {
    'regressor__n_estimators': [100, 200, 500],
    'regressor__max_depth': [3, 6, 9],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}


# In[555]:


import xgboost as xgb


# In[560]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42,n_jobs=-1))
])


# In[561]:


random_search = RandomizedSearchCV(model, param_grid, cv=3, random_state=42, verbose=2)


# In[562]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[563]:


best_model = random_search.best_estimator_


# In[564]:


y_predict = best_model.predict(X_test)


# In[565]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### just not a good model honestly


# In[566]:


metrics.r2_score(y_test,y_predict)


# In[567]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))  


# In[568]:


param_grid = {
    'regressor__n_estimators': [100, 200, 500],
    'regressor__max_depth': [10, 20, 30],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__num_leaves': [31, 62, 127]
}


# In[ ]:





# In[571]:


import lightgbm as lgb


# In[573]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(random_state=42,n_jobs=-1))
])


# In[574]:


random_search = RandomizedSearchCV(model, param_grid, cv=3, random_state=42, verbose=2)


# In[575]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[576]:


best_model = random_search.best_estimator_


# In[577]:


y_predict = best_model.predict(X_test)


# In[578]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### just not a good model honestly


# In[579]:


metrics.r2_score(y_test,y_predict)


# In[580]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict)) 


# In[581]:


param_grid = {
    'regressor__iterations': [100, 200, 500],
    'regressor__depth': [6, 8, 10],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__l2_leaf_reg': [3, 5, 7]
}


# In[583]:


from catboost import CatBoostRegressor


# In[585]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(random_seed=42, silent=True,thread_count=-1))
])


# In[586]:


random_search = RandomizedSearchCV(model, param_grid, cv=3, random_state=42, verbose=2)


# In[587]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[588]:


best_model = random_search.best_estimator_


# In[589]:


y_predict = best_model.predict(X_test)


# In[590]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### just not a good model honestly


# In[591]:


metrics.r2_score(y_test,y_predict)


# In[592]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))


# In[ ]:


############################################################################################################################
#### This dataset has proven to be suboptimal for regression models, as our attempts have consistently resulted in an ######
#### R² value no higher than 0.32. Consequently, we have decided to halt further development of this regression model.######
#### Despite the high variance in mean and standard deviation, making it unsuitable for predicting the donation amounts,####
#### we undertook this challenge to push our limits. If anyone has insights or suggestions on improving its performance,####
#### your input would be greatly appreciated. ##############################################################################
############################################################################################################################


# In[ ]:




