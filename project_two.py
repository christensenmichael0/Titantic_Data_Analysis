# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:57:24 2016

@author: Michael Christensen
"""
#Import packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#Load the data into a pandas dataframe
titanic_df=pd.read_csv('C:/Users/Michael/Desktop/Data Science Udacity/PS2/titanic_data.csv')

#Check to see if there are any repeat PassangerId and remove from Dataframe if there are
original_size=len(titanic_df['PassengerId'])
unique_size=len(pd.unique(titanic_df['PassengerId'].ravel()))
if original_size==unique_size:
    print('No repeat passenger IDs')
    
 
 #Question: What is the age distibution onboard
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
plt.figure(1)
bins=np.arange(0,np.max(titanic_df['Age']),2)
plt.hist(np.array(titanic_df[titanic_df['Age'].notnull()]['Age']),bins=bins)
plt.title('Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')

mean_age=np.array(titanic_df[titanic_df['Age'].notnull()]['Age']).mean()
standard_error_mean_age=np.std(np.array(titanic_df[titanic_df['Age'].notnull()]['Age']))/np.sqrt(len(
np.array(titanic_df[titanic_df['Age'].notnull()]['Age'])))

#calculate a 95% confidence interval for the population mean
t_val=2.2461 #for 713 degrees of freedom and confidence interval =95% (2-tailed)
lower_average=mean_age-(t_val*standard_error_mean_age)
upper_average=mean_age+(t_val*standard_error_mean_age)

#Question: Did a higher proportion of those with family survive?
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#Make a new Family column by combining Parch and SibSp
titanic_df['Family'] =  titanic_df['Parch'] + titanic_df['SibSp']
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

#proportions
family_percent = titanic_df[['Family', 'Survived']].groupby(['Family'],as_index=False).mean()
family_percent_array=np.array(family_percent['Survived']) #create an np array to use for plt.bar
#sum survived
sum_survived_withfam =len(np.where((titanic_df['Survived']==1) & (titanic_df['Family']==1))[0])
sum_survived_nofam =len(np.where((titanic_df['Survived']==1) & (titanic_df['Family']==0))[0])
#sum died
sum_died_withfam =len(np.where((titanic_df['Survived']==0) & (titanic_df['Family']==1))[0])
sum_died_nofam =len(np.where((titanic_df['Survived']==0) & (titanic_df['Family']==0))[0])

#create bar graph
plt.figure(2)

indx=np.arange(0,4,2)
bar_width=.75

plt.subplot(121)
rects1=plt.bar(indx+bar_width,[sum_died_nofam, sum_died_withfam],width=bar_width,color=['red','red'])
rects2=plt.bar(indx, [sum_survived_nofam, sum_survived_withfam],width=bar_width,color='green')

plt.ylabel('Number that Survived or Died')
plt.title('Survival & Death Counts')
plt.xticks(indx + bar_width, ('Without Family', 'With Family'))
plt.legend((rects2[0], rects1[0]), ('Survived', 'Died'),loc='upper right',fontsize=8,frameon=True,shadow=True)

plt.subplot(122)
indx=np.arange(2)
bar_width=.75
bar_handle=plt.bar(indx, family_percent_array,width=bar_width,color=['red','green'])

plt.ylabel('Proportion that Survived')
plt.title('Passengers With/Without Family')
plt.xticks(indx + bar_width/2, ('Without Family', 'With Family'))

#Question: How does survival proportion compare for adults, children, and teenagers?
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#Group ages into child, teenager, adult and look at average survival
def convert_age(age):
    if age>18:
         return 'Adult'
    elif age>12:
         return 'Teenager'
    elif age<=12:
         return 'Child'
    else:
         return None
        
titanic_df['Age_Group'] = titanic_df['Age'].apply(convert_age)

#proportions
age_group_percent=titanic_df[['Age_Group','Survived']].groupby(['Age_Group'],as_index=False).mean()
age_group_percent_array=np.array(age_group_percent['Survived'])
#sum survived
sex_sum_survived_child =len(np.where((titanic_df['Survived']==1) & (titanic_df['Age_Group']=='Child'))[0])
sex_sum_survived_teenager =len(np.where((titanic_df['Survived']==1) & (titanic_df['Age_Group']=='Teenager'))[0])
sex_sum_survived_adult =len(np.where((titanic_df['Survived']==1) & (titanic_df['Age_Group']=='Adult'))[0])
#sum died
sex_sum_died_child =len(np.where((titanic_df['Survived']==0) & (titanic_df['Age_Group']=='Child'))[0])
sex_sum_died_teenager =len(np.where((titanic_df['Survived']==0) & (titanic_df['Age_Group']=='Teenager'))[0])
sex_sum_died_adult =len(np.where((titanic_df['Survived']==0) & (titanic_df['Age_Group']=='Adult'))[0])


#create bar graph
plt.figure(3)

indx=np.arange(0,6,2)
bar_width=.75

plt.subplot(121)
rects1=plt.bar(indx, [sex_sum_survived_adult, sex_sum_survived_child, sex_sum_survived_teenager],width=bar_width,color='green')
rects2=plt.bar(indx+bar_width,[sex_sum_died_adult,sex_sum_died_child, sex_sum_died_teenager],width=bar_width,color='red')

plt.ylabel('Number that Survived or Died')
plt.title('Survival & Death Counts')
plt.xticks(indx + bar_width, ('Adult', 'Child','Teenager'))
plt.legend((rects1[0], rects2[0]), ('Survived', 'Died'),loc='upper right',fontsize=8,frameon=True,shadow=True)

plt.subplot(122)
indx=np.arange(3)
bar_width=.75
bar_handle=plt.bar(indx, age_group_percent_array,width=bar_width,color=['red','green','blue'])

plt.ylabel('Proportion that Survived')
plt.title('Childen/Teenagers/Adults')
plt.xticks(indx + bar_width/2, ('Adult', 'Child','Teenager'))

#Question: Is there a difference between the survival proportions of females versus males
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#proportion
sex_percent = titanic_df[['Survived','Sex']].groupby(['Sex'],as_index=False).mean()
sex_percent_array=np.array(sex_percent['Survived']) #create an np array to use for plt.bar
#sum survived
sex_sum_survived_male =len(np.where((titanic_df['Survived']==1) & (titanic_df['Sex']=='male'))[0])
sex_sum_survived_female =len(np.where((titanic_df['Survived']==1) & (titanic_df['Sex']=='female'))[0])
#sum died
sex_sum_died_male =len(np.where((titanic_df['Survived']==0) & (titanic_df['Sex']=='male'))[0])
sex_sum_died_female =len(np.where((titanic_df['Survived']==0) & (titanic_df['Sex']=='female'))[0])

#create bar graph
plt.figure(4)
indx=np.arange(0,4,2)
bar_width=.75

plt.subplot(121)
rects1=plt.bar(indx, [sex_sum_survived_female, sex_sum_survived_male],width=bar_width,color=['green','green'])
rects2=plt.bar(indx+bar_width,[sex_sum_died_female, sex_sum_died_male],width=bar_width,color=['red','red'])

plt.ylabel('Number that Survived or Died')
plt.title('Survival & Death Counts')
plt.xticks(indx + bar_width, ('Females', 'Males'))
plt.legend((rects1[0], rects2[0]), ('Survived', 'Died'),loc='upper left',fontsize=8,frameon=True,shadow=True)

plt.subplot(122)

indx=np.arange(2)
bar_width=.8
bar_handle=plt.bar(indx, sex_percent_array,width=bar_width,color=['green','red'])

plt.ylabel('Proportion that Survived')
plt.title('Females Vs Males (all ages)')
plt.xticks(indx + bar_width/2, ('Females', 'Males'))



#Did older people have higher fares? Plot survival as well to see if there are any trends
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#only include rows where all relevant data is included.. i.e. Survived, Age, Pclass, Fare
complete_rows=np.where((titanic_df['Pclass']>=1) & (titanic_df['Pclass']<=3)
& (titanic_df['Age'].notnull()) & (titanic_df['Fare']>=0) & (titanic_df['Survived'].notnull()))[0]

survived_np_array=np.array(titanic_df['Survived'])
survived_subset=survived_np_array[complete_rows]

age_np_array=np.array(titanic_df['Age'])
age_subset=age_np_array[complete_rows]

fare_np_array=np.array(titanic_df['Fare'])
fare_subset=fare_np_array[complete_rows]
fare_subset_std=np.std(fare_subset)
fare_subset_scaled=fare_subset/fare_subset_std

#calculate Pearson's r and the 95% confidence interval
pearsons_r=np.corrcoef(age_subset,fare_subset)[0,1] #correlation coeff between age and fare
z = np.arctanh(pearsons_r) #convert pearsons_r to z
standard_error=1/np.sqrt(len(fare_subset)-3) #standard error the sampling distribution of z
lower_limit = z - (1.96*standard_error) 
upper_limit = z + (1.96*standard_error) 
np.tanh([lower_limit, upper_limit])


fill_colors=np.zeros((np.size(complete_rows),3),dtype=float)
fill_colors[survived_subset==1,:]=[0,1,0]
fill_colors[survived_subset==0,:]=[1,0,0]

#create a scatter plot
plt.figure(5)
#plot a couple of points first to create handles for the legend
handle_didnot_survive=plt.scatter(age_subset[0],fare_subset_scaled[0],c=fill_colors[0,:])
handle_survived=plt.scatter(age_subset[1],fare_subset_scaled[1],c=fill_colors[1,:])

scatter_handle=plt.scatter(age_subset,fare_subset_scaled,c=fill_colors)

plt.text(3.256048387096774, 11.014379319998188,'r='+ "{:.3f}".format(pearsons_r) +
' for Age vs. Fare')
plt.xlim(0, np.max(age_subset+5))
plt.ylim(0, np.max(fare_subset_scaled+2))
plt.title('Age, Fare, and Survival')
plt.xlabel('Age (years)')
plt.ylabel('Fare / St.Dev of Fare')

plt.legend((handle_survived,handle_didnot_survive),('Survived','Died'),
scatterpoints=1,loc='upper right',fontsize=8,frameon=True,shadow=True)

#Question: Is the median fare higher for 1st class passengers than 2nd or 3rd class
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
Pclass_np_array=np.array(titanic_df['Pclass'])
Fare_np_array=np.array(titanic_df['Fare'])

Pclass_1_indx=np.where((titanic_df['Pclass']==1) & (titanic_df['Fare']>=0))[0]
Pclass_1_subset=Fare_np_array[Pclass_1_indx]
Pclass1_total=len(Pclass_1_indx)

Pclass_2_indx=np.where((titanic_df['Pclass']==2) & (titanic_df['Fare']>=0))[0]
Pclass_2_subset=Fare_np_array[Pclass_2_indx]
Pclass2_total=len(Pclass_2_indx)

Pclass_3_indx=np.where((titanic_df['Pclass']==3) & (titanic_df['Fare']>=0))[0]
Pclass_3_subset=Fare_np_array[Pclass_3_indx]
Pclass3_total=len(Pclass_3_indx)

#create box plot
plt.figure(6)

plt.subplot(121)
indx=np.arange(3)
bar_width=.75
bar_handle=plt.bar(indx, [Pclass1_total,Pclass2_total,Pclass3_total],width=bar_width,color=['green','blue','red'])
plt.xticks((bar_width/2.)+indx, ('Class1', 'Class2','Class3'))
plt.ylabel('Number of Passengers')
plt.title('Passengers By Class')

plt.subplot(122)
plt.boxplot([Pclass_1_subset,Pclass_2_subset,Pclass_3_subset], 0, '') #no outliers

plt.ylabel('Fare')
plt.xlabel('Passenger Class')
plt.title('Fare Vs. Passenger Class')
plt.xticks([1,2,3],['Class 1','Class 2','Class 3'])
plt.ylim(-2, 180)
plt.text(2.0495967741935486, 173.66324200913243,'*No Outliers Shown')

#Question: Is there a relationship between port of embarkation and ticket fare??
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
Pclass_np_array=np.array(titanic_df['Embarked'])
Fare_np_array=np.array(titanic_df['Fare'])

Embarked_C_indx=np.where((titanic_df['Embarked']=='C') & (titanic_df['Fare']>=0))[0]
Embarked_C_subset=Fare_np_array[Embarked_C_indx]
Embarked_C_total=len(Embarked_C_indx)

Embarked_Q_indx=np.where((titanic_df['Embarked']=='Q') & (titanic_df['Fare']>=0))[0]
Embarked_Q_subset=Fare_np_array[Embarked_Q_indx]
Embarked_Q_total=len(Embarked_Q_indx)

Embarked_S_indx=np.where((titanic_df['Embarked']=='S') & (titanic_df['Fare']>=0))[0]
Embarked_S_subset=Fare_np_array[Embarked_S_indx]
Embarked_S_total=len(Embarked_S_indx)

#create box plot
plt.figure(7)

plt.subplot(121)
indx=np.arange(3)
bar_width=.75
bar_handle=plt.bar(indx, [Embarked_C_total,Embarked_Q_total,Embarked_S_total],width=bar_width,color=['green','red','blue'])
plt.xticks((bar_width/2.)+indx, ('Cherbourg', 'Queenstown','Southampton'))
plt.ylabel('Number of Passengers')
plt.title('Embarkation')

plt.subplot(122)
plt.boxplot([Embarked_C_subset,Embarked_Q_subset,Embarked_S_subset], 0, '') #no outliers

plt.ylabel('Fare')
plt.xlabel('Embarkation Port')
plt.title('Fare Vs. Port of Embarkation')
plt.xticks([1,2,3],['Cherbourg', 'Queenstown','Southampton'])
plt.ylim(-2, 160)
plt.text(2.0629032258064521, 148.81164383561645,'*No Outliers Shown')

#Univariate analysis of Fare per request of reviewer
#------------------------------------------------------------------------
#------------------------------------------------------------------------

plt.figure(8)
plt.boxplot(titanic_df[titanic_df['Fare'].notnull()]['Fare'], 0, 'gD')
plt.ylim(-3,titanic_df[titanic_df['Fare'].notnull()]['Fare'].max()+10)
plt.title('Fare Boxplot')
plt.ylabel('Fare')
plt.xticks([1],['All Passengers'])

#Histograms showing age distribution of men and women
#------------------------------------------------------------------------
#------------------------------------------------------------------------
age_distribution=np.array(titanic_df['Age'])
female_indx=np.where((titanic_df['Sex']=='female') & titanic_df['Age'].notnull())[0]
male_indx=np.where((titanic_df['Sex']=='male') & titanic_df['Age'].notnull())[0]

female_age_subset=age_distribution[female_indx]
male_age_subset=age_distribution[male_indx]

plt.figure(9)

ax1=plt.subplot(121)
bins=np.arange(0,np.max(male_age_subset),2)
plt.hist(male_age_subset,bins=bins)
plt.title('Male Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')


plt.subplot(122,sharey=ax1)
bins=np.arange(0,np.max(male_age_subset),2)
plt.hist(female_age_subset,bins=bins)

plt.title('Female Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')

print('Done')
#REFERENCES
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic/notebook
#http://mathesaurus.sourceforge.net/matlab-python-xref.pdf
#http://stackoverflow.com/questions/14016247/python-find-integer-index-of-rows-with-nan-in-pandas
#http://pandas.pydata.org/pandas-docs/stable/indexing.html
#http://stattrek.com/hypothesis-test/difference-in-proportions.aspx?Tutorial=AP
#http://stackoverflow.com/questions/30390476/equivalent-of-rs-of-cor-test-in-python
#http://www.graphpad.com/quickcalcs/statRatio2/
#http://home.southernct.edu/~mugnor1/past/mat107/Lessons/chap8.doc