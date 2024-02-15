# -*- coding: utf-8 -*-
"""
Association Rule 
@author: Janhavi

"""

from mlxtend.frequent_patterns import apriori
''' Here we are going to use transactional data wherein size of each row we can not use pandas to load this unstructured data here function is called open() is used to create AN empty list 
'''
groceries=[]
with open("E:/datascience/Association_Rule/groceries.csv") as f:groceries=f.read()
''' splitting the data into separate transactions using separator, it is comma separator , we can use new line character "\n"

'''
groceries=groceries.split("\n")
''' Earlier groceries datastructure was in string format, now it will change to 9836, each item is comma separated , our main aim is to calculate #A, #C, 
We will have to separate out each item from each transaction'''

groceries_list=[]

for i in groceries:
    groceries_list.append(i.split(","))
    
'''split function will separate each item from each list,whenever it will find in order to generate association rules, you can directly use groceries_list, Now let us separate out each item from the groceries list
'''
all_groceries_list=[i for item in groceries_list for i in item]
# You will get all the item occured in all transactions
# We will get 43368 items in various transactions

# Now let us count the frequency of each item
# We will import collections package which has Counter Function which will count

from collections import Counter
item_frequencies = Counter(all_groceries_list)
'''item_frequencies is basically dictionary having x[0] as key and x[1] = values
we want to access values and sort based on the count that occured in it. It will show the count of each item purchased in every transaction 
Now let us sort these frequencies in ascending order 
'''
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])

'''
When we execute this, item frequencies will be in sorted form, in the form of tuple item name with count 
Let us separate out items and their count 
'''
item=list(reversed([i[0] for  i in item_frequencies]))
#This is list comprehensions for each item in ite frequencies access the key here you will get items list

frequencies=list(reversed([i[1] for i in item_frequencies]))
# here you will get count of purchase of each items
# Now let us plot bar graph of item frequencies

import matplotlib.pyplot as plt
# Here we are taking frequncies from zero to 11, you can try 0-15 or any other 
plt.bar (height=frequencies[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),item[0:11])
# plt.xticks, You can specify a rotation for the tick labels in degrees or with keywords.
plt.xlabels("items")
plt.ylabel("count")
plt.show()
#___________________________________________________________________________________

import pandas as pd
#___________________________________________________________________________________

''' Now let us try to establish association rule mining we have groceries list in the list format, we need to convert it in dataframe'''
#___________________________________________________________________________________


groceries_series = pd.DataFrame(pd.Series(groceries_list))
#___________________________________________________________________________________

# Now we will get dataframe of size 9836X1 size, column comprises of multiple 
# we had extra row created, check the groceries_series, last row is empty, let us first delete it
#___________________________________________________________________________________

groceries_series=groceries_series.iloc[:9835,:]
#___________________________________________________________________________________

'''
We have taken rows from 0 to 9834 and columns 0 to all
groceries series has column having name 0,let us rename as transactions 
'''
#___________________________________________________________________________________

groceries_series.columns=['Transactions']
#___________________________________________________________________________________

'''
Now we will have to apply 1-hot encoding, before that in one column there are various items separated by ',
',' let us separate it with '*'
'''
#___________________________________________________________________________________

x=groceries_series['Transactions'].str.join(sep='*')
#check the x in variable explorer which has  * separator rather the ','
x=x.str.get_dummies(sep='*')
#___________________________________________________________________________________

'''
You will get one hot encoded dataframe of size 9835X169
This is our input data to apply to apriori algorithm, it will generate !169 that is 0.0075(it must be between 0 to 1),
you can give any number but must be between 0 and 1'''
#___________________________________________________________________________________

frequent_itemset=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
#___________________________________________________________________________________

'''
You will get support values for 1,2,3 and 4 max items
let us sort these support values
'''
#___________________________________________________________________________________

frequent_itemset.sort_values('support',ascending=False,inplace=True)
#___________________________________________________________________________________

'''
Support values will be sorted in descending order
Even EDA was also have the same trend, in EDA there was count and here it is support value we will generate association rules, This associaiton rule will calculate all the matrix of each and every combination '''
#___________________________________________________________________________________
rules = association_rules(frequent_itemset,metric='lift',min_threshold=1)
#___________________________________________________________________________________

# This generate association rules of size 1198X9 columns comprises of antescends, consequences
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)


