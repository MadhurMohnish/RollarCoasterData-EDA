# RollarCoasterData-EDA
I am using Pandas! This notebook includes a tutorial that can be found on the Medallion Data Science YouTube channel.

## **Step 0:** Imports and Reading Data ##
```
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('max_columns', 200)
```

```
df = pd.read_csv('Input File Location')
```

## **Step 1:** Data Understanding ##
* Dataframe Shape
* Head and Tail
* dtypes
* Describe

```
#checks the shape of the DataFrame
df.shape
```

IMG

```
#shows all the columns
df.columns
```

IMG

```
#checks the shape of the DataFrame
df.dtypes
```

IMG

```
#checks the shape of the DataFrame
df.describe()
```

IMG

```
#checks the shape of the DataFrame
df.shape
```


## Step 2: Data Preparation ##

* Dropping irrelevant columns and rows
* Identifying duplicated columns
* Renaming Columns
* Feature Creation

Now you need to Drop the Columns that you do not need to avoid unnecessary columns in the Data. For that, follow the below steps:
Copy the df.columns data and paste it in:

```
df = df[["Column Names Here"]]
```
**Now I have commented out the data that I don't want and saved it in the df**

```
df = df[['coaster_name',
    # 'Length', 'Speed',
    'Location', 'Status',
    # 'Opening date',
    #   'Type',
    'Manufacturer',
#     'Height restriction', 'Model', 'Height',
#        'Inversions', 'Lift/launch system', 'Cost', 'Trains', 'Park section',
#        'Duration', 'Capacity', 'G-force', 'Designer', 'Max vertical angle',
#        'Drop', 'Soft opening date', 'Fast Lane available', 'Replaced',
#        'Track layout', 'Fastrack available', 'Soft opening date.1',
#        'Closing date',
#     'Opened', 
    # 'Replaced by', 'Website',
#        'Flash Pass Available', 'Must transfer from wheelchair', 'Theme',
#        'Single rider line available', 'Restraint Style',
#        'Flash Pass available', 'Acceleration', 'Restraints', 'Name',
       'year_introduced',
        'latitude', 'longitude',
    'Type_Main',
       'opening_date_clean',
    #'speed1', 'speed2', 'speed1_value', 'speed1_unit',
       'speed_mph', 
    #'height_value', 'height_unit',
    'height_ft',
       'Inversions_clean', 'Gforce_clean']].copy()
```
**Now if you check the DataType of the "opening_date_clean", it is of object DataType but we know that it must be of the DateTime Type**
```
df['opening_date_clean'] = pd.to_datetime(df['opening_date_clean'])
```

```
# Rename our columns
df = df.rename(columns={'coaster_name':'Coaster_Name',
                   'year_introduced' : 'Year_Introduced',
                   'opening_date_clean' : 'Opening_Date',
                   'speed_mph' : 'Speed_mph',
                   'height_ft' : 'Height_ft',
                   'Inversions_clean' : 'Inversions',
                   'Gforce_clean' : 'Gforce'})
```

IMG

**Now check the SUm of all the NA values in each of the Column:**

```
df.isna().sum()
```

IMG

```
# Check for duplicate coaster name
df.loc[df.duplicated(subset=['Coaster_Name'])].head(5)
```

IMG

```
# Checking an example duplicate
df.query('Coaster_Name == "Crystal Beach Cyclone"')
```

IMG


* Now in the below code, I have extracted all the duplicate values that have the same values in all of the 3 given columns. 
* After that, I reset the Index. **Notice a "~" symbol (Tilda) at the start**. It just **inverts the commands**. It says that, select all the values that are not Duplicate in the Dataset.
* Further, I have assigned this non-duplicate Dataset in 'df' DataFrame
```
df = df.loc[~df.duplicated(subset=['Coaster_Name','Location','Opening_Date'])] \
    .reset_index(drop=True).copy()
```


## Step 3: Feature Understanding ##

(Univariate analysis)

* Plotting Feature Distributions
* Histogram
* KDE
* Boxplot
  
Now I am going to check which Years have the highest-lowest number of Roller Coasters set up:
```
df['Year_Introduced'].value_counts()
```

IMG


Now I am going to Plot a Bar Graph taking the Year Introduced and the Count of ROllar Coaster set in a particular year:
**Note: plt.show() removes any unnecessary data to have a more cleaner workspace**
```
ax = df['Year_Introduced'].value_counts() \
    .head(10) \
    .plot(kind='bar', title='Top 10 Years Coasters Introduced')
ax.set_xlabel('Year Introduced')
ax.set_ylabel('Count')
plt.show()
```

IMG

Now we will create a Histogram. Histogram is more useful for continuous data plotting like **Speed**. So, I'll plot **Speed data against the frequency**:
```
ax = df['Speed_mph'].plot(kind='hist',
                          bins=20,
                          title='Coaster Speed (mph)')
ax.set_xlabel('Speed (mph)')
```

IMG

```
ax = df['Speed_mph'].plot(kind='kde',
                          title='Coaster Speed (mph)')
ax.set_xlabel('Speed (mph)')
```


## Step 4: Feature Relationships ##

* Scatterplot
* Heatmap Correlation
* Pairplot
* Groupby comparisons


Now I will construct a Scatter Plot using the below command **(Speed vs Height)** column:
```
df.plot(kind='scatter',
        x='Speed_mph',
        y='Height_ft',
        title='Coaster Speed vs. Height')
plt.show()
```

IMG

Now analyze the below command, if you see the graph, there is a Third criterion that I have **introduced which is "Year_Introduced" using Hue**.
```
ax = sns.scatterplot(x='Speed_mph',
                y='Height_ft',
                hue='Year_Introduced',
                data=df)
ax.set_title('Coaster Speed vs. Height')
plt.show()
```

IMG

ANalyse this PairPlot: 
```
sns.pairplot(df,
             vars=['Year_Introduced','Speed_mph',
                   'Height_ft','Inversions','Gforce'],
            hue='Type_Main')
plt.show()
```

IMG

```
df.plot(kind='scatter',
        x='Speed_mph',
        y='Height_ft',
        title='Coaster Speed vs. Height')
plt.show()
```


## Step 5: Ask a Question about the data ##

* **Try to answer a question you have about the data using a plot or statistic.**

  **Q.** What are the locations with the fastest roller coasters (minimum of 10)?

```
ax = df.query('Location != "Other"') \
    .groupby('Location')['Speed_mph'] \
    .agg(['mean','count']) \
    .query('count >= 10') \
    .sort_values('mean')['mean'] \
    .plot(kind='barh', figsize=(12, 5), title='Average Coast Speed by Location')
ax.set_xlabel('Average Coaster Speed')
plt.show()
```

IMG
