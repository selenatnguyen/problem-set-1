'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

# Your code here
pred = pd.read_csv('data/pred_universe_raw.csv', parse_dates=['arrest_date_univ'])
events = pd.read_csv('data/arrest_events_raw.csv', parse_dates=['arrest_date_event'])

df_arrests = pred.copy()
df_arrests = df_arrests[df_arrests['arrest_date_univ'].notna()]
def get_y(row):
    start_date = row['arrest_date_univ'] + pd.Timedelta(days=1)
    end_date = row['arrest_date_univ'] + pd.Timedelta(days=365)
    future_felonies = events[
        (events['person_id'] == row['person_id']) &
        (events['arrest_date_event'] >= start_date) &
        (events['arrest_date_event'] <= end_date) &
        (events['charge_degree'] == 'felony')
    ]
    return 1 if not future_felonies.empty else 0
df_arrests['y'] = df_arrests.apply(get_y, axis=1)


print("The number of people that got arrested again for a felony within a year?")
print(df_arrests['y'].mean())

df_arrests = pd.merge(
    df_arrests,
    events[['person_id', 'arrest_date_event', 'charge_degree']],
    left_on=['person_id', 'arrest_date_univ'],
    right_on=['person_id', 'arrest_date_event'],
    how='left'
)

df_arrests['current_charge_felony'] = (df_arrests['charge_degree'] == 'felony').astype(int)
print("How many current charges are felonies?")
print(df_arrests['current_charge_felony'].mean())
    
def get_last_year(row):
    past = events[
        (events['person_id'] == row['person_id']) &
        (events['arrest_date_event'] < row['arrest_date_univ']) &
        (events['arrest_date_event'] >= row['arrest_date_univ'] - pd.Timedelta(days=365)) &
        (events['charge_degree'] == 'F')
    ]
    return len(past)

df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(get_last_year, axis=1)
print("What's the average number of felony arrests last year?")
print(df_arrests['num_fel_arrests_last_year'].mean())
print(pred.head()) 
df_arrests.to_csv('data/df_arrests.csv', index=False)  




