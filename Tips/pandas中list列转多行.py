issue_key date     pkey          component              case_count
0  1060  2018-03-08  PROJ  console,configuration,management    8   
1  1464  2018-04-24  PROJ2 protocol                            1   
2  611   2017-03-31  PROJ  None                                2
3  2057  2018-10-30  PROJ  ha, console                         0

# code========================================================================
# convert to list
dd['component'] = dd['component'].str.split(',')

# convert list of pd.Series then stack it
dd = (dd
 .set_index(['issue_key','date','pkey','case_count'])['component']
 .apply(pd.Series)
 .stack()
 .reset_index()
 .drop('level_4', axis=1)
 .rename(columns={0:'component'}))

       issue_key        date   pkey  case_count      component
0       1060  2018-03-08   PROJ           8        console
1       1060  2018-03-08   PROJ           8  configuration
2       1060  2018-03-08   PROJ           8     management
3       1464  2018-04-24  PROJ2           1       protocol
4        611  2017-03-31   PROJ           2           None
5       2057  2018-10-30   PROJ           0             ha
6       2057  2018-10-30   PROJ           0        console

