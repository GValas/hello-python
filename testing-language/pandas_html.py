import pandas as pd
s = pd.Series(list('abca'))


print(s)


print(pd.get_dummies(s))