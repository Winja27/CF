import pandas as pd
data = {'a': {'x': 1, 'y': 2}, 'b': {'z': 3, 't': 4}}
df = pd.DataFrame.from_dict(data)
for k in data.items():
    print(k)

