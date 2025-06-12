import pandas as pd
df=pd.read_csv("diamond.csv")
# print(df)


# duplicates=df.duplicated()
# print(duplicates)

# duplicate_value = df[df.duplicated(keep = False)].drop_duplicates()
# print(duplicate_value)

# count_row=df.shape[0]
# print(count_row)

# column_count=df.shape[1]
# print(column_count)

# import matplotlib.pyplot as plt

# plt.show(df.plot.hist(x="price",y="carat"))

# p=df.head(10)
# plt.show(p.plot.hist(x="carat",y="price"))


# t=df.tail(10)
# plt.show(t.plot.hist(x="carat",y="price"))

# e=df.iloc[10:-10]
# plt.show(e.plot.hist(x="carat",y="price"))


