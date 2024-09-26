from tkinter import *
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/Exercise/canada_per_capita_income.csv')
data["income"] = data["per capita income (US$)"].apply(lambda x: x)
data = data.drop(columns="per capita income (US$)")
reg = linear_model.LinearRegression()
X = data[["year"]].values
Y = data[["income"]].values
reg.fit(X,Y)
a  = reg.coef_[0][0]
b= reg.intercept_[0]


root = Tk()
root.title("Accomodation price prediction")
root.geometry("360x180")
my_entry = Entry(root)


def graph():
    plt.scatter(data.year,data.income)
    X = np.linspace(1970,2020,10000)
    Y =a*X+b
    plt.plot(X, Y, color='r')
    plt.show()


def result():
    y = float(reg.predict([[int(my_entry.get())]])[0][0])
    mesg = f"Predicted price: {round(y,2)} $."
    res.config(text=mesg)
    my_entry.delete(0,END)


my_label1 = Label(root, text="You can see a graph presenting the trend in accomodation prices:")
my_label2 = Label(root,text="Write down a year to predict:")
space = Label(root,text=" ")
my_button1= Button(root, text="See graph", command=graph)
my_button2 = Button(root, text="Accept", command=result)
my_entry = Entry(root)
res=Label(root,text="")

my_label1.grid(row=1,column=0)
my_button1.grid(row =5,column=0)
my_label2.grid(row=8)
my_entry.grid()
my_button2.grid()
res.grid()



root.mainloop()