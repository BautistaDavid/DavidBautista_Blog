#!/usr/bin/env python
# coding: utf-8

# ## Python Class To Get Descriptive Statistics 
# 
# 
# <p><span style="font-family: Helvetica;"> I'm going to create a python class that allows us to reference an object and that its attributes are information related to the main descriptive statistics of a sample of n observations. This is an easy exercise to identify how the logic of classes, objects, functions and attributes works in python. </span></p>
# 
# <p><span style="font-family: Helvetica;">First of all it is important to identify the formulas of all the descriptive statistics that we are going to build in code using python. It is also important to say that <span style="color: rgb(220, 25, 27);"> the main idea of this exercise is not to use any pre-existing Python modules.</span></span></p>
# 
# #### What Are We Going To Calculate? 
# 
# Let's compute the following basic descriptive statistics: the mean, standard deviation, variance, median, kurtosis, skewness, and coefficient of variation. The formulas for those descriptive statistics are as follows:
# 
# **Mean:**
# 
# $$\bar{x} = \frac{\sum_{i=1}^{n} x_{i}}{n}$$
# 
# **Standard Deviation:**
# 
# $$\sigma = \sqrt{\frac{\sum_{i=1}^{n} (x_{i} - \bar{x})^2}{n-1}}$$
# 
# **Variance:**
# 
# $$\sigma^2 = \frac{\sum_{i=1}^{n} (x_{i} - \bar{x})^2}{n-1}$$
# 
# **Kurtosis:**
# 
# $$k = \frac{\sum_{i=1}^{n} (x_{i} - \bar{x})^4}{n *\sigma^4}$$
# 
# **Skewness:**
# 
# $$k = \frac{\sum_{i=1}^{n} (x_{i} - \bar{x})^3}{n *\sigma^3}$$
# 
# **Coefficient of Variation:**
# 
# $$Cv = \frac{\sigma}{|\bar{x}|}$$
# 
# Now that we have the formulas of what we want to calculate, we simply have to set up the class and start creating methods for each statistic, even though we could also generate them as attributes of the object.

# In[1]:


class statistics:
  
  def __init__(self,lst):
    self.lst = lst
    return None

  def mean(self): 
    return sum(self.lst) / len(self.lst)
  
  def stand_dev(self):
    return (sum([(i - self.mean())**2 for i in self.lst]) / (len(self.lst)-1))**0.5  

  def variance(self):
    return self.stand_dev()**2

  def median(self):
    return (sorted(self.lst)[len(self.lst)//2] if len(self.lst)%2 != 0 
            else (sorted(self.lst)[len(self.lst)//2-1]+sorted(self.lst)[len(self.lst)//2]) / 2)
  
  def kurtosis(self):
    return  sum([(i - self.mean())**4 for i in self.lst]) / (len(self.lst)*self.stand_dev()**4)

  def Skewness(self):
    return  sum([(i -self.mean())**3 for i in self.lst]) / (len(self.lst)*self.stand_dev()**3)
  
  def coeff_variation(self):
    return self.stand_dev() / abs(self.mean())


# #### Let's Test The Class
# 
# That's it, our class has different methods that allow the object to calculate each statistic, let's try it!. To test our class we are going to generate a normal distribution using ```numpy```, which has approximately mean 10, standard deviation 2.5 and with $n$ equal to 100

# In[2]:


import numpy as np
# Let's create the array
lst = np.random.normal(10,2.5,1000)

my_object = statistics(lst) # Nos let's create the object and use the methods

print(f'Mean = {my_object.mean()}')
print(f'Standard Deviation = {my_object.stand_dev()}')
print(f'Variance = {my_object.variance()}')
print(f'Median = {my_object.median()}')
print(f'Kurtosis = {my_object.kurtosis()}')
print(f'Skewness = {my_object.Skewness()}')
print(f'Coefficient of Variation = {my_object.coeff_variation()}')


# In[3]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
plt.suptitle('Histogram Array',fontsize=16)
plt.hist(lst,color='red',alpha=0.7, bins=50)
plt.show()


# ```python
# 
# class statistics:
#   
#   def __init__(self,lst):
#     self.lst = lst
#     return None
# 
#   def mean(self): 
#     return sum(self.lst) / len(self.lst)
#   
#   def stand_dev(self):
#     return (sum([(i - self.mean())**2 for i in self.lst]) / (len(self.lst)-1))**0.5  
# 
#   def variance(self):
#     return self.stand_dev()**2
# 
#   def median(self):
#     return (sorted(self.lst)[len(self.lst)//2] if len(self.lst)%2 != 0 
#             else (sorted(self.lst)[len(self.lst)//2-1]+sorted(self.lst)[len(self.lst)//2]) / 2)
#   
#   def kurtosis(self):
#     return  sum([(i - self.mean())**4 for i in self.lst]) / (len(self.lst)*self.stand_dev()**4)
# 
#   def Skewness(self):
#     return  sum([(i -self.mean())**3 for i in self.lst]) / (len(self.lst)*self.stand_dev()**3)
#   
#   def coeff_variation(self):
#     return self.stand_dev() / abs(self.mean())
# 
# ```

# 
