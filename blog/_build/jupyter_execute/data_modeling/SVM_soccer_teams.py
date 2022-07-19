#!/usr/bin/env python
# coding: utf-8

# #**Proyecto 2, Aprendizaje Supervisado - Clasificación**
# 
# ---
# 
# Nombres 
# 
# *   David Felipe Bautista Bernal 
# *   Angie Milena Prieto
# *   Santiago Villalobos Barrera 
# 
# > **Nota: Para poder visualizar el coigo interactivo se sugiere abrir el NoteBook desde Google Colab, puesto que GitHub no permite visualizar los elementos interactivos del modulo ```ipywidgets```**
# 

# # ***Algotimo de clasficacíon SVM***
# ## **Equipos de las grandes ligas europeas clasificados a Torneos Continentales**
# 
# Este notebook presenta un codigo interactivo bajo el uso del modulo ```ipywidgets```, el cual permitira visualizar los resultados de un modelo de aprendisaje supervisado de clasificaion (SVM ) planteado entre diferetnes variables que el usuario podra seleccionar.
# 
# El contexto del modelo es que se busca una hiperplano que clasifque a los equipos de las grandes ligas europeas (La liga ~ España, Serie A ~ Italia,  Premier League ~ Inglaterra, Bundes Liga ~ Alemania, League one ~ Francia, Primeria Liga ~ Portugal) que al acabar una temporada logran clasificar a torneos continentales como las UEFA Champions League, UEFA Europa League o la UEFA Europa Conference League bajo el uso de determinadas variables de control que se relacionan con el desempeño de los diferentes Equipos 

# ## **Datos**
# 
# Los datos usados hacen referencia a los resultado de los equipos para las temporadas 2019-2020 y 2020,-2021, los cuales fueron extraidos del portal FBRB y convertidos a fomato txt, para despues cargarlos al reposiotrio de trabajo y poder acceder a esta información desde URL's

# In[1]:


import pandas as pd 
from ipywidgets import widgets,interact,interactive
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
url_1="https://raw.githubusercontent.com/BautistaDavid/Team-1-Machine-Learning/main/five_principals_2020.txt"
url_2="https://raw.githubusercontent.com/BautistaDavid/Team-1-Machine-Learning/main/portugal_2020.txt"
url_3="https://raw.githubusercontent.com/BautistaDavid/Team-1-Machine-Learning/main/five_principals_2019.txt"
url_4="https://raw.githubusercontent.com/BautistaDavid/Team-1-Machine-Learning/main/potugal_2019.txt"

clasificados_2020=['Inter', 'Bayern Munich', 'Manchester City', 'Atlético Madrid',
       'Real Madrid', 'Lille', 'Paris S-G', 'Milan', 'Barcelona',
       'Atalanta', 'Juventus', 'Monaco', 'Napoli', 'Sevilla', 'Lyon',
       'Manchester Utd', 'RB Leipzig', 'Dortmund', 'Liverpool', 'Lazio',
       'Wolfsburg', 'Chelsea', 'Eint Frankfurt', 'Leicester City',
       'West Ham', 'Tottenham', 'Roma', 'Real Sociedad',
       'Betis', 'Marseille','Villarreal', 'Rennes', 'Leverkusen','Sporting CP',
      'Porto', 'Benfica', 'Braga', 'Paços', 'Santa Clara']

clasificados_2019=["Liverpool","Paris S-G","Bayern Munich","Real Madrid","Juventus",
                   "Barcelona","Inter","Manchester City","Atalanta","Lazio","Dortmund",
                   "Marseille","RB Leipzig","M'Gladbach","Leverkusen","Atlético Madrid",
                   "Sevilla","Roma","Rennes","Lille","Manchester Utd","Chelsea","Milan",
                   "Leicester City","Napoli","Villarreal","Tottenham","Hoffenheim","Arsenal",
                   "Real Sociedad","Granada","Nice","Reims","Wolfsburg","Porto","Benfica","Braga",
                   "Sporting CB","Rio Ave"]


# ### **Procesamiento de Datos**
# 
# Se manejan los diferentes DataFrame para poder genera una sola base de datos que contenga la infromacion de las dos temporadas y se genera unas variables adicionales que permitiran comparar las difeentes comparaciones entorno al hecho de que algunos quipos jugaron una cantidad diferente de partidos que otros.
# 
# Las princiaples varibles a tener en cuenta son ```Ratio_victorias``` ```Ratio_derrotas``` ```Pts/P``` ```golesF/P``` ```golesC/P```	 

# In[2]:


five_principals_2020=pd.read_csv(url_1)
portugal_2020=pd.read_csv(url_2)
five_principals_2019=pd.read_csv(url_3)
portugal_2019=pd.read_csv(url_4)

clasificados=[clasificados_2019,clasificados_2020]

for i in ([five_principals_2019,five_principals_2020]):
  i.drop(columns=["xG","xGA","xGD","xGD/90","Asistencia","LgRk","Máximo Goleador del Equipo","Portero"],inplace=True)

for j in ([portugal_2019,portugal_2020]):
  j.insert(10,"Pts/P",j["Pts"]/j["PJ"])
  j.insert(2,"País","por PT")
  j.drop(columns=["Asistencia","Notas","Máximo Goleador del Equipo","Portero"],inplace=True)
  #aca podriamos agregar año
datos_2020=pd.concat([five_principals_2020,portugal_2020])
datos_2019=pd.concat([five_principals_2019,portugal_2019])

for i in ([datos_2020,datos_2019]):
  i["golesF/P"]=i.GF/i.PJ
  i["golesC/P"]=i.GC/i.PJ
  i.insert(11,"Ratio_victorias",i.PG/i.PJ)
  i.insert(12,"Ratio_derrotas",i.PP/i.PJ)

datos_2020["competicion_europea"]=datos_2020["Equipo"].isin(clasificados_2020)
datos_2019["competicion_europea"]=datos_2019["Equipo"].isin(clasificados_2019)

datos=pd.concat([datos_2020,datos_2019])
datos.head()


# In[3]:


datos.describe()


# ## **Modelo y Codigo Interactivo**

# Se genera una gama de diferentes funciones que permiten obtener infromación entrno al modelo a realizar, esto con el fin de poder implementarlas dentro del codigo interactivo.

# ### **Generando Funciones** 

# In[4]:


def informacion_principal(x_1,x_2):
  ''' Esta funcion imprime infomacion sobre las 
  variables seleccionadas dentro del codigo interactivo'''
  print("Variable X_1",x_1)
  print("Variable X_2:",x_2)
  print(" ")

def variables_modelo(x_1,x_2):
  '''Esta funcion genera las diferentes variables usadas
  dentro del desarollo del modelo '''
  global X,y,X_train,X_eval,y_train,y_eval
  X=datos[[x_1,x_2]]
  y=datos["competicion_europea"]
  X_train,X_eval,y_train,y_eval=train_test_split(X,y,test_size=0.4,random_state=777)

def codigo_modelo(x_1,x_2):
  '''Esta funcion es el codigo princiapl del clasificador,
   ademas de genera una variable en relación al tiempo de 
   ejecucíon del modelo'''
  global clf,y_pred_in,y_pred_out,y_pred_proba_out,support_vector_,coef_x1,coef_x2,matriz_c_out,intercepto
  clf=svm.SVC(kernel="linear",probability=True)
  start_time=time.process_time()
  clf.fit(X_train,y_train)
  execution_time=time.process_time()-start_time
  print(f"Tiempo de ejecución del modelo: {execution_time} segundos")

  y_pred_in=clf.predict(X_train)
  y_pred_out=clf.predict(X_eval)

  y_pred_proba_out=clf.predict_proba(X_eval)
   
  support_vector_ = clf.support_vectors_
  intercepto=clf.intercept_[0]
  coef_x1=clf.coef_[0][0]
  coef_x2=clf.coef_[0][1]
  matriz_c_out=confusion_matrix(y_eval,y_pred_out)

def datos_graficar(x_1,x_2):
  '''Esta funcion genera los datos que permiten graficar los 
  diferentes resultados del modelo'''
  global datos_total_0,datos_total_1,datos_train_0,datos_train_1,datos_eval_0,datos_eval_1,datos_train,datos_eval

  datos_total_0=datos.loc[datos.loc[:,"competicion_europea"]==0]
  datos_total_1=datos.loc[datos.loc[:,"competicion_europea"]==1]

  datos_train=pd.concat([X_train,y_train],axis=1)
  datos_train_0=datos_train.loc[datos_train.loc[:,"competicion_europea"]==0]
  datos_train_1=datos_train.loc[datos_train.loc[:,"competicion_europea"]==1]

  datos_eval=pd.concat([X_eval,y_eval],axis=1)
  datos_eval_0=datos_eval.loc[datos_eval.loc[:,"competicion_europea"]==0]
  datos_eval_1=datos_eval.loc[datos_eval.loc[:,"competicion_europea"]==1]

def graficas_modelo(x_1,x_2):
  '''Esta Funcion genera las graficas de los resultados del modelo'''
  print(" ")
  fig,ax=plt.subplots(1,3,figsize=(15,5))
  plt.subplots_adjust(left=0.1, right=0.9,top=0.9, hspace=0.4)
  x=np.linspace(-10,10,100)

  ax[0].scatter(datos_train_0[x_1].values,datos_train_0[x_2].values,alpha=0.5,color="orange",label="No Clasificados")
  ax[0].scatter(datos_train_1[x_1].values,datos_train_1[x_2].values,alpha=0.5,color="blue",label="Clasificados")
  ax[0].set_title("SVM Muestra De Entrenamiento")

  ax[1].scatter(datos_eval_0[x_1].values,datos_eval_0[x_2].values,alpha=0.5,color="orange",label="No Clasificados")
  ax[1].scatter(datos_eval_1[x_1].values,datos_eval_1[x_2].values,alpha=0.5,color="blue",label="Clasificados")
  ax[1].set_title("SVM Muestra De Evaluacion")

  ax[2].scatter(datos_total_0[x_1].values,datos_total_0[x_2].values,alpha=0.5,color="orange",label="No Clasificados")
  ax[2].scatter(datos_total_1[x_1].values,datos_total_1[x_2].values,alpha=0.5,color="blue",label="Clasificados")
  ax[2].set_title("SVM Total de la Muestra")

  for i in range(3):
    ax[i].plot(x,(intercepto/-coef_x2)+(coef_x1/-coef_x2)*x,color="black")
    ax[i].scatter(support_vector_[:,0], support_vector_[:,1], color='red',alpha=0.7,label="Vectores de Soporte")
    ax[i].set_xlim(datos[x_1].min()-0.2,datos[x_1].max()+0.2)
    ax[i].set_ylim(datos[x_2].min()-0.2,datos[x_2].max()+0.2)
    ax[i].grid()
    ax[i].set_xlabel(x_1,fontsize=14)
    ax[i].set_ylabel(x_2,fontsize=14)
    ax[i].legend()

def resultados_modelo(x_1,x_2):
  '''esta funcion imprime un DataFrame con los princiaples
  resultados del modelo'''
  rest={"":["Intercepto","Coeficiente x_1","Coeficiente x_2"],
        "Resultados":[intercepto,coef_x1,coef_x2]}
  resultados=pd.DataFrame(rest)
  resultados.set_index("",inplace=True)
  print(resultados)
  print() 
  

def evaluacion_modelo(x_1,x_2):
  '''Esta Funcion imprime parametros de Evaluacion del modelo '''
  error_cuadratico=mean_squared_error(y_eval.astype(np.float32),y_pred_out.astype(np.float32))
  score=accuracy_score(y_eval.astype(np.float32),y_pred_out.astype(np.float32))
  print(f"El error cuadratico del modelo fuera de muestra es:{error_cuadratico}")
  print(f"El puntaje de precision del modelo fuera de muestra es:{score}")


def informacion_out_sample(x_1,x_2,opcion):
  print(" ")
  if opcion=="Matriz de confusion":
    print("Matriz de confusión muestra de Evaluación:")
    print(" ")
    matriz=pd.DataFrame(matriz_c_out)
    print(matriz)
  if opcion=="Matriz probabilidades":
    print("Matriz de probabilidades de Evaluación:")
    print(" ")
    print(y_pred_proba_out)
  if opcion=="Matriz predicciones":
    print("Matriz de predicciones de clase:")
    print(" ")
    print(y_pred_out.astype(np.float32))


# ### **Codigo Interactivo**

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import mean_squared_error,accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')

def codigo_interactivo(x_1,x_2,opcion):
  informacion_principal(x_1,x_2) #Funcion definida previamente
  variables_modelo(x_1,x_2)      #Funcion definida previamente 
  codigo_modelo(x_1,x_2)        #Funcion definifa previamente
  print(" ")

  datos_graficar(x_1,x_2)      #Funcion definida previamente 
  graficas_modelo(x_1,x_2)     # Funcion definida previamente 
  
  resultados_modelo(x_1,x_2)   #Funcion definida previamente 
  evaluacion_modelo(x_1,x_2)   #Funcion definida previamente 
 
  plt.figure(figsize=(3,0.5))
  plt.text(0.2,0.5,f"Ecuación Hiperplano: {round(intercepto,5)}+ {round(coef_x1,5)}"+r'$x_1+$'+\
           f"{round(coef_x2,5)}"+r'$x_2 = 0$',fontsize=15)
  plt.axis("off")




# In[6]:


print("Codigo Interactivo")
interactive_plot=interactive(codigo_interactivo,x_1=widgets.RadioButtons(options=['Pts/P', 'Ratio_victorias',"Ratio_derrotas", 'golesF/P', 'golesC/P'],
                          layout={'width': 'max-content'},description="Variable X_1:", continuos_update=False),
                          x_2=widgets.RadioButtons(options=['golesF/P','Pts/P', 'Ratio_victorias', 'Ratio_derrotas', 'golesC/P'],
                          layout={'width': 'max-content'},description="Variable X_2", continuos_update=False),
                          opcion=widgets.RadioButtons(options=["Matriz de confusion","Matriz probabilidades","Matriz predicciones"],description="Información Adiccional")

)
interactive_plot


# In[ ]:




