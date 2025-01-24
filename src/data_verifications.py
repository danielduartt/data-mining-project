import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
x = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
#print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
#print(predict_students_dropout_and_academic_success.variables) 

#print(X.head())
#print(x.dtypes)


#armazena o tipo das colunas que faltam dados
def type_column_missing_value(data):
    columns_with_missing_values = data.columns[data.isnull().any()]
    column_types = {}  # Dicionário para armazenar os tipos de cada coluna
    for column in columns_with_missing_values:
        column_type = data[column].dtype
        column_types[column] = column_type  # Armazenar tipo por coluna
        print(f"'{column}': {column_type}")
    return column_types

print(type_column_missing_value(x))

# Soma todos os dados faltantes por coluna
#missing_data = x.isnull().sum()

# Soma todos os dados faltantes em todo dataset
#total_missing = missing_data.sum()

#Verifica se falta algum dado
#if total_missing == 0:
#    print("Sem dados faltantes")
#else:
 #   print(f"Total de dados faltantes: {total_missing}")
 #   print(missing_data)