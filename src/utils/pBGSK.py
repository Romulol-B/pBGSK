import numpy as np
import os
import pandas as pd
import random
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as  plt 
from src.utils.data_importer import data_loader

def k_factor(kf =0.95):
    resultado = random.random()
    if resultado >=kf:
        return 1
    else:
        return 0
    
class Individual:
    def __init__(self,data_set_name: str,individual_id :int, features: list,X_train,X_test,y_train, y_test,columns_names,knn_val =5):
        """Cada individuo possui um vetor binário representando a seleção de features."""
        self.data_set_name = data_set_name
        self.number_of_features = int(np.sum(features))
        self.individual_id = individual_id
        self.features = features # vetor [True,False,True]
        self.numeric_features = features +0# vetor [1,0,1]
        
        self.column_names = columns_names
        self.X_train = X_train
        self.X_test= X_test
        self.y_train = y_train
        self.y_test = y_test
        self.acc = 0
        self.score = 2
        self.classifier = KNeighborsClassifier(n_neighbors=knn_val)
        self.df = None

    def calculate_fitness(self):
        """ 1 - acuracia  + 1 - features utilizadas /total de features  """
        
        if sum(self.features) == 0:
            self.score = 2.0
            return

        # Select the features from the data
        X_train_selected = self.X_train.loc[:, self.features]
        X_test_selected = self.X_test.loc[:, self.features]

        # Train and score the model
        self.classifier.fit(X_train_selected, self.y_train)
        y_pred = self.classifier.predict(X_test_selected)
        self.acc =accuracy_score(self.y_test, y_pred)
        self.score = 1-self.acc + 1 -(sum(self.numeric_features))/len(self.features)# basico de 

class Population:

    def __init__(self, individuals: list,knowledge=0.95,partition = 0.1):
        self.individuals = individuals
        self.len = len(individuals)
        self.df =None
        self.geng_df =pd.DataFrame()
        self.d_junior =0
        self.d_senior =0 
        self.junior_features = None# vetor binario de features
        self.senior_features = None # vetor binario de features
        self.knowledge =knowledge# taxa de aprendizado
        self.partition = partition
        self.nfe =0
    def sort(self,t_sort ='fitness'):
        
        for indiv in self.individuals:
            indiv.calculate_fitness()
        if t_sort =='fitness':
            self.individuals.sort(key=lambda ind: ind.score)
            self.df.sort_values('score',ascending=False)
        elif t_sort =='accuracy':
            self.individuals.sort(key=lambda ind: ind.acc)
            self.df.sort_values('acc',ascending=False)
        else:
            print('Erro, em tipo de sort')# alguma maneira de passar um erro
            return -1
        self.nfe += len(self.individuals)


    def dimension_distribution(self,nfe_total):
        """nfe: Number of functions evaluations, quantidade de vezes que calculate_fitness foi chamada
        nfe_total: orçamento computacional total
        Atualiza a distribuição e retorna verdadeiro ou falso em relação da alteração da distribuição """
        d = len(self.individuals[0].features)-1
        novo_d_junior = min(round(d*(1-self.nfe/nfe_total)**self.knowledge),d-1)
        novo_d_senior = d - novo_d_junior
        diff = self.d_junior- novo_d_junior 
        #print(f'novo dimension distribution {novo_d_senior}')
        self.d_junior =novo_d_junior
        self.d_senior = novo_d_senior

        return  diff # a distribuição foi alterada
    def dimension_classification(self,nfe_total):
        d = len(self.individuals[0].features)
        if self.d_junior==0:#talvez nao precise desse if.
            self.dimension_distribution(nfe_total)
            idxs= np.array(random.sample(range(1,d),self.d_junior)) 
            self.junior_features = np.array([0]*d)
            for i in idxs:
                self.junior_features[idxs] = 1
            self.senior_features = np.array([1]*d) - self.junior_features
        else:
            diff = self.dimension_distribution(nfe_total) 
            jf = self.junior_features.copy()
            sf = self.senior_features.copy()
            
            while diff >0:
                wh = np.squeeze(np.where(jf>0))
                idx_r = int(random.random()*((wh.size)-1))#escolhe aleatoriamente um dos indexes disponiveis.
                jf[wh[idx_r]] =0
                sf[wh[idx_r]] =1
                diff =-1
            self.junior_features = jf
            self.senior_features = sf
        return

    def beginner_gsk(self,knowledge_ratio=0.95):    
        for t_idx in range(1,self.len-1):
             for dimension in np.where(self.junior_features >0):
                if random.random() <knowledge_ratio:
                    rand_idx = int(random.random()*self.len)
                    rand_indiv = self.individuals[rand_idx]
                    xt =self.individuals[t_idx]
                    
                    prev_idx = t_idx-1
                    next_idx = t_idx+1
                    t_prev =self.individuals[prev_idx] if (prev_idx <0) else self.individuals[prev_idx+1]
                    t_next =self.individuals[next_idx+1] if (next_idx +1<len(self.individuals)) else self.individuals[next_idx]

                    if xt.score>rand_indiv.score:
                        xtk =xt.numeric_features[dimension] + k_factor()*(t_prev.numeric_features[dimension] -t_next.numeric_features[dimension] +xt.numeric_features[dimension]- rand_indiv.numeric_features[dimension] )
                    else:
                        xtk =xt.numeric_features[dimension] +  k_factor()*(t_prev.numeric_features[dimension] -t_next.numeric_features[dimension] +xt.numeric_features[dimension]- rand_indiv.numeric_features[dimension] )
                    self.individuals[t_idx].features[dimension] =(xtk >0)
                    
                    self.individuals[t_idx].numeric_features[dimension] = (xtk >0) +0


    def intermediate_gsk(self,knowledge_ratio=0.95):
        len_p = int(self.len*(self.partition))
        for t_idx in range(1,self.len-1):
             for dimension in (np.where(self.senior_features)):
                if random.random() <knowledge_ratio:
                    rand_idx = int(random.random()*len(self.individuals))
                    rand_indiv = self.individuals[rand_idx]
                    xt =self.individuals[t_idx]

                    best_idx = np.random.randint(0,len_p)
                    middle_idx = np.random.randint(len_p,self.len-len_p)
                    worst_idx= np.random.randint(self.len-len_p,self.len)

                    best_x = self.individuals[best_idx]
                    middle_x = self.individuals[middle_idx]
                    worst_x = self.individuals[worst_idx]

                    if xt.score>rand_indiv.score:
                        xtk =xt.numeric_features[dimension] +  k_factor()*(best_x.numeric_features[dimension] -worst_x.numeric_features[dimension] + middle_x.numeric_features[dimension]- xt.numeric_features[dimension] )#ainda falta o kf
                    else:
                        xtk =xt.numeric_features[dimension] + k_factor()*(best_x.numeric_features[dimension] -worst_x.numeric_features[dimension] + xt.numeric_features[dimension]- middle_x.numeric_features[dimension] )# ainda falta o kf
                    self.individuals[t_idx].features[dimension] =(xtk >0)

                    self.individuals[t_idx].numeric_features[dimension] = (xtk >0) +0

    def population_reduction(self, nfe_total,low_b=0.9,high_b=0.95):
        np_min =self.len*low_b
        np_max =self.len*high_b
        old_len = self.len
        np_new = int((np_min - np_max) * (self.nfe / nfe_total) + np_max -1)

        ks =pd.DataFrame(self.df.loc[:,self.df.columns[:-3]].sum(axis=0))

        km = self.df.loc[:,['score','n_features','acc']].mean(axis=0).to_numpy()
        ks= pd.pivot_table(ks,columns =ks.index,values=0 )
        ks[['mean_score','mean_n_features','mean_acc']] = km 
        kf = pd.DataFrame(ks)
        self.geng_df = pd.concat([kf,self.geng_df])
    
        if np_new >12:#minimo viavel
            for i in range(old_len - np_new):
                self.individuals.pop()
                self.len -=1
            return False
        else:
            return True

    def ploting_score(self):
        import seaborn as sns

        df_p = pd.DataFrame()
        for column in self.df.columns[:-2]:
            row ={ 'feature':f'{column}','score':self.df[self.df.loc[:,column]]['score'],'n_features':self.df[self.df.loc[:,column]]['n_features']}
            row =pd.DataFrame(row)
            df_p=pd.concat([row,df_p],ignore_index=True)
        f, ax = plt.subplots(figsize=(21, 18))
        sns.boxplot(
            df_p, x="score", y="feature", hue="feature",
            whis=[0, 100], width=.6, palette="vlag"
        )
        sns.stripplot(df_p, x="score", y="feature", size=4, color=".3")

        ax.xaxis.grid(True)
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)

    def dataframe(self):
        
        lista_features = [indiv.features for indiv in self.individuals]
        pop_df = pd.DataFrame(np.array(lista_features),columns=self.individuals[0].column_names)
        pop_df['score']= [indiv.score for indiv in self.individuals]
        pop_df['n_features'] = pop_df.loc[:,pop_df.columns!='score'].sum(axis=1)
        pop_df['acc'] = [indiv.acc for indiv in self.individuals]
        self.df =pop_df
        return pop_df
def population_creation(num_population:int, lower_k :int, upper_k:int,data_tuple,data_set_name,columns_names,knowledge=0.95) -> Population:
    """Criando uma população inicial de indivíduos. com features variando entre lower_k e upper_k.
    num_population: número de indivíduos na população"""
    X_train, X_test, y_train, y_test = data_tuple
    population = []
    total_features = X_train.shape[1]
    for i in range(num_population):
        k = lower_k + random.random()*(upper_k-lower_k)
        k = int(k)
        features = random.sample(range(0, total_features), k)
        bin_feature = np.array([False]*total_features)

        for feature in features:
            bin_feature[feature] = True  # Mark selected

        indiv = Individual(data_set_name=data_set_name,
                individual_id=i,
                features=bin_feature,
                X_train=X_train,  # Pass the reference
                y_train=y_train,  # Pass the reference
                X_test=X_test,    # Pass the reference
                y_test=y_test,     # Pass the reference
                columns_names=columns_names
            )
        population.append(indiv)
    pop =Population(population,knowledge=knowledge)
    return pop
def feature_selection(data_tuple,num_population:int,nfe_total:int ,lower_k:int,upper_k:int,columns_names,k=0.95,p=0.9,data_set_name='dataset_1'):


    pop = population_creation(
        num_population=num_population,
        lower_k=lower_k,
        upper_k=upper_k,
        data_tuple=data_tuple,
        data_set_name=data_set_name,
        columns_names=columns_names,
        knowledge=k
    )
    for indiv in pop.individuals:
        indiv.calculate_fitness()

    pop.dataframe()#criando o dataframe
    pop.sort()#Rankeando os individuos
    best_score = 2 # 0 acc, com todas as features

    while pop.nfe<nfe_total and pop.len>12:
        pop.dimension_classification(nfe_total=nfe_total)
        pop.beginner_gsk()
        pop.intermediate_gsk()
        if pop.population_reduction(nfe_total=nfe_total):
            break
        for indiv in pop.individuals:
            indiv.calculate_fitness()
        pop.sort()
        pop.dataframe()
        if pop.individuals[0].score< best_score:
            best_individual_features =pop.individuals[0].features
            best_score = pop.individuals[0].score
    
        
    return pop
