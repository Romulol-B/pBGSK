import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def k_factor(kf=0.95):
    """Returns 1 if random value >= kf, otherwise 0."""
    return 1 if random.random() >= kf else 0


class Individual:
    def __init__(
        self,
        data_set_name: str,
        individual_id: int,
        features: np.ndarray,
        columns_names: list[str],
    ):
        """Cada individuo possui um vetor binário representando a seleção de features."""
        self.data_set_name = data_set_name
        self.individual_id = individual_id
        
        # Guardando como booleanos para máscara e inteiros para cálculos matemáticos
        self.features = np.array(features, dtype=bool)
        self.numeric_features = np.array(features, dtype=int)
        
        self.number_of_features = np.sum(self.numeric_features)
        self.column_names = columns_names
        
        self.acc = 0.0
        self.score = 2.0  # Pior score possível (1 - acc + 1 - ratio)
        self.df = None


class Population:
    def __init__(
        self,
        individuals: list,
        data_tuple: tuple,
        knowledge: float = 0.95,
        partition: float = 0.1,
        knn_val: int = 5,
    ):
        self.X_train,  self.X_test, self.y_train,self.y_test = data_tuple
        self.individuals = individuals
        self.len = len(individuals)
        
        # Classifier unificado para poupar memória
        self.classifier = KNeighborsClassifier(n_neighbors=knn_val)
        
        self.df = None
        self.geng_df = pd.DataFrame()
        
        self.d_junior = 0
        self.d_senior = 0
        self.junior_features = None  # vetor binario de features
        self.senior_features = None  # vetor binario de features
        
        self.knowledge = knowledge
        self.partition = partition
        self.nfe = 0

    def calculate_fitness(self, individual: Individual):
        """1 - acuracia + 1 - (features utilizadas / total de features)"""
        if np.sum(individual.features) == 0:
            individual.score = 2.0
            return

        # Seleciona apenas as features ativas
        X_train_selected = self.X_train.loc[:, individual.features]
        X_test_selected = self.X_test.loc[:, individual.features]

        # Treina e pontua o modelo
        self.classifier.fit(X_train_selected, self.y_train)
        y_pred = self.classifier.predict(X_test_selected)
        
        individual.acc = accuracy_score(self.y_test, y_pred)
        
        # Atualiza a quantidade de features antes de calcular o score
        individual.number_of_features = np.sum(individual.numeric_features)
        feature_ratio = individual.number_of_features / len(individual.features)
        
        individual.score = (1 - individual.acc) + (1 - feature_ratio)

    def sort(self, t_sort="fitness"):
        # Garante que todos foram avaliados
        for indiv in self.individuals:
            if indiv.score == 2.0 and np.sum(indiv.features) > 0: # Evita reavaliar desnecessariamente se já foi feito
                 self.calculate_fitness(indiv)
                 
        self.nfe += len(self.individuals)

        if t_sort == "fitness":
            self.individuals.sort(key=lambda ind: ind.score)
            if self.df is not None:
                self.df.sort_values("score", ascending=True, inplace=True) # Score menor é melhor
        elif t_sort == "accuracy":
            self.individuals.sort(key=lambda ind: ind.acc, reverse=True)
            if self.df is not None:
                self.df.sort_values("acc", ascending=False, inplace=True)
        else:
            raise ValueError("Tipo de sort inválido. Use 'fitness' ou 'accuracy'.")

    def dimension_distribution(self, nfe_total: int):
        """Calcula a distribuição das dimensões entre os estágios Junior e Senior."""
        d = len(self.individuals[0].features)
        novo_d_junior = min(
            round(d * (1 - self.nfe / nfe_total) ** self.knowledge), d - 1
        )
        novo_d_senior = d - novo_d_junior
        diff = self.d_junior - novo_d_junior
        
        self.d_junior = novo_d_junior
        self.d_senior = novo_d_senior

        return diff 

    def dimension_classification(self, nfe_total: int):
        d = len(self.individuals[0].features)
        
        if self.d_junior == 0:
            self.dimension_distribution(nfe_total)
            idxs = np.array(random.sample(range(0, d), self.d_junior))
            
            self.junior_features = np.zeros(d, dtype=int)
            self.junior_features[idxs] = 1
            self.senior_features = np.ones(d, dtype=int) - self.junior_features
        else:
            diff = self.dimension_distribution(nfe_total)
            jf = self.junior_features.copy()
            sf = self.senior_features.copy()

            while diff > 0:
                wh = np.squeeze(np.where(jf > 0))
                if wh.size == 0:
                    break
                idx_r = random.choice(wh) if wh.ndim == 1 else wh
                jf[idx_r] = 0
                sf[idx_r] = 1
                diff -= 1
                
            self.junior_features = jf
            self.senior_features = sf

    def beginner_gsk(self, knowledge_ratio=0.95):
        for t_idx in range(1, self.len - 1):
            for dimension in np.where(self.junior_features > 0)[0]:
                if random.random() < knowledge_ratio:
                    rand_idx = random.randint(0, self.len - 1)
                    rand_indiv = self.individuals[rand_idx]
                    xt = self.individuals[t_idx]

                    # t_idx garante bounds seguros entre 1 e self.len - 2
                    t_prev = self.individuals[t_idx - 1]
                    t_next = self.individuals[t_idx + 1]

                    kf = k_factor()
                    if xt.score > rand_indiv.score: # xt é pior que o aleatorio
                        xtk = xt.numeric_features[dimension] + kf * (
                            t_prev.numeric_features[dimension]
                            - t_next.numeric_features[dimension]
                            + xt.numeric_features[dimension]
                            - rand_indiv.numeric_features[dimension]
                        )
                    else:
                        xtk = xt.numeric_features[dimension] + kf * (
                            t_prev.numeric_features[dimension]
                            - t_next.numeric_features[dimension]
                            + xt.numeric_features[dimension]
                            - rand_indiv.numeric_features[dimension]
                        )
                    
                    new_val = 1 if xtk > 0 else 0
                    self.individuals[t_idx].features[dimension] = bool(new_val)
                    self.individuals[t_idx].numeric_features[dimension] = new_val

    def intermediate_gsk(self, knowledge_ratio=0.95):
        len_p = max(1, int(self.len * self.partition)) # Garante no minimo 1
        
        for t_idx in range(1, self.len - 1):
            for dimension in np.where(self.senior_features > 0)[0]:
                if random.random() < knowledge_ratio:
                    rand_idx = random.randint(0, self.len - 1)
                    rand_indiv = self.individuals[rand_idx]
                    xt = self.individuals[t_idx]

                    best_x = self.individuals[random.randint(0, len_p - 1)]
                    middle_x = self.individuals[random.randint(len_p, self.len - len_p - 1)]
                    worst_x = self.individuals[random.randint(self.len - len_p, self.len - 1)]

                    kf = k_factor()
                    if xt.score > rand_indiv.score:
                        xtk = xt.numeric_features[dimension] + kf * (
                            best_x.numeric_features[dimension]
                            - worst_x.numeric_features[dimension]
                            + middle_x.numeric_features[dimension]
                            - xt.numeric_features[dimension]
                        )
                    else:
                        xtk = xt.numeric_features[dimension] + kf * (
                            best_x.numeric_features[dimension]
                            - worst_x.numeric_features[dimension]
                            + xt.numeric_features[dimension]
                            - middle_x.numeric_features[dimension]
                        )
                        
                    new_val = 1 if xtk > 0 else 0
                    self.individuals[t_idx].features[dimension] = bool(new_val)
                    self.individuals[t_idx].numeric_features[dimension] = new_val

    def population_reduction(self, nfe_total, low_b=0.9, high_b=0.95):
        np_min = self.len * low_b
        np_max = self.len * high_b
        old_len = self.len
        np_new = int((np_min - np_max) * (self.nfe / nfe_total) + np_max)

        # Extrai métricas atuais para log
        km = self.df.loc[:, ["score", "n_features", "acc"]].mean().to_frame().T
        km.rename(columns={"score": "mean_score", "n_features": "mean_n_features", "acc": "mean_acc"}, inplace=True)
        self.geng_df = pd.concat([km, self.geng_df], ignore_index=True)

        if np_new >= 12 and np_new < old_len:
            # População deve estar ordenada; cortamos os piores do final
            amount_to_pop = old_len - np_new
            for _ in range(amount_to_pop):
                self.individuals.pop()
            self.len = len(self.individuals)
            return False
        return True

    def dataframe(self):
        lista_features = [indiv.numeric_features for indiv in self.individuals]
        pop_df = pd.DataFrame(lista_features, columns=self.individuals[0].column_names)
        
        pop_df["score"] = [indiv.score for indiv in self.individuals]
        pop_df["n_features"] = [indiv.number_of_features for indiv in self.individuals]
        pop_df["acc"] = [indiv.acc for indiv in self.individuals]
        
        self.df = pop_df
        return pop_df

    def ploting_score(self):
        df_melted = self.df.melt(
            id_vars=["score", "n_features", "acc"],
            var_name="feature",
            value_name="is_selected"
        )
        # Filtra apenas as features selecionadas
        df_p = df_melted[df_melted["is_selected"] == 1]

        f, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(
            data=df_p,
            x="score",
            y="feature",
            whis=[0, 100],
            width=0.6,
            palette="vlag",
            ax=ax
        )
        sns.stripplot(data=df_p, x="score", y="feature", size=4, color=".3", ax=ax)

        ax.xaxis.grid(True)
        ax.set(ylabel="Features", xlabel="Score")
        sns.despine(trim=True, left=True)
        plt.show()


def population_creation(
    num_population: int,
    lower_k: int,
    upper_k: int,
    data_tuple: tuple,
    data_set_name: str,
    columns_names: list,
    knowledge: float = 0.95,
) -> Population:
    
    X_train, X_test, y_train, y_test = data_tuple
    population = []
    total_features = X_train.shape[1]
    
    for i in range(num_population):
        k = random.randint(lower_k, upper_k)
        features_idx = random.sample(range(0, total_features), k)
        
        bin_feature = np.zeros(total_features, dtype=bool)
        bin_feature[features_idx] = True

        indiv = Individual(
            data_set_name=data_set_name,
            individual_id=i,
            features=bin_feature,
            columns_names=columns_names,
        )
        population.append(indiv)
        
    pop = Population(population, knowledge=knowledge, data_tuple=data_tuple)
    return pop


def feature_selection(
    data_tuple: tuple,
    num_population: int,
    nfe_total: int,
    lower_k: int,
    upper_k: int,
    columns_names: list,
    k: float = 0.95,
    p: float = 0.1,
    data_set_name: str = "dataset_1",
):
    pop = population_creation(
        num_population=num_population,
        lower_k=lower_k,
        upper_k=upper_k,
        data_tuple=data_tuple,
        data_set_name=data_set_name,
        columns_names=columns_names,
        knowledge=k,
    )
    pop.partition = p

    # Avaliação Inicial
    for indiv in pop.individuals:
        pop.calculate_fitness(individual=indiv)

    pop.dataframe()
    pop.sort(t_sort="fitness")
    
    best_score = pop.individuals[0].score
    best_individual_features = pop.individuals[0].features.copy()

    while pop.nfe < nfe_total and pop.len > 12:
        pop.dimension_classification(nfe_total=nfe_total)
        pop.beginner_gsk()
        pop.intermediate_gsk()
        
        for indiv in pop.individuals:
            pop.calculate_fitness(indiv)
            
        pop.dataframe()
        pop.sort(t_sort="fitness")
        
        # Guarda o melhor globalmente
        current_best = pop.individuals[0]
        if current_best.score < best_score:
            best_score = current_best.score
            best_individual_features = current_best.features.copy()
            
        # Reduz a população, se necessário
        pop.population_reduction(nfe_total=nfe_total)

    return pop, best_individual_features, best_score