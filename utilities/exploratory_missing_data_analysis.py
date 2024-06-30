# define functions for EMDA
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr


class sequential_EMDA:
    def __init__(self, path_data_frame, temporal_index, features = 'all'):
        self.df = pd.read_csv(path_data_frame)
        self.temporal_index = temporal_index

        if features == 'all':
            self.features = list(self.df.columns)
            self.features.remove(self.temporal_index)

        else:
            self.features == features

    # heat-map feature pearson correlation
    def plot_feature_correlation(self, name = 'Feature Correlation', save = False):
        # select features to plot

        df_no_na = self.df.dropna() # remove rows with na for correlation test
        
        # initialize and fill heatmap
        matrix = np.zeros((len(self.features), len(self.features))) 

        for i, feature1 in enumerate(self.features):
            for j, feature2 in enumerate(self.features):
                matrix[i, j] = round(pearsonr(df_no_na[feature1], df_no_na[feature2])[0],2) # correlation test 

        # Create a heatmap using seaborn
        plt.figure(figsize=(10, 9))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=self.features, yticklabels=self.features)
        plt.title(f'{name} -- mean corrleation: {round(matrix.mean(),2)}', size = 16)
        if save:
            plt.savefig(f"figures/{name.replace(' ','_')}.png", dpi = 180)
        plt.show()


    # na_per_feature
    def na_per_feature(self, df):
        print('N missing data per feature:')
        return df.isna().sum()
    
    # n missing at timestep
    def missing_together(self):
        na_series_list = []
        na_series_dict = {}

        for n in range(len(self.df)):
            if self.df.loc[n].isna().sum() > 0:
                na_series_list.append(self.df.loc[n].isna().sum())

        for n in range(1, max(na_series_list) +1 ):
            na_series_dict[n] = na_series_list.count(n)
        print('number of feature missing at same timestep {n(features): n(time steps)}')
        return na_series_dict
    
    # len_series_missing
    def len_series_missing(self):
        series_missing_dict = {}

        for feature in self.df[self.features]:

            # initialize list to store the number of directly following missing values for current feature
            current_series_list = []
            # list of index (time stamp) for missing value
            index_missing_list = list(self.df[self.df[feature].isna()][self.temporal_index])
            # 
            
            count = 1
            for n in range(1, len(index_missing_list)):
                
                if index_missing_list[n-1] + 1 == index_missing_list[n]:
                    count += 1
                else:
                    current_series_list.append(count)
                    count = 1
            
            current_series_list.append(count)

            current_series_list.sort(reverse= True)
            series_missing_dict[feature] = list(current_series_list)
            
        return series_missing_dict


    # plot missing data over time
    def plot_na(self, title = 'Missing Data over Time Series', save = False):         

        colors = ['b','g','r','c','m','y','k','orange','purple','brown','pink']

        plt.figure(figsize=(len(self.features) +3 , 3)) 

        for n in range(len(self.features)):
            current_missing = self.df[self.df[self.features[n]].isna()][self.temporal_index]
            missing_series = pd.Series([1] * len(current_missing), index=current_missing)
            plt.plot(missing_series.index, missing_series.values + (n*0.1), 'x', label=f'{self.features[n]} (N={len(current_missing)})', color = colors[n])

        plt.xlabel('Date')
        plt.title(title)
        plt.legend(fontsize = 9, ncols = 2 )
        plt.tight_layout()
        if save:
            plt.savefig('figures/na_over_time.png', dpi = 180)
        plt.show()

    # proportion missing together 
    def prop_missing_together(self, feature1, feature2):
        missing_1 = list(self.df[self.df[feature1].isna()][self.temporal_index])
        missing_2 = list(self.df[self.df[feature2].isna()][self.temporal_index])

        not_missing_together = []

        for datapoint in missing_1:
            if datapoint not in missing_2:
                not_missing_together.append(datapoint)

        for datapoint in missing_2:
            if datapoint not in missing_1:
                not_missing_together.append(datapoint)

        not_missing_together = round(len(not_missing_together)/(len(missing_1)+len(missing_2)),3)
        return 1 -not_missing_together
    
    def plot_prop_na_together(self, save = False):
        
        matrix = np.zeros((len(self.features), len(self.features))) # initalize matrix

        for i, feature1 in enumerate(self.features):
            for j, feature2 in enumerate(self.features):
                matrix[i, j] = self.prop_missing_together(feature1, feature2)

        # Create a heatmap using seaborn
        plt.figure(figsize=(10, 9))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=self.features, yticklabels=self.features)
        plt.title('Heatmap of Proportion of Missing Values Together', size = 14)
        if save:
            plt.savefig(f"figures/Heatmap_prop_missing_Together.png", dpi = 180)
        plt.show()
    