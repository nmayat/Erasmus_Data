# Erasmus_Data

 This is a project based on the data provided by the European Comission about the Erasmus+ project. The data can be found here https://data.europa.eu/data/datasets/erasmus-mobility-statistics-2014-2019-v2?locale=en.
 The project cleans tha data, analyses it and creates a recommendation engine based on a kNN algorithm. It is acompanied by a Medium blog post which can be found here: https://medium.com/@nils_88771/where-you-should-do-your-erasmus-program-76421269c585
 
 ## Data
 Unfortunaly the datasets are abit to big so you need to donwnload them yourself from the webside provided above. Place the files in a directory called 'Erasmus_Data'. The dataset is split in to csv files for each year and an explanatory file. In the Jupyter Notebook 'Erasmus_Data_cleaning' thedata cleaning steps are explained. Some of the data is than pickled for further use. 
 
 In the Jupyter notebook 'Erasmus_Data_exploration' the data is analyzed and visually depicted. Some of the plots are also used for the medium blog post. If you are interested in how to inculde plotly plots in a medium blog post, this helped me: https://jennifer-banks8585.medium.com/how-to-embed-interactive-plotly-visualizations-on-medium-blogs-710209f93bd. My plots can also be found here: https://chart-studio.plotly.com/~Mahuvej#/
 
 ## Recommendation Engine
 The steps for the recommendation engine is explained in the notebook 'Erasmus_Data_recommendation'. It is based on a kNN algorithm with k = 5. As a evaluation metric Mean Reciprocal Rank (MRR) was choosen (https://en.wikipedia.org/wiki/Mean_reciprocal_rank). The current model recieves a MRR of 0.412 after more than 5 hours of testing.
 
 ## Further Features
 - Parallising the testing process for more speed
 - Creating a web app for easier input:
   - Could include filtered inputs to choose features for recommendation
   - Differen plots. Maybe also with plots about different countries which user can choose
 - Different model. Maybe based on Deep Learning. 
