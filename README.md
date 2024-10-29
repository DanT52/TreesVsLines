# TreesVsLines

i need to make 4 models

Linear regression
Random Forest
Decision Tree
Gradient Boosting Regressor

cannotuse neiborhood feature.

i will ahve 4 grids, each grid excpet linear regression must have at least 2 keys besides colum selection.
linear regression only needs one key...

sklearn.tree.plot_tree( the_tree ) : to visualize
tree = DecisionTreeClassifier(criterion = 'entropy’ )
tree.fit( xs, ys )

grid = {
'max_depth': range(1, 10), #this is the parameter we want
}
search = GridSearchCV( tree, grid, cv = 10 )
search.fit( plants_x, plants_y )

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( max_depth = 2, n_estimators = 50 );

dr1 = DecisionTreeRegressor( max_depth = 2 )
dr1.fit( xs, ys )
p1 = dr1.predict( xs 


use scaling as another pipeline step