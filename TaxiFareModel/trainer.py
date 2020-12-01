# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y



    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # distance
        pipe_dist = Pipeline([
            ('distance', DistanceTransformer()),
            ('scaler', StandardScaler())])

        # time
        pipe_time = Pipeline([
            ('features', TimeFeaturesEncoder('pickup_datetime')),
            ('OneHot', OneHotEncoder())])

        # preproc
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preprocess_pipe = ColumnTransformer([
            ('dist', pipe_dist, dist_cols),
            ('time', pipe_time, time_cols)])

        # model pipeline
        self.pipeline = Pipeline([
            ('preprocessing', preprocess_pipe),
            ('regression', LinearRegression())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data(nrows=10_000)
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns=['fare_amount'])
    y = df.fare_amount
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainedpipe = Trainer(X_train, y_train)
    trainedpipe.run()
    # evaluate
    results = trainedpipe.evaluate(X_test, y_test)
    print(results)
