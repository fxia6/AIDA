import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import ast

# helper class and methods that convert TabularData Object to numpy arrays
class DataConversion:

    # a static function to identify whether a variable contains a numerical value
    def is_number(n):
        try:
            float(n)
        except ValueError:
            return False
        return True

    # extract_X and extract_y using .matrix instead of .cdata
    #def extract_X(TabularDataX):
    #   data_X=TabularDataX.matrix
    #   row_indices=list()

    #   for i in range(data_X.shape[0]):
    #       if DataConversion.is_number(data_X[i][0]):
    #           row_indices.append(i)

    #   X=data_X[row_indices,].transpose()
    #   return X

    #def extract_y(TabularDatay):
    #   data_y=TabularDatay.matrix
    #   i=0
    #   while (i<data_y.shape[0]):
    #        if DataConversion.is_number(data_y[i][0]):
    #           break
    #       else:
    #            i+=1
    #   y=data_y[i]
    #    return y
    
    # a static function to convert the TabularData containing X values into numpy matrix 
    def extract_X(TabularDataX):
        data_X=TabularDataX.cdata
        key_list=list()
        for key in data_X:
            key_list.append(key)

        # a list of indices of keys which contain numerical values in TabularDataX
        numerical_indices=list()
        for i in range(len(key_list)):
            # assume all values under one column are of the same type
            # if the first value of a column is numerical, then the column is a numerical column
            n=data_X.get(key_list[i])[0]
            if DataConversion.is_number(n):
                numerical_indices.append(i)
        # if TabularDataX does not have numerical columns
        if (len(numerical_indices)==0):
            raise ValueError("Error: No X values are numerical")

        # TabularDataX has numerical columns, then extract the numpy arrays as features 
        X=data_X.get(key_list[numerical_indices[0]]).reshape(-1,1)
        for index in range(1,len(numerical_indices)):
            # a matrix of shape (n_sample,n_feature)
            X=np.concatenate((X,data_X.get(key_list[numerical_indices[index]]).reshape(-1,1)),axis=1)

        return X

    # a static function to convert the TabularData containing y values into numpy array
    def extract_y(TabularDatay):
        data_y=TabularDatay.cdata
        count=0
        for key in data_y:
            n=data_y[key][0]
            if DataConversion.is_number(n):
                break
            else:
                count+=1

        # if TabularDataObject2 does not have numerical columns
        if count>=len(data_y):
            raise ValueError("Error: No y values are numerical")

        # TabularDataObject2 has numerical columns, then extract the numpy aray as label
        y=data_y.get(key)
        return y

class LogisticRegressionModel:
    
    # initialize a LogisticRegressionModel object with "model" attribute containing an actual LogisticRegression object from the sklearn module    
    def __init__(self,*args,**kwargs):
        self.model=LogisticRegression(*args,**kwargs)

    # a function that returns the actual LinearRegression object which the called LogisticRegressionModel object wraps around
    def get_model(self):
        return self.model

    def decision_function(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.decision_function(X)

    def fit(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        self.model.fit(X,y,sample_weight)
        return self

    def predict(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict(X)

    def predict_log_proba(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_log_proba(X)

    def predict_proba(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_proba(X)

    def score(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        return self.model.score(X,y,sample_weight)

    def __getattribute__(self,item):
        # if the called function/attribute does not require X,y tabularData conversion, get the attribute value by calling the function on the actual LogisticRegression model in skLearn module

        # check if this object has the requested attribute
        try:
            return super().__getattribute__(item)
        except:
            pass;
        # otherwise fetch it from the actual linear regression object
        return getattr(self.model,item)

copyreg.pickle(LogisticRegressionModel,LogisticRegressionModelRemoteStub.serializeObj);

class LinearRegressionModel:

    # initialize a LinearRegressionModel object with "model" attribute containing an actual LinearRegression object from the skLearn module
    def __init__(self,*args,**kwargs):
        self.model=LinearRegression(*args,**kwargs)

    # a function that returns the actual LinearRegression object which the called LinearRegressionModel object wraps around
    def get_model(self):
        return self.model

    def fit(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        self.model.fit(X,y,sample_weight)
        return self
    
    def get_params(self,deep=True):
        return self.model.get_params(deep)

    def predict(self,X):
        # if statement added to avoid converting TabularData twice when predict() is called inside score()
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict(X)

    def score(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)        
        return self.model.score(X,y,sample_weight)

    def set_params(self,**params):
        return self.model.set_params(**params)
    
    '''
    # for testing purposes
    def __getattribute__(self,item):
            logging.info("The function being called: "+str(item))
            if (item in ('fit','predict','model','get_model','score')):
                return super().__getattribute__(item)
    '''
    
    def __getattribute__(self,item):
        # if the called function/attribute does not require X,y tabularData conversion, get the attribute value by calling the function on the actual LinearRegression model in skLearn module
        
        # check if this object has the requested attribute
        try:
            return super().__getattribute__(item)
        except:
            pass;
        # otherwise fetch it from the actual linear regression object
        return getattr(self.model,item)        

        '''
        if (item not in ('model','get_model','extract_X','extract_y','fit','predict','score')):
            return getattr(self.model,item)
        # else, call the function/attribute defined in the local module
        else:
            return object.__getattribute__(self,item)
        '''

copyreg.pickle(LinearRegressionModel,LinearRegressionModelRemoteStub.serializeObj);	

class DecisionTreeModel:
    # initialize a DecisionTreeModel object with "model" attribute containing an actual DecisionTreeClassifier object from the skLearn module
    def __init__(self,*args,**kwargs):
        self.model = DecisionTreeClassifier(*args, **kwargs)

    def get_model(self):
        return self.model

    def apply(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.apply(X,check_input)

    def cost_complexity_pruning_path(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        return self.model.cost_complexity_pruning_path(X,y,sample_weight)        
    def decision_path(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.decision_path(X,check_input)
    
    def fit(self,X,y,sample_weight=None,check_input=True,X_idx_sorted=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        self.model.fit(X,y,sample_weight,check_input,X_idx_sorted)
        return self

    def predict(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict(X,check_input)

    def predict_log_proba(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_log_proba(X)

    def predict_proba(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_proba(X,check_input)

    def score(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        return self.model.score(X,y,sample_weight)

    def __getattribute__(self,item):
        try:
            return super().__getattribute__(item)
        except:
            pass;
        return getattr(self.model,item)

copyreg.pickle(DecisionTreeModel,DecisionTreeModelRemoteStub.serializeObj);

class HelloWorld(metaclass=ABCMeta):
    def _helloWorld(self):
        logging.info("Hello World")
copyreg.pickle(HelloWorld,HelloWorldRemoteStub.serializeObj);

class DBC(metaclass=ABCMeta):
    _dataFrameClass_ = None;

    class ModelRepository:
        def __init__(self,dbc):
            self.dbc = dbc

        def __getattribute__(self,item):
            try:
                return object.__getattribute__(self,item)
            except:
                pass
            logging.info("_load('{}')".format(item))
            m=self.dbc._load('{}'.format(item))
            logging.info(type(m))
            self.__setattr__(item,m)
            return object.__getattribute__(self,item)

    class SQLTYPE(Enum):
        SELECT=1; CREATE=2; DROP=3; INSERT=4; DELETE=5;
        
            def _helloWorld(self):
        hw=HelloWorld()
        return hw        

    def _linearRegression(self,*args,**kwargs):
        model=LinearRegressionModel(*args,**kwargs)
        return model

    def _logisticRegression(self,*args,**kwargs):
        model=LogisticRegressionModel(*args,**kwargs)
        return model

    def _decisionTree(self,*args,**kwargs):
        model=DecisionTreeModel(*args,**kwargs)
        return model

    def _save(self,model_name,model,update=False):
        
        # Code using MERGE 
        '''
        m = model.get_model()
        model_type=type(m)
        pickled_m = pickle.dumps(m)
        pickled_m = str(pickled_m)
        pickled_m = pickled_m.replace("'","''")
        pickled_m = pickled_m.replace("\\","\\\\")
        if (update==True):
            self._executeQry("MERGE INTO _sys_models_ AS to_update USING _sys_models_ AS models_update ON to_update.model_name = models._update.model_name WHEN MATCHED THEN DELETE;",sqlType=DBC.SQLTYPE.MERGE)
            self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
        # update==False
        else:
            try:
                self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
            except:
                raise Exception("There already exists a model in the database with the same model_name. Please set \'update\' to True to overwrite" )        
        '''   
        
        duplicate_exist = False
        logging.info("before exeucing query")
        # check if there is another model already saved with <model_name>
        temp = self._executeQry("SELECT COUNT(model) FROM _sys_models_ WHERE model_name='{}';".format(model_name))
        # if the above SELECT COUNT query returns integer not equal to 0
        if temp[0]['L3'][0]!=0:
            duplicate_exist = True
        logging.info("after executing query once , and check duplicate")
    
        # throw an error if update=False and there is another model already saved with <model_name>
        if (update==False and duplicate_exist==True):
            raise Exception("There already exists a model in the database with the same model_name. Please set \'update\' to True to overwrite" ) 
        # delete the model saved with <model_name> if update=True
        elif (update==True and duplicate_exist==True):
            self._executeQry("DELETE FROM _sys_models_ WHERE model_name='{}';".format(model_name),sqlType=DBC.SQLTYPE.DELETE)
        else:
            pass

        m = model.get_model()
        model_type = type(m)
        model_type = str(model_type)
        model_type = model_type.replace("'","''")

        pickled_m = pickle.dumps(m)
        pickled_m = str(pickled_m)
        pickled_m = pickled_m.replace("'","''")
        pickled_m = pickled_m.replace("\\","\\\\")
        self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
    
    def _load(self,model_name):

        unpickled_m = self._executeQry("SELECT model FROM _sys_models_ WHERE model_name = '{}';".format(model_name))
        try:
            unpickled_m = unpickled_m[0]['model'][0]
        except:
            raise Exception("no model with such name found.")

        model=pickle.loads(ast.literal_eval(unpickled_m))
        logging.info(type(model))
        # default as linear regression model
        model_wrapper = LinearRegressionModel()

        if (isinstance(model,sklearn.linear_model._logistic.LogisticRegression)):
            logging.info("I am here, model is type logistic")
            model_wrapper = LogisticRegressionModel()
        elif (isinstance(model,sklearn.tree.DecisionTreeClassifier)):
            model_wrapper = DecisionTreeModel()

        model_wrapper.model=model
        return model_wrapper
