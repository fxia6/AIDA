class LogisticRegressionModelRemoteStub(aidacommon.rop.RObjStub):

    @aidacommon.rop.RObjStub.RemoteMethod()
    def get_model(self):
        pass;
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def decision_function(self,X):
        pass;
 
    @aidacommon.rop.RObjStub.RemoteMethod()
    def fit(self,X,y,sample_weight=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_log_proba(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_proba(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def score(self,X,y,sample_weight=None):
        pass;

copyreg.pickle(LogisticRegressionModelRemoteStub, LogisticRegressionModelRemoteStub.serializeObj);


class LinearRegressionModelRemoteStub(aidacommon.rop.RObjStub):

    @aidacommon.rop.RObjStub.RemoteMethod()
    def get_model(self):
        pass;
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def fit(self,X,y,sample_weight=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def get_params(self,deep=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def score(self,X,y,sample_weight=None):
        pass;
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def set_params(self,**params):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def coef(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def rank(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def singular(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def intercept(self):
        pass;
 
copyreg.pickle(LinearRegressionModelRemoteStub, LinearRegressionModelRemoteStub.serializeObj);

class DecisionTreeModelRemoteStub(aidacommon.rop.RObjStub):
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def apply(self,X,check_input=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def cost_complexity_pruning_path(self,X,y,sample_weight=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def decision_path(self,X,check_input=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def fit(self,X,y,sample_weight=None,check_input=True,X_idx_sorted=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict(self,X,check_input=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_log_proba(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_proba(self,X,check_input=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def score(self,X,y,sample_weight=None):
        pass;

copyreg.pickle(DecisionTreeModelRemoteStub,DecisionTreeModelRemoteStub.serializeObj);

class HelloWorldRemoteStub(aidacommon.rop.RObjStub):
    @aidacommon.rop.RObjStub.RemoteMethod()
    def _helloWorld(self):
        pass;
copyreg.pickle(HelloWorldRemoteStub, HelloWorldRemoteStub.serializeObj);

class DBCRemoteStub(aidacommon.rop.RObjStub):
    
    class ModelRepositoryRemoteStub(aidacommon.rop.RObjStub):
        def __getattribute__(self,item):
            try:
                return object.__getattribute__(self,item)
            except:
                pass
            result = super().__getattribute__(item)

#            if (isinstance(result,aidacommon.rop.RObjStub)):
 #               self._registerProxy_(item,result.proxyid);
  #              super().__setattr__(item, result);
            return result
            
 @aidacommon.rop.RObjStub.RemoteMethod()
    def _helloWorld(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _linearRegression(self, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _logisticRegression(self, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _decisionTree(self,*args,**kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _save(self,model_name,model,update=False):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _load(self,model_name):
        pass 
