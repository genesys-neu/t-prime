import numpy as np
import matlab.engine
import matlab

class MatlabEngine():
    def __init__(self, paths=[]):
        self.paths = paths
        self.eng = self.startup()


    def startup(self):
        """
        This method initialize Matlab engine and add the corresponding list of paths to the Matlab environment
        :param paths: a list of paths pointing to directories with Matlab code
        :return: the engine object
        """
        self.eng = matlab.engine.start_matlab()
        for p in self.paths:
            self.eng.addpath(p, nargout=0)
        return self.eng

    def getProperty(self, matlab_obj, property: str):
        """
        Retrieve a property from a matlab object. This property would be obfuscated from the Python environment.
        :param matlab_obj:
        :param property:
        :return:
        """
        self.eng.workspace['myObj'] = matlab_obj
        return self.eng.eval('myObj.'+property)

    def py2mat_array(self, x: np.ndarray):
        if np.iscomplexobj(x):
            return matlab.double(x.tolist(), is_complex=True)
        else:
            return matlab.double(x.tolist())
