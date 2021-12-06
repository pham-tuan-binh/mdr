import json

class Config:
    '''
    Config is a class used to import variables from a JSON config file at runti$

    Attributes:
        propSpin_port (int): coefficient indicating which direction the port pr$
                             Acceptable values are +1 or -1
        propSpin_star (int): coefficient indicating which direction the starboa$
                             Acceptable values are +1 or -1
        bias (float): Correction term to account for inequality between port an$
                      percentage between -100.0 and +100.0. Positive bias cause$
                      while negative bias causes a right (starboard) turn.
    '''

    def __init__(self, filepath):
        '''
        Creates a Config object, parses target JSON config file and stores its $

        Args:
            filepath (str): Filepath to a JSON config file to be parsed.
        '''
        self.propSpin_port = 1
        self.propSpin_star = 1
        self.bias = 0.0
        self.parse_configs(filepath)

    def parse_configs(self, filepath):
        '''
        Parses target JSON config file and stores its data in this object.

        Args:
            filepath (str): Filepath to a JSON config file to be parsed.
        '''
        with open(filepath, 'r') as jsonFile:
            config = json.load(jsonFile)

        self.propSpin_port = config['PropellerSpin']['port']
        self.propSpin_star = config['PropellerSpin']['starboard']
        self.bias = config['bias']
